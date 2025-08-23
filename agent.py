from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Any, Dict
import numpy as np
import json
import logging
import os

# Import your existing modules
from rag import RAGVectorDB, EEGGeminiAgent
from eeg_assessment import analyze_patient_eeg, format_patient_assessment, edf_to_numpy
import google.generativeai as genai
from settings_service import SettingsService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simplified state definition for EDF files only
class AgentState(TypedDict):
    user_query: str
    uploaded_file_path: Optional[str]  # Path to uploaded EDF file
    eeg_data: Optional[np.ndarray]
    sampling_rate: int
    channel_names: Optional[List[str]]
    needs_rag: bool
    needs_eeg: bool
    rag_results: Optional[str]
    eeg_results: Optional[Dict]
    final_response: Optional[str]
    error_message: Optional[str]
    conversation_history: List[Dict[str, str]]
    session_context: Dict[str, Any]

class SimplifiedEEGWorkflow:
    """Simplified LangGraph workflow for EEG analysis with EDF file upload support"""
    
    def __init__(self, rag_tool: RAGVectorDB, model_name: str = "gemini-2.0-flash"):
        # Initialize components
        self.rag_agent = EEGGeminiAgent(rag_tool, model_name)
        
        # Configure Gemini
        genai.configure(api_key=SettingsService().settings.google_api_key)
        self.model = genai.GenerativeModel(
            model_name,
            generation_config={"response_mime_type": "text/plain"}
        )
        
        # Conversation memory
        self.conversation_history = []
        self.session_context = {
            "last_eeg_analysis": None,
            "current_patient_context": None,
            "recent_topics": [],
            "current_edf_file": None  # Track current EDF file
        }
        
        # Build the workflow graph
        self.workflow = self._build_graph()
        self.app = self.workflow.compile()
    
    def _build_graph(self) -> StateGraph:
        """Build the simplified LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_query", self.analyze_query)
        workflow.add_node("execute_rag", self.execute_rag)
        workflow.add_node("execute_eeg", self.execute_eeg)
        workflow.add_node("synthesize_response", self.synthesize_response)
        
        # Set entry point
        workflow.set_entry_point("analyze_query")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "analyze_query",
            self.route_after_analysis,
            {
                "rag_only": "execute_rag",
                "eeg_only": "execute_eeg",
                "both": "execute_rag",  # Start with RAG for 'both' cases
                "direct": "synthesize_response"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_rag",
            self.route_after_rag,
            {
                "eeg_next": "execute_eeg",
                "synthesize": "synthesize_response"
            }
        )
        
        workflow.add_edge("execute_eeg", "synthesize_response")
        workflow.add_edge("synthesize_response", END)
        
        return workflow
    
    def _load_edf_file(self, file_path: str) -> Dict[str, Any]:
        """Load and parse EDF file"""
        try:
            if not os.path.exists(file_path):
                return {"success": False, "error": f"EDF file not found: {file_path}"}
            
            if not file_path.lower().endswith(('.edf', '.bdf')):
                return {"success": False, "error": f"File must be EDF format (.edf or .bdf), got: {file_path}"}
            
            logger.info(f"Loading EDF file: {file_path}")
            voltage_data, sampling_rate, channel_names = edf_to_numpy(file_path)
            
            file_info = {
                "file_path": file_path,
                "n_channels": voltage_data.shape[0],
                "n_timepoints": voltage_data.shape[1],
                "duration_seconds": voltage_data.shape[1] / sampling_rate,
                "sampling_rate": sampling_rate,
                "channel_names": channel_names
            }
            
            return {
                "success": True,
                "voltage_data": voltage_data,
                "sampling_rate": sampling_rate,
                "channel_names": channel_names,
                "file_info": file_info
            }
            
        except Exception as e:
            logger.error(f"Error loading EDF file: {e}")
            return {"success": False, "error": f"Failed to load EDF file: {str(e)}"}
    
    def analyze_query(self, state: AgentState) -> AgentState:
        """Analyze the user query and load EDF file if provided"""
        user_query = state["user_query"]
        uploaded_file = state.get("uploaded_file_path")
        has_direct_eeg_data = state["eeg_data"] is not None
        conversation_history = state.get("conversation_history", [])
        session_context = state.get("session_context", {})
        
        # Load EDF file if provided
        edf_loaded_successfully = False
        if uploaded_file and not has_direct_eeg_data:
            edf_result = self._load_edf_file(uploaded_file)
            if edf_result["success"]:
                state["eeg_data"] = edf_result["voltage_data"]
                state["sampling_rate"] = edf_result["sampling_rate"]
                state["channel_names"] = edf_result["channel_names"]
                
                # Update session context
                session_context = state.get("session_context", {})
                session_context["current_edf_file"] = edf_result["file_info"]
                state["session_context"] = session_context
                
                edf_loaded_successfully = True
                logger.info(f"EDF file loaded successfully: {edf_result['file_info']}")
            else:
                state["error_message"] = edf_result["error"]
                logger.error(f"EDF loading failed: {edf_result['error']}")
        
        # Now we have either direct EEG data or loaded EDF data
        has_eeg_data = state["eeg_data"] is not None
        
        try:
            # Build context from conversation history
            history_context = ""
            if conversation_history:
                recent_history = conversation_history[-3:]  # Last 3 interactions
                history_context = "Recent conversation:\n" + "\n".join([
                    f"User: {h['user']}\nAssistant: {h['assistant'][:150]}...\nTools used: RAG={h['used_rag']}, EEG={h['used_eeg']}"
                    for h in recent_history
                ])
            
            # Build session context
            context_info = ""
            if session_context.get("last_eeg_analysis"):
                context_info += "\n- Previous EEG analysis available in session context"
            if session_context.get("current_edf_file"):
                edf_info = session_context["current_edf_file"]
                context_info += f"\n- Current EDF file loaded: {edf_info['n_channels']} channels, {edf_info['duration_seconds']:.1f}s duration"
            if session_context.get("recent_topics"):
                context_info += f"\n- Recent topics discussed: {', '.join(session_context['recent_topics'])}"
            
            analysis_prompt = f"""
            You are an intelligent EEG analysis agent. Analyze this query and determine what tools you need to use.

            Current Query: "{user_query}"
            Has EEG data available: {has_eeg_data}
            EDF file uploaded: {uploaded_file is not None}
            EDF loaded successfully: {edf_loaded_successfully}
            
            {history_context}
            
            Session Context:{context_info}

            Available Tools (choose wisely):
            1. RAG (Knowledge Base) - Use when you need to:
               - Look up medical information, definitions, or clinical guidelines
               - Find research findings or literature references
               - Explain EEG concepts, normal values, or abnormal patterns
               - Provide educational content about EEG
               - Answer questions about EEG interpretation
            
            2. EEG Analysis - Use when you need to:
               - Analyze the loaded EDF data or provided EEG voltage data
               - Generate clinical assessments of brain state
               - Detect seizure activity or abnormal patterns in the data
               - Provide quantitative analysis of EEG signals
               - Give specific findings about the patient's EEG
            
            Decision Guidelines:
            - If the query asks for information/knowledge about EEG → use RAG
            - If the query wants analysis of the EEG data → use EEG Analysis (only if data is available)
            - If the query needs both knowledge AND analysis → use both tools
            - Consider conversation context - follow-up questions might reference previous analysis
            - If user uploaded EDF file and asks to "analyze", "assess", or "interpret" → likely needs EEG Analysis
            - If user asks "what is", "explain", "tell me about" → likely needs RAG
            - If unsure, err on the side of providing more comprehensive information
            
            Think through your reasoning, then respond with EXACTLY this format:
            REASONING: [Your reasoning for the decision]
            NEEDS_RAG: true/false
            NEEDS_EEG: true/false
            """
            
            response = self.model.generate_content(analysis_prompt)
            
            if response and response.text:
                response_text = response.text.strip()
                logger.info(f"LLM Analysis: {response_text}")
                
                # Extract reasoning for debugging
                reasoning_match = response_text.split("REASONING:")
                if len(reasoning_match) > 1:
                    reasoning = reasoning_match[1].split("NEEDS_RAG:")[0].strip()
                    logger.info(f"LLM Reasoning: {reasoning}")
                
                # Parse the structured response
                needs_rag = "NEEDS_RAG: true" in response_text
                needs_eeg = "NEEDS_EEG: true" in response_text and has_eeg_data
                
                state["needs_rag"] = needs_rag
                state["needs_eeg"] = needs_eeg
                
                logger.info(f"LLM Decision: RAG={needs_rag}, EEG={needs_eeg}")
                
            else:
                logger.warning("LLM returned no response, using conservative approach")
                # Conservative fallback: if we have EEG data and query mentions analysis, use EEG
                # Otherwise, default to RAG for knowledge queries
                query_lower = user_query.lower()
                analysis_keywords = ["analy", "assess", "interpret", "diagnos", "report", "finding"]
                knowledge_keywords = ["what", "how", "explain", "tell me", "definition", "normal", "abnormal"]
                
                has_analysis_intent = any(keyword in query_lower for keyword in analysis_keywords)
                has_knowledge_intent = any(keyword in query_lower for keyword in knowledge_keywords)
                
                if has_analysis_intent and has_eeg_data:
                    state["needs_rag"] = has_knowledge_intent  # Might need both
                    state["needs_eeg"] = True
                else:
                    state["needs_rag"] = True  # Default to providing knowledge
                    state["needs_eeg"] = False
                
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            # Conservative fallback
            query_lower = user_query.lower()
            state["needs_rag"] = True  # Default to knowledge lookup
            state["needs_eeg"] = has_eeg_data and ("analy" in query_lower or "assess" in query_lower)
            
        logger.info(f"Final decision: RAG={state['needs_rag']}, EEG={state['needs_eeg']}")
        return state
                
    def route_after_analysis(self, state: AgentState) -> str:
        """Route based on analysis results"""
        needs_rag = state["needs_rag"]
        needs_eeg = state["needs_eeg"]
        
        if needs_rag and needs_eeg:
            return "both"
        elif needs_rag:
            return "rag_only"
        elif needs_eeg:
            return "eeg_only"
        else:
            return "direct"
    
    def execute_rag(self, state: AgentState) -> AgentState:
        """Execute RAG query with enhanced error handling and None checking"""
        user_query = state["user_query"]
        conversation_history = state.get("conversation_history", [])
        session_context = state.get("session_context", {})
        edf_info = session_context.get("current_edf_file", {}) if session_context else {}
        
        try:
            logger.info("Executing RAG query...")
            
            # Enhanced query refinement considering EDF context
            search_refinement_prompt = f"""
            The user asked: "{user_query}"
            
            Additional context:
            - EDF file loaded: {bool(edf_info.get("file_path"))}
            - Number of channels: {edf_info.get("n_channels", "N/A")}
            - Recording duration: {edf_info.get("duration_seconds", "N/A")} seconds
            - Has EEG data for analysis: {state.get("eeg_data") is not None}
            
            You need to search a medical knowledge base about EEG. 
            What would be the most effective search query to find relevant information?
            
            Consider:
            - The user's specific question
            - The context of analyzing EDF files
            - What medical/clinical information would be most helpful
            - Key terms that would find relevant EEG literature
            - EEG interpretation guidelines and standards
            
            Respond with just the refined search query (no explanation):
            """
            
            # Query refinement with better error handling
            refined_query = user_query  # Default fallback
            try:
                refinement_response = self.model.generate_content(search_refinement_prompt)
                if refinement_response and hasattr(refinement_response, 'text') and refinement_response.text:
                    refined_query = refinement_response.text.strip()
                    if refined_query:  # Make sure it's not empty
                        logger.info(f"Refined search query: {refined_query}")
                    else:
                        refined_query = user_query
                else:
                    logger.warning("Query refinement returned empty response, using original query")
                    refined_query = user_query
            except Exception as e:
                logger.warning(f"Query refinement failed: {e}, using original query")
                refined_query = user_query
            
            # Execute the RAG search with comprehensive error handling
            logger.info(f"Executing RAG search with query: '{refined_query}'")
            
            try:
                # Check if RAG agent exists and is properly initialized
                if not hasattr(self, 'rag_agent') or self.rag_agent is None:
                    raise ValueError("RAG agent is not properly initialized")
                
                # Check if RAG agent has the required method
                if not hasattr(self.rag_agent, 'answer_question'):
                    raise ValueError("RAG agent does not have 'answer_question' method")
                
                # Execute the RAG query
                logger.info("Calling RAG agent...")
                rag_result = self.rag_agent.answer_question(refined_query)
                logger.info(f"RAG agent returned: {type(rag_result)} - {str(rag_result)[:100] if rag_result else 'None'}")
                
                # Handle different types of RAG responses
                if rag_result is None:
                    logger.warning("RAG agent returned None")
                    state["rag_results"] = "No relevant information found in the knowledge base for your query."
                    
                elif isinstance(rag_result, str):
                    if rag_result.strip():
                        state["rag_results"] = rag_result
                        logger.info("RAG query completed successfully with string result")
                    else:
                        state["rag_results"] = "The knowledge base search returned an empty response."
                        
                elif isinstance(rag_result, dict):
                    # Handle dictionary responses (some RAG systems return structured data)
                    if 'answer' in rag_result:
                        state["rag_results"] = str(rag_result['answer'])
                    elif 'response' in rag_result:
                        state["rag_results"] = str(rag_result['response'])
                    elif 'text' in rag_result:
                        state["rag_results"] = str(rag_result['text'])
                    else:
                        # Convert entire dict to string
                        state["rag_results"] = f"Knowledge base response: {str(rag_result)}"
                    logger.info("RAG query completed with dictionary result")
                    
                elif hasattr(rag_result, 'text'):
                    # Handle objects with text attribute
                    state["rag_results"] = str(rag_result.text) if rag_result.text else "No text content in RAG response."
                    logger.info("RAG query completed with object having text attribute")
                    
                else:
                    # Handle any other type by converting to string
                    logger.warning(f"RAG agent returned unexpected type: {type(rag_result)}")
                    state["rag_results"] = str(rag_result) if rag_result else "No relevant information found."
                
                # Final validation of the result
                if not state.get("rag_results") or state["rag_results"].strip() == "":
                    state["rag_results"] = "The knowledge base search completed but returned no readable content."
                
                logger.info("RAG processing completed successfully")
                    
            except AttributeError as attr_error:
                logger.error(f"RAG agent attribute error: {attr_error}")
                if "'NoneType' object has no attribute" in str(attr_error):
                    state["rag_results"] = "RAG agent initialization error. The knowledge base may not be properly loaded."
                else:
                    state["rag_results"] = f"RAG agent configuration error: {str(attr_error)}"
                    
            except Exception as rag_error:
                logger.error(f"RAG agent execution error: {rag_error}")
                # Provide more specific error messages based on the error type
                if "connection" in str(rag_error).lower():
                    state["rag_results"] = "Unable to connect to the knowledge base. Please check your connection settings."
                elif "timeout" in str(rag_error).lower():
                    state["rag_results"] = "Knowledge base search timed out. Please try with a simpler query."
                elif "not found" in str(rag_error).lower():
                    state["rag_results"] = "Knowledge base not found. Please ensure the RAG system is properly initialized."
                else:
                    state["rag_results"] = f"Knowledge base error: {str(rag_error)}"
                    
        except Exception as e:
            logger.error(f"Critical error in RAG query execution: {e}")
            state["rag_results"] = f"Critical error retrieving knowledge: {str(e)}"
            if not state.get("error_message"):
                state["error_message"] = str(e)
        
        # Final safety check
        if state.get("rag_results") is None:
            state["rag_results"] = "RAG query completed but no results were generated."
        
        return state

    # Also add this method to help debug RAG initialization issues
    def debug_rag_agent(self):
        """Debug method to check RAG agent status"""
        debug_info = {
            "has_rag_agent": hasattr(self, 'rag_agent'),
            "rag_agent_is_none": getattr(self, 'rag_agent', None) is None,
            "rag_agent_type": type(getattr(self, 'rag_agent', None)).__name__,
            "has_answer_question_method": hasattr(getattr(self, 'rag_agent', None), 'answer_question'),
        }
        
        if hasattr(self, 'rag_agent') and self.rag_agent is not None:
            try:
                # Try to get more info about the RAG agent
                debug_info["rag_agent_attributes"] = [attr for attr in dir(self.rag_agent) if not attr.startswith('_')]
            except:
                debug_info["rag_agent_attributes"] = "Unable to get attributes"
        
        logger.info(f"RAG Agent Debug Info: {debug_info}")
        return debug_info

    # Also update the __init__ method to add better RAG initialization logging
    def __init__(self, rag_tool: RAGVectorDB, model_name: str = "gemini-2.0-flash"):
        """Simplified LangGraph workflow for EEG analysis with EDF file upload support"""
        
        try:
            # Initialize components with validation
            if rag_tool is None:
                raise ValueError("RAG tool cannot be None")
            
            logger.info(f"Initializing EEG workflow with RAG tool: {type(rag_tool)}")
            self.rag_agent = EEGGeminiAgent(rag_tool, model_name)
            
            if self.rag_agent is None:
                raise ValueError("Failed to create RAG agent")
            
            logger.info(f"RAG agent created successfully: {type(self.rag_agent)}")
            
            # Validate RAG agent has required methods
            if not hasattr(self.rag_agent, 'answer_question'):
                raise ValueError("RAG agent missing required 'answer_question' method")
            
            # Configure Gemini
            genai.configure(api_key=SettingsService().settings.google_api_key)
            self.model = genai.GenerativeModel(
                model_name,
                generation_config={"response_mime_type": "text/plain"}
            )
            
            # Conversation memory
            self.conversation_history = []
            self.session_context = {
                "last_eeg_analysis": None,
                "current_patient_context": None,
                "recent_topics": [],
                "current_edf_file": None
            }
            
            # Build the workflow graph
            self.workflow = self._build_graph()
            self.app = self.workflow.compile()
            
            logger.info("EEG workflow initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize EEG workflow: {e}")
            raise
    
    def route_after_rag(self, state: AgentState) -> str:
        """Route after RAG execution"""
        return "eeg_next" if state["needs_eeg"] else "synthesize"
    
    def execute_eeg(self, state: AgentState) -> AgentState:
        """Execute EEG analysis on EDF data with proper error handling"""
        if state["eeg_data"] is None:
            state["eeg_results"] = {"error": "No EEG data available for analysis. Please upload an EDF file."}
            return state
        
        try:
            logger.info("Executing EEG analysis on EDF data...")
            
            # Check if the analysis function exists and model is available
            try:
                assessment = analyze_patient_eeg(
                    state["eeg_data"], 
                    state["sampling_rate"], 
                    state["channel_names"]
                )
                formatted_report = format_patient_assessment(assessment)
                
                state["eeg_results"] = {
                    "assessment": assessment,
                    "formatted_report": formatted_report
                }
                
                # Update session context with EEG analysis
                session_context = state.get("session_context", {})
                session_context["last_eeg_analysis"] = {
                    "assessment": assessment,
                    "edf_file": session_context.get("current_edf_file"),
                    "analysis_timestamp": "now"  # You can use datetime here
                }
                state["session_context"] = session_context
                
                logger.info("EEG analysis completed successfully")
                
            except FileNotFoundError as model_error:
                logger.error(f"EEG model not found: {model_error}")
                # Provide a fallback analysis based on basic signal properties
                fallback_analysis = self._create_fallback_eeg_analysis(state)
                state["eeg_results"] = fallback_analysis
                
            except Exception as analysis_error:
                logger.error(f"EEG analysis failed: {analysis_error}")
                # Provide basic file information as fallback
                edf_info = state.get("session_context", {}).get("current_edf_file", {})
                fallback_msg = f"""
EEG Analysis Error: {str(analysis_error)}

However, I can provide basic information about your EDF file:
- File: {edf_info.get('file_path', 'Unknown')}
- Channels: {edf_info.get('n_channels', 'Unknown')} 
- Duration: {edf_info.get('duration_seconds', 'Unknown')} seconds
- Sampling Rate: {edf_info.get('sampling_rate', 'Unknown')} Hz
- Channel Names: {', '.join(edf_info.get('channel_names', [])[:10])}{'...' if len(edf_info.get('channel_names', [])) > 10 else ''}

The EEG data has been loaded successfully, but the analysis model is not available. 
Please ensure the EEG analysis model is properly installed and the checkpoint file is available.
                """.strip()
                
                state["eeg_results"] = {
                    "error": str(analysis_error),
                    "fallback_info": fallback_msg
                }
                
        except Exception as e:
            logger.error(f"Error in EEG analysis: {e}")
            state["eeg_results"] = {"error": f"Error analyzing EEG data: {str(e)}"}
            if not state.get("error_message"):
                state["error_message"] = str(e)
        
        return state
    
    def _create_fallback_eeg_analysis(self, state: AgentState) -> Dict[str, Any]:
        """Create a basic fallback analysis when the full EEG model is not available"""
        try:
            eeg_data = state["eeg_data"]
            sampling_rate = state["sampling_rate"]
            channel_names = state.get("channel_names", [])
            edf_info = state.get("session_context", {}).get("current_edf_file", {})
            
            # Basic signal statistics
            voltage_range = np.ptp(eeg_data, axis=1)  # Peak-to-peak per channel
            voltage_std = np.std(eeg_data, axis=1)    # Standard deviation per channel
            mean_voltage = np.mean(eeg_data, axis=1)  # Mean voltage per channel
            
            # Simple frequency analysis (if scipy is available)
            try:
                from scipy import signal
                freq_analysis = {}
                for i, ch_name in enumerate(channel_names[:5]):  # Analyze first 5 channels
                    freqs, psd = signal.welch(eeg_data[i], sampling_rate, nperseg=min(1024, eeg_data.shape[1]//4))
                    dominant_freq = freqs[np.argmax(psd)]
                    freq_analysis[ch_name] = {
                        "dominant_frequency": round(dominant_freq, 2),
                        "power_in_alpha": np.sum(psd[(freqs >= 8) & (freqs <= 13)]),
                        "power_in_beta": np.sum(psd[(freqs >= 14) & (freqs <= 30)])
                    }
            except ImportError:
                freq_analysis = {"note": "Frequency analysis requires scipy"}
            
            fallback_report = f"""
BASIC EEG FILE ANALYSIS (Fallback Mode)
=====================================

FILE INFORMATION:
- File: {edf_info.get('file_path', 'Unknown')}
- Channels: {len(channel_names)} channels
- Duration: {edf_info.get('duration_seconds', 'Unknown')} seconds  
- Sampling Rate: {sampling_rate} Hz
- Total Data Points: {eeg_data.shape[1]:,}

BASIC SIGNAL PROPERTIES:
- Voltage Range: {np.min(voltage_range):.1e} to {np.max(voltage_range):.1e} V
- Mean Voltage Levels: {np.min(mean_voltage):.1e} to {np.max(mean_voltage):.1e} V
- Signal Variability: {np.min(voltage_std):.1e} to {np.max(voltage_std):.1e} V (std)

CHANNELS: {', '.join(channel_names)}

FREQUENCY ANALYSIS: {freq_analysis}

NOTE: This is a basic analysis only. The full EEG assessment model is not available.
For complete clinical analysis, please ensure the EEG model checkpoint is properly installed.
            """.strip()
            
            return {
                "assessment": {
                    "basic_stats": {
                        "voltage_range": voltage_range.tolist(),
                        "voltage_std": voltage_std.tolist(),
                        "mean_voltage": mean_voltage.tolist(),
                        "frequency_analysis": freq_analysis
                    },
                    "file_info": edf_info
                },
                "formatted_report": fallback_report,
                "is_fallback": True
            }
            
        except Exception as e:
            logger.error(f"Fallback analysis failed: {e}")
            return {
                "error": f"Both full analysis and fallback analysis failed: {str(e)}",
                "formatted_report": "Unable to analyze EEG data due to missing dependencies."
            }
    
    def synthesize_response(self, state: AgentState) -> AgentState:
        """Synthesize final response using all available information and conversation history"""
        user_query = state["user_query"]
        rag_results = state.get("rag_results", "")
        eeg_results = state.get("eeg_results", {})
        conversation_history = state.get("conversation_history", [])
        session_context = state.get("session_context", {})
        edf_info = session_context.get("current_edf_file", {})
        
        # Handle None values
        if rag_results is None:
            rag_results = ""
        if eeg_results is None:
            eeg_results = {}
        
        try:
            # Build synthesis prompt with EDF and conversation context
            synthesis_prompt = f"""
            Current User Query: "{user_query}"
            """
            
            # Add EDF file context
            if edf_info:
                synthesis_prompt += f"""
            
            EDF File Context:
            - File: {edf_info.get('file_path', 'Unknown')}
            - Channels: {edf_info.get('n_channels', 'N/A')} ({', '.join(edf_info.get('channel_names', [])[:8])})
            - Duration: {edf_info.get('duration_seconds', 'N/A')} seconds
            - Sampling Rate: {edf_info.get('sampling_rate', 'N/A')} Hz
            """
            
            # Add conversation context if available
            if conversation_history:
                recent_context = conversation_history[-2:]  # Last 2 exchanges for context
                context_str = "\n".join([
                    f"Previous Q: {h['user']}\nPrevious A: {h['assistant'][:200]}..."
                    for h in recent_context
                ])
                synthesis_prompt += f"""
            
            Recent Conversation Context:
            {context_str}
            """
            
            # Add session context
            if session_context.get("last_eeg_analysis"):
                synthesis_prompt += f"""
            
            Previous EEG Analysis Available: Yes (can reference if relevant to current question)
            """
            
            # Add RAG results if available
            if rag_results and not str(rag_results).startswith("Error"):
                synthesis_prompt += f"""
            
            Knowledge Base Information:
            {rag_results}
            """
            
            # Add EEG results if available
            if isinstance(eeg_results, dict) and not eeg_results.get("error"):
                if eeg_results.get("formatted_report"):
                    synthesis_prompt += f"""
            
            Current EEG Analysis Results:
            {eeg_results.get("formatted_report", "")}
            """
                elif eeg_results.get("fallback_info"):
                    synthesis_prompt += f"""
            
            EEG File Information (Analysis Model Unavailable):
            {eeg_results.get("fallback_info", "")}
            """
            elif isinstance(eeg_results, dict) and eeg_results.get("error"):
                if eeg_results.get("fallback_info"):
                    synthesis_prompt += f"""
            
            EEG Analysis Issue: {eeg_results['error']}
            
            Available Information:
            {eeg_results.get("fallback_info", "")}
            """
                else:
                    synthesis_prompt += f"""
            
            Note: EEG analysis could not be completed: {eeg_results['error']}
            """
            
            synthesis_prompt += """
            
            Instructions:
            - Provide a clear, professional response to the current question
            - Reference the EDF file context when relevant
            - Use available information (knowledge base, EEG analysis, previous context) to give the most helpful answer
            - If this is a follow-up question, connect it to previous context appropriately
            - Be specific about findings from the EDF file when available
            - Maintain continuity in the conversation while addressing the current query
            - Focus on practical, clinically relevant information
            """
            
            response = self.model.generate_content(synthesis_prompt)
            
            # Handle None response from Gemini
            if response is None or response.text is None:
                logger.warning("Gemini returned None response, using fallback")
                # Create fallback response with context awareness
                fallback_parts = []
                
                if isinstance(eeg_results, dict):
                    if eeg_results.get("formatted_report"):
                        fallback_parts.append(f"EEG Analysis Results from your EDF file:\n{eeg_results['formatted_report']}")
                    elif eeg_results.get("fallback_info"):
                        fallback_parts.append(f"EDF File Information:\n{eeg_results['fallback_info']}")
                    elif eeg_results.get("error"):
                        fallback_parts.append(f"EEG Analysis Error: {eeg_results['error']}")
                
                if rag_results and not str(rag_results).startswith("Error"):
                    fallback_parts.append(f"Relevant medical information:\n{rag_results}")
                
                if edf_info:
                    fallback_parts.append(f"Your EDF file has been loaded with {edf_info.get('n_channels', 'unknown')} channels and {edf_info.get('duration_seconds', 'unknown')} seconds of data.")
                
                if not fallback_parts:
                    fallback_parts.append("I'm ready to analyze your EDF file and answer EEG-related questions.")
                
                state["final_response"] = "\n\n".join(fallback_parts)
            else:
                state["final_response"] = response.text
                
        except Exception as e:
            logger.error(f"Error in response synthesis: {e}")
            # Create error response with available information
            error_msg = "I encountered an error while generating the final response."
            
            if isinstance(eeg_results, dict):
                if eeg_results.get("formatted_report"):
                    error_msg += f"\n\nHowever, here are your EEG analysis results:\n{eeg_results['formatted_report']}"
                elif eeg_results.get("fallback_info"):
                    error_msg += f"\n\nEDF file information:\n{eeg_results['fallback_info']}"
                elif eeg_results.get("error"):
                    error_msg += f"\n\nEEG analysis error: {eeg_results['error']}"
            
            if rag_results and not str(rag_results).startswith("Error"):
                error_msg += f"\n\nRelevant information from knowledge base:\n{rag_results}"
            
            if edf_info:
                error_msg += f"\n\nYour EDF file ({edf_info.get('file_path', 'unknown')}) has been loaded successfully."
                
            state["final_response"] = error_msg
        
        return state
    
    def process_query(self, 
                     query: str, 
                     uploaded_file_path: Optional[str] = None,
                     eeg_data: Optional[np.ndarray] = None,
                     sampling_rate: int = 250,
                     channel_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process a user query through the workflow with optional EDF file upload
        
        Args:
            query: User's question
            uploaded_file_path: Path to uploaded EDF file (if any)
            eeg_data: Direct EEG data (if any) - will be ignored if EDF file is provided
            sampling_rate: Sampling rate for direct EEG data
            channel_names: Channel names for direct EEG data
        """
        # Initialize state with conversation history
        initial_state = {
            "user_query": query,
            "uploaded_file_path": uploaded_file_path,
            "eeg_data": eeg_data,
            "sampling_rate": sampling_rate,
            "channel_names": channel_names,
            "needs_rag": False,
            "needs_eeg": False,
            "rag_results": None,
            "eeg_results": None,
            "final_response": None,
            "error_message": None,
            "conversation_history": self.conversation_history.copy(),
            "session_context": self.session_context.copy()
        }
        
        try:
            # Execute the workflow
            final_state = self.app.invoke(initial_state)
            
            # Update conversation history
            interaction = {
                "user": query,
                "assistant": final_state["final_response"],
                "used_rag": final_state["needs_rag"],
                "used_eeg": final_state["needs_eeg"],
                "had_edf_file": uploaded_file_path is not None,
                "had_eeg_data": final_state["eeg_data"] is not None,
                "timestamp": json.dumps({"timestamp": "now"})  # You can use datetime here
            }
            
            self.conversation_history.append(interaction)
            
            # Keep only last 10 interactions to prevent memory bloat
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            # Update session context
            if final_state.get("session_context"):
                self.session_context = final_state["session_context"]
            
            # Track recent topics for better context
            topics = self._extract_topics(query)
            self.session_context["recent_topics"] = list(set(
                self.session_context.get("recent_topics", []) + topics
            ))[-5:]  # Keep last 5 unique topics
            
            # Return structured response
            return {
                "query": query,
                "response": final_state["final_response"],
                "used_rag": final_state["needs_rag"],
                "used_eeg": final_state["needs_eeg"],
                "rag_results": final_state["rag_results"],
                "eeg_results": final_state["eeg_results"],
                "success": final_state["error_message"] is None,
                "error": final_state["error_message"],
                "conversation_length": len(self.conversation_history),
                "session_context": self.session_context.copy(),
                "edf_file_loaded": final_state.get("session_context", {}).get("current_edf_file") is not None
            }
            
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            
            # Still add to conversation history even on error
            error_interaction = {
                "user": query,
                "assistant": f"Error: {str(e)}",
                "used_rag": False,
                "used_eeg": False,
                "had_edf_file": uploaded_file_path is not None,
                "had_eeg_data": eeg_data is not None,
                "timestamp": json.dumps({"timestamp": "now"})
            }
            self.conversation_history.append(error_interaction)
            
            return {
                "query": query,
                "response": f"I'm sorry, I encountered an error processing your request: {str(e)}",
                "used_rag": False,
                "used_eeg": False,
                "rag_results": None,
                "eeg_results": None,
                "success": False,
                "error": str(e),
                "conversation_length": len(self.conversation_history),
                "session_context": self.session_context.copy(),
                "edf_file_loaded": False
            }

    def _extract_topics(self, query: str) -> List[str]:
        """Extract key topics from the query for context tracking"""
        topics = []
        query_lower = query.lower()
        
        # EEG-related topics
        eeg_topics = ["seizure", "epilepsy", "brain waves", "alpha", "beta", "theta", "delta", 
                     "voltage", "frequency", "channels", "sleep", "awake", "edf", "analysis"]
        for topic in eeg_topics:
            if topic in query_lower:
                topics.append(topic)
        
        return topics

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history"""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear the conversation history and session context"""
        self.conversation_history.clear()
        self.session_context = {
            "last_eeg_analysis": None,
            "current_patient_context": None,
            "recent_topics": [],
            "current_edf_file": None
        }
        logger.info("Conversation history and session context cleared")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session"""
        edf_info = self.session_context.get("current_edf_file", {})
        return {
            "total_interactions": len(self.conversation_history),
            "recent_topics": self.session_context.get("recent_topics", []),
            "has_eeg_analysis": self.session_context.get("last_eeg_analysis") is not None,
            "current_edf_file": edf_info.get("file_path") if edf_info else None,
            "edf_channels": edf_info.get("n_channels") if edf_info else None,
            "edf_duration": edf_info.get("duration_seconds") if edf_info else None,
            "conversation_preview": [
                {"user": h["user"][:50] + "...", "assistant": h["assistant"][:50] + "..."}
                for h in self.conversation_history[-3:]
            ]
        }

# Factory function to create the workflow
def create_simplified_eeg_workflow(cache_dir: str = "./rag_cache_eeg", 
                                 chroma_db_path: str = "./chroma_db") -> SimplifiedEEGWorkflow:
    """Create and return the simplified EEG workflow"""
    rag_tool = RAGVectorDB(cache_dir=cache_dir, chroma_db_path=chroma_db_path, skip_indexing=True)
    return SimplifiedEEGWorkflow(rag_tool)

# Test function
def test_edf_workflow():
    """Test the workflow with EDF file uploads"""
    try:
        print("Creating EDF workflow...")
        workflow = create_simplified_eeg_workflow()
        
        print("\n=== EDF Workflow Test ===")
        
        # Test 1: General EEG knowledge query
        print("1. User: What are normal EEG frequencies?")
        result1 = workflow.process_query("What are normal EEG frequencies?")
        print(f"   Assistant: {result1['response'][:200]}...")
        print(f"   Tools used - RAG: {result1['used_rag']}, EEG: {result1['used_eeg']}")
        
        # Test 2: with EDF file path
        print("\n2. User provides EDF file path: 'Please analyze chb01_01.edf'")
        result5 = workflow.process_query(
            "Please analyze this EEG recording",
            uploaded_file_path="data/chb01_01.edf"
        )
        print(f"   Assistant: {result5['response'][:200]}...")
        print(f"   Success: {result5['success']}")
        print(f"   Error (expected): {result5['error']}")
        
        # Show session summary
        print("\n=== Session Summary ===")
        summary = workflow.get_session_summary()
        print(f"Total interactions: {summary['total_interactions']}")
        print(f"Recent topics: {summary['recent_topics']}")
        print(f"Has EEG analysis: {summary['has_eeg_analysis']}")
        print(f"Current EDF file: {summary['current_edf_file']}")
        print(f"EDF channels: {summary['edf_channels']}")
        print(f"EDF duration: {summary['edf_duration']}")
        
        # Show conversation history
        print("\n=== Recent Conversation History ===")
        history = workflow.get_conversation_history()
        for i, interaction in enumerate(history[-3:], 1):  # Show last 3
            print(f"{i}. User: {interaction['user']}")
            print(f"   Assistant: {interaction['assistant'][:100]}...")
            print(f"   Tools - RAG: {interaction['used_rag']}, EEG: {interaction['used_eeg']}, EDF: {interaction.get('had_edf_file', False)}")
        
        print("\n=== Workflow Test Completed Successfully ===")
        
    except Exception as e:
        print(f"EDF workflow test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_edf_workflow()