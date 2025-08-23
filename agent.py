from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Any, Dict
import numpy as np
import json
import logging

# Import your existing modules
from rag import RAGVectorDB, EEGGeminiAgent
from eeg_assessment import analyze_patient_eeg, format_patient_assessment
import google.generativeai as genai
from settings_service import SettingsService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simplified state definition
class AgentState(TypedDict):
    user_query: str
    eeg_data: Optional[np.ndarray]
    sampling_rate: int
    channel_names: Optional[List[str]]
    needs_rag: bool
    needs_eeg: bool
    rag_results: Optional[str]
    eeg_results: Optional[Dict]
    final_response: Optional[str]
    error_message: Optional[str]

class SimplifiedEEGWorkflow:
    """Simplified LangGraph workflow for EEG analysis without prebuilt dependencies"""
    
    def __init__(self, rag_tool: RAGVectorDB, model_name: str = "gemini-2.0-flash"):
        # Initialize components
        self.rag_agent = EEGGeminiAgent(rag_tool, model_name)
        
        # Configure Gemini
        genai.configure(api_key=SettingsService().settings.google_api_key)
        self.model = genai.GenerativeModel(
            model_name,
            generation_config={"response_mime_type": "text/plain"}
        )
        
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
    
    def analyze_query(self, state: AgentState) -> AgentState:
        """Analyze the user query to determine what tools are needed"""
        user_query = state["user_query"]
        has_eeg_data = state["eeg_data"] is not None
        
        # First try keyword-based analysis (more reliable)
        query_lower = user_query.lower()
        
        # Keywords for different needs
        rag_keywords = ["what", "explain", "define", "normal", "abnormal", "guideline", 
                       "research", "literature", "study", "clinical", "medical"]
        eeg_keywords = ["analyze", "assess", "seizure", "brain state", "voltage", 
                       "frequency", "amplitude", "pattern", "abnormal", "interpret"]
        
        # Keyword-based detection
        keyword_needs_rag = any(word in query_lower for word in rag_keywords)
        keyword_needs_eeg = has_eeg_data and any(word in query_lower for word in eeg_keywords)
        
        # Try LLM analysis as enhancement, but don't rely on it
        try:
            analysis_prompt = f"""
            Analyze this EEG-related query and determine what tools are needed.
            
            Query: "{user_query}"
            Has EEG data: {has_eeg_data}
            
            Reply with exactly this format - no other text:
            NEEDS_RAG: true/false
            NEEDS_EEG: true/false
            """
            
            response = self.model.generate_content(analysis_prompt)
            if response and response.text:
                response_text = response.text.strip()
                logger.info(f"LLM Analysis response: {response_text}")
                
                # Parse the structured response
                llm_needs_rag = "NEEDS_RAG: true" in response_text
                llm_needs_eeg = has_eeg_data and "NEEDS_EEG: true" in response_text
                
                # Combine keyword and LLM analysis
                state["needs_rag"] = keyword_needs_rag or llm_needs_rag
                state["needs_eeg"] = keyword_needs_eeg or llm_needs_eeg
            else:
                # Fall back to keyword analysis
                state["needs_rag"] = keyword_needs_rag
                state["needs_eeg"] = keyword_needs_eeg
                
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            # Use keyword-based analysis as fallback
            state["needs_rag"] = keyword_needs_rag
            state["needs_eeg"] = keyword_needs_eeg
        
        logger.info(f"Final analysis: RAG={state['needs_rag']}, EEG={state['needs_eeg']}")
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
        """Execute RAG query"""
        try:
            logger.info("Executing RAG query...")
            rag_result = self.rag_agent.answer_question(state["user_query"])
            state["rag_results"] = rag_result
            logger.info("RAG query completed successfully")
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            state["rag_results"] = f"Error retrieving knowledge: {str(e)}"
            state["error_message"] = str(e)
        
        return state
    
    def route_after_rag(self, state: AgentState) -> str:
        """Route after RAG execution"""
        return "eeg_next" if state["needs_eeg"] else "synthesize"
    
    def execute_eeg(self, state: AgentState) -> AgentState:
        """Execute EEG analysis"""
        if state["eeg_data"] is None:
            state["eeg_results"] = {"error": "No EEG data provided for analysis"}
            return state
        
        try:
            logger.info("Executing EEG analysis...")
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
            logger.info("EEG analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error in EEG analysis: {e}")
            state["eeg_results"] = {"error": f"Error analyzing EEG data: {str(e)}"}
            state["error_message"] = str(e)
        
        return state
    
    def synthesize_response(self, state: AgentState) -> AgentState:
        """Synthesize final response using all available information"""
        user_query = state["user_query"]
        rag_results = state.get("rag_results", "")
        eeg_results = state.get("eeg_results", {})
        
        # Handle None values
        if rag_results is None:
            rag_results = ""
        if eeg_results is None:
            eeg_results = {}
        
        try:
            # Build synthesis prompt
            synthesis_prompt = f"""
            User Query: "{user_query}"
            
            Please provide a comprehensive response based on the available information below.
            """
            
            # Add RAG results if available
            if rag_results and not str(rag_results).startswith("Error"):
                synthesis_prompt += f"""
            
            Knowledge Base Information:
            {rag_results}
            """
            
            # Add EEG results if available
            if isinstance(eeg_results, dict) and eeg_results.get("formatted_report") and not eeg_results.get("error"):
                formatted_report = eeg_results.get("formatted_report", "")
                synthesis_prompt += f"""
            
            EEG Analysis Results:
            {formatted_report}
            """
            elif isinstance(eeg_results, dict) and eeg_results.get("error"):
                synthesis_prompt += f"""
            
            Note: EEG analysis could not be completed: {eeg_results['error']}
            """
            
            synthesis_prompt += """
            
            Instructions:
            - Provide a clear, professional response to the user's question
            - Use the available information to give the most helpful answer possible
            - If both EEG analysis and knowledge base information are available, integrate them meaningfully
            - If there were any errors, acknowledge them but still provide helpful guidance where possible
            - Keep the response focused on the user's specific question
            """
            
            response = self.model.generate_content(synthesis_prompt)
            
            # Handle None response from Gemini
            if response is None or response.text is None:
                logger.warning("Gemini returned None response, using fallback")
                # Create fallback response
                if rag_results and not str(rag_results).startswith("Error"):
                    state["final_response"] = f"Based on the available information:\n\n{rag_results}"
                elif isinstance(eeg_results, dict) and eeg_results.get("formatted_report"):
                    state["final_response"] = f"EEG Analysis Results:\n\n{eeg_results['formatted_report']}"
                else:
                    state["final_response"] = "I'm sorry, I couldn't generate a complete response, but I've processed your query."
            else:
                state["final_response"] = response.text
                
        except Exception as e:
            logger.error(f"Error in response synthesis: {e}")
            # Create error response with available information
            error_msg = "I encountered an error while generating the final response."
            
            if rag_results and not str(rag_results).startswith("Error"):
                error_msg += f"\n\nHowever, I did find this relevant information from the knowledge base:\n{rag_results}"
            
            if isinstance(eeg_results, dict) and eeg_results.get("formatted_report"):
                error_msg += f"\n\nEEG Analysis Results:\n{eeg_results['formatted_report']}"
                
            state["final_response"] = error_msg
        
        return state
    
    def process_query(self, 
                     query: str, 
                     eeg_data: Optional[np.ndarray] = None,
                     sampling_rate: int = 250,
                     channel_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process a user query through the workflow
        """
        # Initialize state
        initial_state = {
            "user_query": query,
            "eeg_data": eeg_data,
            "sampling_rate": sampling_rate,
            "channel_names": channel_names,
            "needs_rag": False,
            "needs_eeg": False,
            "rag_results": None,
            "eeg_results": None,
            "final_response": None,
            "error_message": None
        }
        
        try:
            # Execute the workflow
            final_state = self.app.invoke(initial_state)
            
            return {
                "query": query,
                "response": final_state["final_response"],
                "used_rag": final_state["needs_rag"],
                "used_eeg": final_state["needs_eeg"],
                "rag_results": final_state["rag_results"],
                "eeg_results": final_state["eeg_results"],
                "success": final_state["error_message"] is None,
                "error": final_state["error_message"]
            }
            
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            return {
                "query": query,
                "response": f"I'm sorry, I encountered an error processing your request: {str(e)}",
                "used_rag": False,
                "used_eeg": False,
                "rag_results": None,
                "eeg_results": None,
                "success": False,
                "error": str(e)
            }

# Factory function to create the workflow
def create_simplified_eeg_workflow(cache_dir: str = "./rag_cache_eeg", 
                                 chroma_db_path: str = "./chroma_db") -> SimplifiedEEGWorkflow:
    """Create and return the simplified EEG workflow"""
    rag_tool = RAGVectorDB(cache_dir=cache_dir, chroma_db_path=chroma_db_path, skip_indexing=True)
    return SimplifiedEEGWorkflow(rag_tool)

# Test function
def test_simplified_workflow():
    """Test the simplified workflow"""
    try:
        workflow = create_simplified_eeg_workflow()
        
        # Test 1: Knowledge query only
        print("=== Test 1: Knowledge Query ===")
        result1 = workflow.process_query("What are normal EEG voltage ranges for adults?")
        print(f"Response: {result1['response']}")
        print(f"Used RAG: {result1['used_rag']}, Used EEG: {result1['used_eeg']}")
        print(f"Success: {result1['success']}")
        
        # # Test 2: General query
        # print("\n=== Test 2: General Query ===")
        # result2 = workflow.process_query("Hello, how can you help me?")
        # print(f"Response: {result2['response']}")
        # print(f"Used RAG: {result2['used_rag']}, Used EEG: {result2['used_eeg']}")
        
        # # Test 3: With synthetic EEG data (if available)
        # try:
        #     from eeg_assessment import generate_synthetic_eeg_data
        #     eeg_data, sr, channels = generate_synthetic_eeg_data()
            
        #     print("\n=== Test 3: EEG Analysis ===")
        #     result3 = workflow.process_query(
        #         "Analyze this EEG data and tell me about the patient's brain state",
        #         eeg_data=eeg_data,
        #         sampling_rate=sr,
        #         channel_names=channels
        #     )
        #     print(f"Response: {result3['response'][:200]}...")
        #     print(f"Used RAG: {result3['used_rag']}, Used EEG: {result3['used_eeg']}")
            
        # except ImportError:
        #     print("\n=== Test 3: Skipped (no synthetic data function) ===")
        
    except Exception as e:
        print(f"Test failed with error: {e}")

if __name__ == "__main__":
    test_simplified_workflow()