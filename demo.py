#!/usr/bin/env python3
"""
EEG Analysis CLI Tool
A command-line interface for analyzing EEG data and answering EEG-related questions.
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors by default
    format='%(levelname)s: %(message)s'
)

def setup_imports():
    """Setup imports with helpful error messages"""
    try:
        from agent import create_simplified_eeg_workflow, SimplifiedEEGWorkflow
        return create_simplified_eeg_workflow, SimplifiedEEGWorkflow
    except ImportError as e:
        print(f"Error: Required modules not found: {e}")
        print("Please ensure all dependencies are installed and the EEG workflow module is available.")
        sys.exit(1)

class EEGCLITool:
    """Command-line interface for EEG analysis workflow"""
    
    def __init__(self, cache_dir: str = "./rag_cache_eeg", chroma_db_path: str = "./chroma_db", verbose: bool = False):
        """Initialize the CLI tool"""
        self.cache_dir = cache_dir
        self.chroma_db_path = chroma_db_path
        self.verbose = verbose
        self.workflow = None
        
        # Setup logging level based on verbose flag
        if verbose:
            logging.getLogger().setLevel(logging.INFO)
            print("Verbose mode enabled")
        
        self._initialize_workflow()
    
    def _initialize_workflow(self):
        """Initialize the EEG workflow"""
        try:
            print("Initializing EEG analysis workflow...")
            create_workflow_func, _ = setup_imports()
            self.workflow = create_workflow_func(
                cache_dir=self.cache_dir,
                chroma_db_path=self.chroma_db_path
            )
            print("✓ Workflow initialized successfully\n")
        except Exception as e:
            print(f"✗ Failed to initialize workflow: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    
    def validate_file_path(self, file_path: str) -> bool:
        """Validate if the file path exists and is an EDF file"""
        if not file_path:
            return True  # Empty path is valid (no file upload)
        
        path = Path(file_path)
        
        if not path.exists():
            print(f"✗ Error: File does not exist: {file_path}")
            return False
        
        if not path.is_file():
            print(f"✗ Error: Path is not a file: {file_path}")
            return False
        
        # Check file extension
        if not str(path).lower().endswith(('.edf', '.bdf')):
            print(f"⚠ Warning: File does not have .edf or .bdf extension: {file_path}")
            response = input("Continue anyway? [y/N]: ").lower().strip()
            if response not in ['y', 'yes']:
                return False
        
        return True
    
    def process_query(self, question: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Process a single query"""
        if not question.strip():
            print("✗ Error: Question cannot be empty")
            return {"success": False, "error": "Empty question"}
        
        # Validate file path if provided
        if file_path and not self.validate_file_path(file_path):
            return {"success": False, "error": "Invalid file path"}
        
        try:
            print("Processing your query...")
            if file_path:
                print(f"📁 File: {file_path}")
            print(f"❓ Question: {question}")
            print("-" * 50)
            
            # Process through workflow
            result = self.workflow.process_query(
                query=question,
                uploaded_file_path=file_path if file_path else None
            )
            
            return result
            
        except Exception as e:
            print(f"✗ Error processing query: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def display_result(self, result: Dict[str, Any]):
        """Display the result in a formatted way"""
        if not result["success"]:
            print(f"✗ Error: {result.get('error', 'Unknown error')}")
            return
        
        print("🤖 Assistant Response:")
        print("=" * 50)
        print(result["response"])
        print("=" * 50)
        
        # Show tool usage
        tools_used = []
        if result["used_rag"]:
            tools_used.append("Knowledge Base")
        if result["used_eeg"]:
            tools_used.append("EEG Analysis")
        
        if tools_used:
            print(f"🔧 Tools used: {', '.join(tools_used)}")
        
        # Show EDF file status
        if result.get("edf_file_loaded"):
            print("📁 EDF file loaded and processed")
        
        # Show conversation length
        print(f"💬 Conversation length: {result['conversation_length']} interactions")
        
        print()  # Add space before next interaction
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print("🧠 EEG Analysis Interactive Mode")
        print("=" * 40)
        print("Commands:")
        print("  • Type your question and press Enter")
        print("  • Type 'file <path>' to load an EDF file")
        print("  • Type 'clear' to clear conversation history")
        print("  • Type 'history' to see conversation history")
        print("  • Type 'summary' to see session summary")
        print("  • Type 'help' for this help message")
        print("  • Type 'quit' or 'exit' to exit")
        print("=" * 40)
        print()
        
        current_file = None
        
        while True:
            try:
                # Get user input
                if current_file:
                    prompt = f"[{Path(current_file).name}] EEG> "
                else:
                    prompt = "EEG> "
                
                user_input = input(prompt).strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    print("Commands:")
                    print("  file <path>  - Load an EDF file")
                    print("  clear        - Clear conversation history")
                    print("  history      - Show conversation history")
                    print("  summary      - Show session summary")
                    print("  help         - Show this help")
                    print("  quit/exit    - Exit the program")
                    continue
                
                elif user_input.lower() == 'clear':
                    self.workflow.clear_conversation_history()
                    current_file = None
                    print("🗑️ Conversation history cleared")
                    continue
                
                elif user_input.lower() == 'history':
                    self._show_history()
                    continue
                
                elif user_input.lower() == 'summary':
                    self._show_summary()
                    continue
                
                elif user_input.lower().startswith('file '):
                    file_path = user_input[5:].strip().strip('"\'')
                    if self.validate_file_path(file_path):
                        current_file = file_path
                        print(f"📁 File loaded: {current_file}")
                    continue
                
                # Process as a regular question
                result = self.process_query(user_input, current_file)
                self.display_result(result)
                
                # Update current file status based on result
                if result.get("edf_file_loaded"):
                    session_context = result.get("session_context", {})
                    edf_info = session_context.get("current_edf_file", {})
                    if edf_info.get("file_path"):
                        current_file = edf_info["file_path"]
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break
    
    def _show_history(self):
        """Show conversation history"""
        history = self.workflow.get_conversation_history()
        if not history:
            print("📝 No conversation history")
            return
        
        print(f"📝 Conversation History ({len(history)} interactions):")
        print("-" * 50)
        
        for i, interaction in enumerate(history, 1):
            print(f"{i}. User: {interaction['user']}")
            print(f"   Assistant: {interaction['assistant'][:100]}{'...' if len(interaction['assistant']) > 100 else ''}")
            tools = []
            if interaction['used_rag']:
                tools.append("RAG")
            if interaction['used_eeg']:
                tools.append("EEG")
            if interaction.get('had_edf_file'):
                tools.append("EDF")
            if tools:
                print(f"   Tools: {', '.join(tools)}")
            print()
    
    def _show_summary(self):
        """Show session summary"""
        summary = self.workflow.get_session_summary()
        print("📊 Session Summary:")
        print("-" * 30)
        print(f"Total interactions: {summary['total_interactions']}")
        print(f"Recent topics: {', '.join(summary['recent_topics']) if summary['recent_topics'] else 'None'}")
        print(f"Has EEG analysis: {'Yes' if summary['has_eeg_analysis'] else 'No'}")
        
        if summary['current_edf_file']:
            print(f"Current EDF file: {summary['current_edf_file']}")
            print(f"Channels: {summary['edf_channels']}")
            print(f"Duration: {summary['edf_duration']:.1f} seconds" if summary['edf_duration'] else 'Unknown')
        else:
            print("Current EDF file: None")
        
        print()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="EEG Analysis CLI Tool - Analyze EEG data and answer questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Interactive mode
  %(prog)s "What are normal EEG frequencies?" # Single question
  %(prog)s -f data.edf "Analyze this EEG"     # Question with file
  %(prog)s --interactive                      # Explicit interactive mode
  
File formats supported: .edf, .bdf
        """
    )
    
    # Arguments
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask (if not provided, enters interactive mode)"
    )
    
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Path to EDF file to analyze"
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Force interactive mode even with question provided"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./rag_cache_eeg",
        help="Directory for RAG cache (default: ./rag_cache_eeg)"
    )
    
    parser.add_argument(
        "--chroma-db",
        type=str,
        default="./chroma_db",
        help="Path for ChromaDB storage (default: ./chroma_db)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="EEG Analysis CLI v1.0"
    )
    
    args = parser.parse_args()
    
    # Initialize CLI tool
    cli = EEGCLITool(
        cache_dir=args.cache_dir,
        chroma_db_path=args.chroma_db,
        verbose=args.verbose
    )
    
    # Determine mode
    if args.interactive or not args.question:
        # Interactive mode
        cli.interactive_mode()
    else:
        # Single question mode
        result = cli.process_query(args.question, args.file)
        cli.display_result(result)


if __name__ == "__main__":
    main()