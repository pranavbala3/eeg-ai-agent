import chromadb
from chromadb.utils import embedding_functions
import pickle
import os
import logging
from typing import List, Dict, Any
import google.generativeai as genai
from settings_service import SettingsService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Vector DB Tool
# ----------------------------
class RAGVectorDB:
    def __init__(self, cache_dir: str = "./rag_cache_eeg", chroma_db_path: str = "./chroma_db", skip_indexing: bool = True):
        self.cache_dir = cache_dir
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.client.get_or_create_collection(
            name="rag-docs",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
        self.skip_indexing = skip_indexing

    def _load_pickle(self, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    def _get_pickle_files(self) -> List[str]:
        return [
            os.path.join(self.cache_dir, f)
            for f in os.listdir(self.cache_dir)
            if f.endswith("_state.pickle")
        ]

    def index_all_pickles(self):
        for pickle_file in self._get_pickle_files():
            self._index_pickle(pickle_file)

    def _index_pickle(self, pickle_path: str):
        if self.skip_indexing:
            logger.info(f"[SKIP_ALL] Skipping indexing for {pickle_path} due to skip_indexing=True")
            return
        
        # Load the pickle
        with open(pickle_path, "rb") as f:
            doc_state = pickle.load(f)

        doc_name = os.path.basename(pickle_path)

        # Check if this document is already indexed
        try:
            existing = self.collection.get(where={"document_name": doc_name}, limit=1)
            if existing.get("ids"):
                logger.info(f"[SKIP] {doc_name} is already indexed, skipping.")
                return
        except Exception as e:
            logger.warning(f"Error checking existing entries for {doc_name}: {e}")

        # Add entries to collection
        ids, texts, metadatas = [], [], []
        for page_idx, page in enumerate(doc_state.extracted_layouts):
            for item_idx, item in enumerate(page.get("layout_items", [])):
                summary = item.get("summary", "").strip()
                if not summary:
                    continue

                layout_id = f"{doc_name}_{page_idx}_{item_idx}"
                ids.append(layout_id)
                texts.append(summary)
                metadatas.append({
                    "document_name": doc_name,
                    "page_number": page_idx,
                    "element_type": item.get("element_type", ""),
                    "section": item.get("section", "")
                })

        if ids:
            self.collection.add(ids=ids, documents=texts, metadatas=metadatas)
            logger.info(f"[INDEXED] {doc_name} with {len(ids)} items.")
        else:
            logger.info(f"[EMPTY] {doc_name} has no indexable items, skipping.")


    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            retrieved_docs = results.get("documents", [[]])[0]
            retrieved_metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            combined_results = []
            for doc, meta, dist in zip(retrieved_docs, retrieved_metas, distances):
                combined_results.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": dist
                })

            return {"query": query_text, "results": combined_results}

        except Exception as e:
            logger.error(f"Error querying vector DB: {e}")
            return {"query": query_text, "results": [], "error": str(e)}

# ----------------------------
# Agent using the vector DB as a tool
# ----------------------------
class EEGAgent:
    def __init__(self, rag_tool: RAGVectorDB):
        self.rag_tool = rag_tool

    def answer_question(self, question: str):
        # Use the vector DB tool for retrieval
        retrieved = self.rag_tool.query(question, n_results=5)

        if not retrieved.get("results"):
            return "No relevant information found in RAG cache."

        # Combine retrieved summaries as context
        context = "\n\n".join([f"[{i+1}] {r['text']}" for i, r in enumerate(retrieved["results"])])

        # Here you can call any LLM model to generate the final answer using the context
        # For simplicity, we just return the context + question
        return f"Question: {question}\n\nContext retrieved from RAG cache:\n{context}"

# ----------------------------
# Test function
# ----------------------------
def test_rag_agent():
    rag_tool = RAGVectorDB(cache_dir="./rag_cache_eeg", chroma_db_path="./chroma_db")
    rag_tool.index_all_pickles()

    agent = EEGAgent(rag_tool)
    question = "What is the normal voltage for an adult?"
    answer = agent.answer_question(question)
    print(answer)


# ----------------------------
# EEG Agent with Gemini LLM
# ----------------------------
class EEGGeminiAgent:
    def __init__(self, rag_tool: RAGVectorDB, model_name: str = "gemini-2.0-flash"):
        self.rag_tool = rag_tool
        self.model_name = model_name

        # Configure Gemini
        genai.configure(api_key=SettingsService().settings.google_api_key)
        self.answer_model = genai.GenerativeModel(
            self.model_name,
            generation_config={"response_mime_type": "text/plain"}
        )

    def answer_question(self, question: str) -> str:
        retrieved = self.rag_tool.query(question, n_results=5)
        if not retrieved.get("results"):
            return "No relevant information found in RAG cache."

        # Combine retrieved summaries as context
        context_text = "\n\n".join([f"[{i+1}] {r['text']}" for i, r in enumerate(retrieved["results"])])

        # Build messages for Gemini
        messages = [
            {"text": f"Use the following context to answer the question:\n\n{context_text}"},
            {"text": f"Question: {question}"}
        ]

        response = self.answer_model.generate_content(messages)
        return response.text


# ----------------------------
# Test function
# ----------------------------
def run_test_gemini_agent():
    rag_tool = RAGVectorDB(cache_dir="./rag_cache_eeg", chroma_db_path="./chroma_db", skip_indexing=True)
    rag_tool.index_all_pickles()  # Will skip indexing

    agent = EEGGeminiAgent(rag_tool)
    question = "Summarize all the information about EEGs."
    answer = agent.answer_question(question)
    print("\n=== Gemini LLM RAG Response ===")
    print(answer)


if __name__ == "__main__":
    run_test_gemini_agent()