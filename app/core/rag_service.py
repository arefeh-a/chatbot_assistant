# app/core/rag_service.py
import logging
import asyncio
from typing import List, Dict, Optional, AsyncIterator
from pathlib import Path
import sys

# --- Path Correction for Direct Execution ---
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

# --- Core Application & Langchain Imports ---
# The service now only depends on the config for path settings and RAG parameters.
from app import config
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


class RAGService:
    """
    The core RAG engine. It is initialized via pure Dependency Injection,
    receiving fully configured model objects to perform its work.
    """
    def __init__(
        self,
        llm: BaseChatModel,
        llm_rewrite: BaseChatModel,
        embedding_model: Embeddings
    ):
        """
        Initializes the RAGService by injecting fully-formed components.
        """
        logger.info("Initializing RAGService with injected, ready-to-use models...")
        self.settings = config
        
        # Directly assign the injected, pre-configured models.
        self.llm = llm
        self.llm_rewrite = llm_rewrite
        self.embedding_model = embedding_model

        # Load other components that depend on the injected models.
        self._load_prompts()
        self._load_vector_store_and_retriever()

        if not all([self.embedding_model, self.vector_store, self.retriever, self.llm, self.llm_rewrite]):
            raise RuntimeError("RAGService initialization failed: a critical component is missing.")
        
        logger.info("RAGService initialized successfully.")

    def _load_prompts(self):
        """Loads prompt templates from files."""
        self.system_prompt_text = self.settings.SYSTEM_PROMPT_FILE_PATH.read_text(encoding="utf-8")
        self.query_rewrite_prompt_text = self.settings.QUERY_REWRITING_FILE_PATH.read_text(encoding="utf-8")

    def _load_vector_store_and_retriever(self):
        """Loads the FAISS vector store using the injected embedding model."""
        db_path = self.settings.VECTOR_DB_DIR
        index_name = self.settings.FAISS_INDEX_NAME
        logger.info(f"Loading FAISS vector store from: {db_path}")
        
        if not (db_path / f"{index_name}.faiss").exists():
            raise FileNotFoundError(f"FAISS index not found at {db_path}. Please run build_kb.py.")
            
        self.vector_store: FAISS = FAISS.load_local(
            folder_path=str(db_path),
            embeddings=self.embedding_model, # Uses the injected model
            index_name=index_name,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.settings.RETRIEVER_K}
        )

    async def get_response(self, query: str, history: List[Dict[str, str]]) -> AsyncIterator[str]:
        """Streams the RAG response token by token."""
        logger.info(f"Processing streaming query: '{query[:50]}...'")
        
        try:
            query_for_retrieval = await self._get_standalone_query(query, history)
            retrieved_docs = await self.retriever.ainvoke(query_for_retrieval)
            
            context = self._format_retrieved_docs(retrieved_docs)
            final_messages = self._prepare_final_messages(history, query, context)
            
            # Streams the final response using the injected LLM.
            async for chunk in self.llm.astream(final_messages):
                yield str(chunk.content)

        except Exception as e:
            logger.exception(f"Error during streaming RAG response generation: {e}")
            yield "I'm sorry, an error occurred while generating a response."

    async def _get_standalone_query(self, query: str, history: List[Dict[str, str]]) -> str:
        """Rewrites a follow-up query into a standalone query."""
        if not history: return query

        history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
        
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", self.query_rewrite_prompt_text),
            ("human", "Chat History:\n{history}\n\nFollow-up User Query:\n{query}\n\nStandalone Search Query:")
        ])
        rewrite_chain = rewrite_prompt | self.llm_rewrite | StrOutputParser()
        
        return await rewrite_chain.ainvoke({"history": history_str, "query": query})

    def _format_retrieved_docs(self, docs: List[Document]) -> str:
        """Formats retrieved documents into a single string."""
        if not docs: return "No relevant information was found."
        return "\n\n".join([f"--- Context from: {doc.metadata.get('source', 'N/A')} ---\n{doc.page_content}" for doc in docs])

    def _prepare_final_messages(self, history: List[Dict[str, str]], query: str, context: str) -> List[BaseMessage]:
        """Prepares the final list of messages for the LLM."""
        messages: List[BaseMessage] = [SystemMessage(content=self.system_prompt_text)]
        for msg in history:
            if msg["role"] == "user": messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant": messages.append(AIMessage(content=msg["content"]))
        
        final_prompt = f"Use the following context to answer the user's question.\n\nContext:\n{context}\n\nUser's Question: {query}"
        messages.append(HumanMessage(content=final_prompt))
        return messages



if __name__ == '__main__':
    # This block simulates how app/main.py will assemble and inject dependencies.
    from providers.openai_provider import OpenAILLMProvider, OpenAIEmbeddingProvider
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

    def assemble_dependencies():
       
        assert config.LLM_PROVIDER == config.Provider.OPENAI and config.EMBEDDING_PROVIDER == config.Provider.OPENAI
        
        llm_provider = OpenAILLMProvider()
        embedding_provider = OpenAIEmbeddingProvider()
        
        return llm_provider, embedding_provider

    async def run_test():
        print("--- RAGService Standalone Test (Pure Dependency Injection) ---")
        try:
            # --- 1. Assemble the Provider Instances ---
            llm_provider, embedding_provider = assemble_dependencies()

            # --- 2. Create the ready-to-use model instances ---
            llm_api_key, chat_model, rewrite_model = config.OPENAI_API_KEY, config.OPENAI_LLM_CHAT_MODEL, config.OPENAI_QUERY_REWRITE_MODEL
            embedding_api_key, embedding_model = config.OPENAI_API_KEY, config.OPENAI_EMBEDDING_MODEL
         
            main_llm_instance = llm_provider.get_chat_model(llm_api_key, chat_model, config.LLM_TEMPERATURE)
            rewrite_llm_instance = llm_provider.get_chat_model(llm_api_key, rewrite_model, config.QUERY_REWRITE_TEMPERATURE)
            embedding_instance = embedding_provider.get_embedding_model(embedding_api_key, embedding_model)

            # --- 3. Inject the final components into the RAGService ---
            rag_service = RAGService(
                llm=main_llm_instance,
                llm_rewrite=rewrite_llm_instance,
                embedding_model=embedding_instance
            )
            
            # --- 4. Run Test Conversation ---
            test_history = []
            queries_to_test = [("خرید بورسی چگونه است؟", "Standalone query")]
            
            for query_text, _ in queries_to_test:
                print(f"\nUSER QUERY: {query_text}")
                print("SinaBot: ", end="", flush=True)
                full_response = ""
                async for chunk in rag_service.get_response(query_text, test_history):
                    print(chunk, end="", flush=True)
                    full_response += chunk
                print()

        except Exception as e:
            logging.exception(f"Test failed: {e}")

    asyncio.run(run_test())
