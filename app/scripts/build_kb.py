# app/scripts/build_kb.py
import os
import sys
import logging
import json
from typing import List, Optional
from pathlib import Path

# --- Add project root to sys.path ---
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

# --- Configure Basic Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Core Application & Provider Imports ---
try:
    from app import config
    from providers.openai_provider import OpenAIEmbeddingProvider
    from providers.google_provider import GoogleEmbeddingProvider
except ImportError as e:
    logger.exception(f"CRITICAL: Failed to import modules: {e}")
    sys.exit(1)

try:
    from app.scripts.lib.data_loader import DataLoader
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_community.vectorstores import FAISS
except ImportError as e:
    logger.exception(f"CRITICAL: Failed to import a library: {e}")
    sys.exit(1)


def load_all_source_documents() -> Optional[List[Document]]:
    """Loads all documents from the source directory using the custom loader."""
    all_docs: List[Document] = []
    source_dir = config.SOURCE_DOCS_DIR
    logger.info(f"Step 1: Loading documents from: '{source_dir}'")
    if not source_dir.is_dir():
        logger.error(f"Source directory not found: '{source_dir}'.")
        return None

    for item_path in sorted(source_dir.glob("*.txt")):
        logger.info(f"  - Processing file: '{item_path.name}'")
        loader = DataLoader(file_path=item_path)
        all_docs.extend(loader.load())
    
    logger.info(f"Loaded a total of {len(all_docs)} topic-block document(s).")
    return all_docs


def save_documents_to_json(docs: List[Document]):
    """Saves the final list of documents to a JSON file for inspection."""
    if not docs: return
    logger.info(f"Step 2: Saving {len(docs)} documents to JSON at '{config.CHUNKED_DOCS_JSON_PATH}'")
    docs_as_dicts = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    try:
        config.CHUNKED_DOCS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(config.CHUNKED_DOCS_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(docs_as_dicts, f, ensure_ascii=False, indent=4)
        logger.info("Successfully saved documents to JSON.")
    except Exception as e:
        logger.exception("Error saving documents to JSON.")


def initialize_embedding_model() -> Optional[Embeddings]:
    """Initializes the embedding model based on the provider selected in config."""
    provider_name = config.EMBEDDING_PROVIDER
    logger.info(f"Step 3: Initializing embedding model from provider: '{provider_name.value}'")

    try:
        if provider_name == config.Provider.OPENAI:
            embedding_provider = OpenAIEmbeddingProvider()
            api_key = config.OPENAI_API_KEY
            model_name = config.OPENAI_EMBEDDING_MODEL
        elif provider_name == config.Provider.GOOGLE:
            embedding_provider = GoogleEmbeddingProvider()
            api_key = config.GOOGLE_API_KEY
            model_name = config.GOOGLE_EMBEDDING_MODEL
        else:
            logger.error(f"Unsupported embedding provider configured: {provider_name}")
            return None

        embedding_model = embedding_provider.get_embedding_model(
            api_key=api_key,
            model_name=model_name
        )
        logger.info(f"Successfully initialized '{model_name}' from {provider_name.value} provider.")
        return embedding_model

    except ValueError as e:
        logger.critical(f"Configuration error for {provider_name.value} provider: {e}")
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred during embedding model initialization: {e}")
        return None


def build_and_persist_vector_store(docs: List[Document], embeddings: Embeddings):
    """Creates and saves a FAISS vector store by processing documents in batches."""
    if not docs or not embeddings:
        logger.error("Missing documents or embedding model. Cannot build vector store.")
        return


    DOC_BATCH_SIZE = config.DOC_BATCH_SIZE
    
    logger.info(f"Step 4: Building FAISS vector store from {len(docs)} documents in batches of {DOC_BATCH_SIZE}...")

    vector_store: Optional[FAISS] = None
    total_docs_processed = 0

    for i in range(0, len(docs), DOC_BATCH_SIZE):
        batch = docs[i:i + DOC_BATCH_SIZE]
        logger.info(f"  - Processing batch {i // DOC_BATCH_SIZE + 1}/{(len(docs) + DOC_BATCH_SIZE - 1) // DOC_BATCH_SIZE} ({len(batch)} documents)...")
        
        try:
            if vector_store is None:
                # Create the vector store with the first batch
                vector_store = FAISS.from_documents(batch, embeddings)
                logger.info("    FAISS vector store initialized with the first batch.")
            else:
                # Add subsequent batches to the existing vector store
                vector_store.add_documents(batch)
                logger.info("    Added batch to existing FAISS vector store.")
            total_docs_processed += len(batch)
        except Exception as e:
            logger.exception(f"  An error occurred while creating/updating FAISS vector store with batch starting at index {i}.")
            logger.error(f"  Failed to process batch. {total_docs_processed}/{len(docs)} documents processed before failure.")
            return # Stop processing if a batch fails

    if vector_store is None:
        logger.error("FAISS vector store could not be initialized (e.g., first batch failed or no documents).")
        return

    logger.info(f"Successfully processed all {total_docs_processed} documents into FAISS vector store in memory.")

    try:
        output_dir = config.VECTOR_DB_DIR
        index_name = config.FAISS_INDEX_NAME
        output_dir.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(folder_path=str(output_dir), index_name=index_name)
        logger.info(f"FAISS vector store saved successfully to '{output_dir}'.")
    except Exception as e:
        logger.exception("Error saving final FAISS vector store.")


if __name__ == "__main__":
    logger.info("--- Knowledge Base Build Script START ---")

    # Step 1: Load documents
    source_documents = load_all_source_documents()
    if not source_documents:
        logger.critical("No documents loaded. Aborting build process.")
        sys.exit(1)
    
    # Step 2: Save for inspection
    save_documents_to_json(source_documents)

    # Step 3: Initialize the embedding model
    embedding_model_instance = initialize_embedding_model()
    if not embedding_model_instance:
        logger.critical("Failed to initialize embedding model. Aborting build process.")
        sys.exit(1)
        
    # Step 4: Build and persist the vector store
    build_and_persist_vector_store(source_documents, embedding_model_instance)

    logger.info("--- Knowledge Base Build Script END ---")