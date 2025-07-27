import os
import pickle
import pandas as pd
import faiss
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_rag_system(save_dir="rag_system_checkpoint"):
    """Load complete RAG system from saved files using absolute paths."""
    logger.info("Loading RAG system...")

    # Get the absolute path to the save_dir, relative to this file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, save_dir)
    
    logger.info(f"Loading from directory: {data_dir}")

    try:
        # Load passages list
        passages_path = os.path.join(data_dir, 'passages_list.pkl')
        if not os.path.exists(passages_path):
            raise FileNotFoundError(f"passages_list.pkl not found at {passages_path}")
        
        with open(passages_path, 'rb') as f:
            passages_list = pickle.load(f)
        logger.info(f"Loaded {len(passages_list)} passages")

        # Load passage IDs
        ids_path = os.path.join(data_dir, 'passage_ids.pkl')
        if not os.path.exists(ids_path):
            raise FileNotFoundError(f"passage_ids.pkl not found at {ids_path}")
            
        with open(ids_path, 'rb') as f:
            passage_ids = pickle.load(f)
        logger.info(f"Loaded {len(passage_ids)} passage IDs")

        # Load FAISS index
        faiss_path = os.path.join(data_dir, 'bioasq_faiss_index.index')
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"bioasq_faiss_index.index not found at {faiss_path}")
            
        faiss_index = faiss.read_index(faiss_path)
        logger.info(f"Loaded FAISS index with {faiss_index.ntotal} vectors")

        # Try to load config if it exists
        config_path = os.path.join(data_dir, 'system_config.json')
        config = None
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = f.read()
                logger.info("Loaded system configuration")
            except Exception as e:
                logger.warning(f"Could not load config: {e}")

        # Try to load LangChain vectorstore (optional)
        vectorstore = None
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS as LangChainFAISS
            
            # Create embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # Try to load existing vectorstore
            langchain_path = os.path.join(data_dir, 'langchain_vectorstore')
            if os.path.exists(langchain_path):
                vectorstore = LangChainFAISS.load_local(
                    langchain_path, 
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("✅ LangChain vectorstore loaded successfully!")
            else:
                logger.warning(f"LangChain vectorstore not found at {langchain_path}")
                
        except Exception as e:
            logger.warning(f"Could not load LangChain vectorstore: {e}")

        logger.info("✅ RAG system loaded successfully!")
        return faiss_index, passages_list, passage_ids, vectorstore, config

    except Exception as e:
        logger.error(f"Failed to load RAG system: {e}")
        logger.error(f"Make sure the following files exist in {data_dir}:")
        logger.error("  - passages_list.pkl")
        logger.error("  - passage_ids.pkl") 
        logger.error("  - bioasq_faiss_index.index")
        raise