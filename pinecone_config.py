from typing import Dict
import os

# Pinecone Settings
PINECONE_CONFIG = {
    "api_key": os.getenv("PINECONE_API_KEY", ""),
    "environment": os.getenv("PINECONE_ENVIRONMENT", ""),
    "index_name": os.getenv("PINECONE_INDEX_NAME", "koalitionskompass"),
    "namespace": os.getenv("PINECONE_NAMESPACE", "default")
}

def get_pinecone_settings() -> Dict:
    """Get Pinecone settings."""
    return {
        "api_key": PINECONE_CONFIG["api_key"],
        "environment": PINECONE_CONFIG["environment"],
        "index_name": PINECONE_CONFIG["index_name"],
        "namespace": PINECONE_CONFIG["namespace"]
    } 