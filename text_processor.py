from typing import List, Dict
import re
from langchain.schema import Document
from langchain.text_splitter import SpacyTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

class TextProcessor:
    def __init__(self):
        self.text_splitter = SpacyTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            pipeline="sentencizer"  # Use the sentencizer pipeline which is faster and doesn't require full spaCy models
        )
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep German umlauts
        text = re.sub(r'[^\w\s\däöüßÄÖÜ.,!?-]', '', text)
        # Normalize whitespace
        text = text.strip()
        return text
    
    def prepare_document(self, document: Document) -> Document:
        """Prepare a single document."""
        # Clean the text content
        cleaned_text = self.clean_text(document.page_content)
        
        # Update document with cleaned text
        document.page_content = cleaned_text
        
        # Add metadata if not present
        if 'source' not in document.metadata:
            document.metadata['source'] = 'unknown'
        if 'page' not in document.metadata:
            document.metadata['page'] = 0
            
        return document
    
    def split_document(self, document: Document) -> List[Document]:
        """Split a document into chunks."""
        return self.text_splitter.split_documents([document])
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process a list of documents."""
        processed_docs = []
        
        for doc in documents:
            # Clean and prepare the document
            prepared_doc = self.prepare_document(doc)
            # Split into chunks
            chunks = self.split_document(prepared_doc)
            processed_docs.extend(chunks)
            
        return processed_docs 