import os
from supabase import create_client, Client
import logging
from typing import Optional

class SupabaseService:
    def __init__(self):
        """
        Initialize Supabase client using environment variables
        """
        self.client: Optional[Client] = None
        self.logger = logging.getLogger(__name__)
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the Supabase client connection"""
        try:
            url = os.environ.get("SUPABASE_URL")
            key = os.environ.get("SUPABASE_ANON_KEY")

            # Check if environment variables are set
            self.logger.info(f"Supabase URL: {url}")
            self.logger.info(f"Supabase Key: {key}")
            
            if not url or not key:
                self.logger.error("Supabase credentials not configured")
                raise ValueError("Missing Supabase configuration")

            self.logger.info(f"Connecting to Supabase at: {url}")
            self.client = create_client(url, key)
            
            # Test connection by fetching a dummy record
            try:
                test = self.client.table('prompts').select("*").limit(1).execute()
                self.logger.debug(f"Supabase connection test successful: {test}")
            except Exception as test_error:
                self.logger.error(f"Supabase connection test failed: {str(test_error)}")
                raise
                
            self.logger.info("Supabase client initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing Supabase client: {str(e)}", exc_info=True)
            raise

    def save_prompt(
        self,
        session_id: str,
        prompt_text: str,
        response_text: Optional[str] = None,
        language_mode: str = "standard",
        is_einfache_sprache: bool = False
    ) -> bool:
        """
        Save a user prompt and optional response to Supabase
        
        Args:
            session_id: Unique chat session identifier
            prompt_text: The user's question text
            response_text: The bot's response (optional)
            language_mode: Language mode used
            is_einfache_sprache: Simple language flag
            
        Returns:
            bool: True if save was successful
        """
        if not self.client:
            self.logger.error("Supabase client not initialized")
            return False

        try:
            data = {
                "session_id": session_id,
                "prompt_text": prompt_text,
                "response_text": response_text,
                "language_mode": language_mode,
                "einfache_sprache": is_einfache_sprache
            }
            
            result = self.client.table("prompts").insert(data).execute()
            self.logger.debug(f"Saved prompt to Supabase: {result}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving prompt to Supabase: {str(e)}")
            return False