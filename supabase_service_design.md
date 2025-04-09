# Supabase Service Module Design

## Overview
Module to handle all Supabase interactions for the Koalitionskompass application.

## File Location
`/supabase_service.py`

## Dependencies
```python
from supabase import create_client
import os
import logging
from typing import Optional
```

## Configuration
Will use secrets.toml with following structure:
```toml
[supabase]
url = "your-project-url"
anon_key = "your-anon-key"
```

## Class Definition
```python
class SupabaseService:
    def __init__(self):
        """
        Initialize Supabase client using secrets.toml configuration
        """
        self.client = None
        self.logger = logging.getLogger(__name__)
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the Supabase client connection"""
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_ANON_KEY")
        
        if not url or not key:
            self.logger.error("Supabase credentials not configured")
            raise ValueError("Missing Supabase configuration")

        self.client = create_client(url, key)

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
        try:
            data = {
                "session_id": session_id,
                "prompt_text": prompt_text,
                "response_text": response_text,
                "language_mode": language_mode,
                "einfache_sprache": is_einfache_sprache
            }
            self.client.table("prompts").insert(data).execute()
            return True
        except Exception as e:
            self.logger.error(f"Error saving prompt: {str(e)}")
            return False
```

## Integration Notes
1. Add to app.py:
```python
from supabase_service import SupabaseService
# Initialize in main()
supabase_service = SupabaseService()
```

2. Call in render_chat_interface():
```python
supabase_service.save_prompt(
    session_id=st.session_state.session_id,
    prompt_text=user_input,
    response_text=response,
    language_mode=st.session_state.active_tab,
    is_einfache_sprache=(st.session_state.active_tab == "simple")
)
```

## Error Handling
1. Connection errors should not break chat functionality
2. Failed saves should log but not interrupt user flow
3. Automatic retries for transient failures

## Testing Requirements
1. Unit tests for:
   - Client initialization
   - Successful saves
   - Error cases
2. Integration test with Supabase mock

## Next Steps
1. Create actual implementation in Code mode
2. Add to requirements.txt:
```
supabase>=2.0.0
```
3. Update deploy.sh to verify Supabase configuration