from typing import Protocol, Any, Dict, Optional
from google import genai
from google.genai import types
from .config import Config

class Tool(Protocol):
    """Protocol that all analysis tools must implement"""
    name: str
    description: str
    
    async def analyze(self, input_path: str) -> Dict[str, Any]:
        """Perform analysis on the input and return structured data"""
        ...

class JudgeAgent:
    def __init__(self):
        Config.validate()
        self.client = genai.Client(api_key=Config.GOOGLE_API_KEY)
        # Try different model name formats - common ones are:
        # "gemini-1.5-flash-latest", "gemini-2.0-flash-exp", "gemini-1.5-pro"
        self.model_id = Config.GEMINI_MODEL or "gemini-1.5-flash-latest"
    
    async def evaluate(self, tool_output: Dict[str, Any], context_prompt: str) -> str:
        """
        Send tool output to LLM for evaluation
        """
        import json
        full_prompt = f"""
        {context_prompt}
        
        Here is the technical data from the computer vision analysis:
        {json.dumps(tool_output, indent=2)}
        
        Please provide your judging assessment based on the rules provided.
        """
        
        # Try multiple model name formats
        model_names_to_try = [
            self.model_id,
            f"{self.model_id}-latest" if not self.model_id.endswith("-latest") else self.model_id,
            "gemini-1.5-flash-latest",
            "gemini-2.0-flash-exp",
            "gemini-1.5-pro-latest"
        ]
        
        last_error = None
        for model_name in model_names_to_try:
            try:
                response = await self.client.aio.models.generate_content(
                    model=model_name,
                    contents=full_prompt
                )
                return response.text
            except Exception as e:
                last_error = e
                if "404" not in str(e) and "not found" not in str(e).lower():
                    # If it's not a 404, it's a different error, raise it
                    raise
                continue
        
        # If all models failed, raise the last error with helpful message
        raise RuntimeError(
            f"Could not find a valid Gemini model. Tried: {model_names_to_try}. "
            f"Last error: {last_error}. "
            f"Please check your API key and available models."
        ) from last_error
