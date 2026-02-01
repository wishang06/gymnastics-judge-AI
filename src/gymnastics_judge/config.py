import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    # Default to a currently supported Gemini model id (the old "gemini-1.5-*-latest" aliases
    # are not available for all API keys / API versions and can 404).
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Optional: legacy path to dance_judge project (rhythmic tools now use in-project engine only)
    _default_rhythmic_root = os.path.join(os.path.expanduser("~"), "Desktop", "dance_judge", "my_dance_project")
    RHYTHMIC_RULES_PROJECT_ROOT = os.getenv(
        "RHYTHMIC_RULES_PROJECT_ROOT",
        os.getenv("DANCE_PROJECT_ROOT", _default_rhythmic_root),
    )

    @classmethod
    def validate(cls):
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
