import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys and External Services
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
GROK_MODEL = os.getenv("GROK_MODEL", "x-ai/grok-4.1-fast") # Default to fast model if not specified
# Support both keys as per user environment
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY2")

# Project Defaults
MARKET_INDEX = "^GSPC"  # S&P 500
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"
NEWS_SOURCE = "tavily" # Default to tavily since we have the key

# Validation
if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY not found in environment variables.")

if not TAVILY_API_KEY:
    print("WARNING: TAVILY_API_KEY not found in environment variables.")
