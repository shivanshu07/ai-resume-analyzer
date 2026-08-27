from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RESUME_DIR = DATA_DIR / "resume"
JD_DIR = DATA_DIR / "job_description"

# Output directories
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = BASE_DIR / "logs"

# LLM Configuration (Groq -- free tier, no credit card required)
# Get a key at https://console.groq.com/keys
#
# Groq's API is OpenAI-compatible, so the existing `openai`
# Python client is reused -- it's just pointed at Groq's
# base_url below instead of OpenAI's.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv(
    "GROQ_BASE_URL",
    "https://api.groq.com/openai/v1"
)

# llama-3.3-70b-versatile was DEPRECATED by Groq on June 17,
# 2026 and no longer works (404 model_not_found). Their docs
# point to openai/gpt-oss-120b as the direct replacement --
# still free-tier, comfortably within rate limits for
# single-request interactive use. Override via .env if you
# want a different free-tier model.
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")