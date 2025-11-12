"""
Configuration for Kosmos Framework
"""

# API Configuration
# Use environment variable OPENAI_API_KEY instead of hardcoding
OPENAI_API_KEY = None
MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0.7

# Framework Settings
MAX_CYCLES = 5
MAX_PARALLEL_TASKS = 3
OUTPUT_DIR = "outputs"

# Data Settings
DATA_DIR = "data"
LITERATURE_DIR = "knowledge/literature"

# Agent Settings
MAX_CODE_RETRIES = 3
CODE_TIMEOUT = 300  # seconds
