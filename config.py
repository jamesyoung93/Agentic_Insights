"""
Configuration for Kosmos Framework
"""

# API Configuration
OPENAI_API_KEY = "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"  # Replace with your OpenAI API key
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
