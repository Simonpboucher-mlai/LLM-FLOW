import asyncio
import aiohttp
import nest_asyncio
import os
import logging
from dotenv import load_dotenv
from colorama import init, Fore, Style
from graphviz import Source
from IPython.display import display
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import json
import re
import tiktoken
import csv
import io
from fpdf import FPDF

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize colorama and nest_asyncio for Jupyter environments
init(autoreset=True)
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Check if API keys are set
api_keys = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "mistral": os.getenv("MISTRAL_API_KEY")
}

for api, key in api_keys.items():
    if not key:
        logging.warning(f"{api.upper()}_API_KEY is not set in the environment variables.")

def num_tokens_from_string(string: str, model_name: str) -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(string))

class APIException(Exception):
    """Custom exception for API-related errors."""
    pass
