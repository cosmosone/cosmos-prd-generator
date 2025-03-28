#!/usr/bin/env python3
"""
Enhanced Dynamic Phased PRD Generator Script using Anthropic Messages API
"""

#######################################################################################
# DOCUMENTATION
#######################################################################################
"""
OVERVIEW:
This script generates a PRD (Product Requirements Document) with a dynamic number of phases
based on user input, using the Anthropic API.

The PRD is designed to be technology-agnostic by default, focusing on clean architecture,
design principles, and phased implementation. Technology-specific suggestions will be made
only if the user explicitly specifies a technology stack in their project goal.

WORKFLOW FOR IDE AI INTEGRATION:

1. Generate PRD using this script.
2. Locate the generated PRD directory (e.g., <project_name>_prd).
3. Open 'phase_00.md' and provide its content as the initial prompt to your IDE AI chat.
   - Wait for the IDE AI to acknowledge and understand the project instructions.
4. For each phase of implementation:
   - Open the corresponding 'phase_XX.md' file (e.g., 'phase_01.md').
   - Copy the ENTIRE content of the phase document.
   - Paste the content into the IDE AI chat interface.
   - For each implementation step in the phase document, ask the IDE AI for assistance,
     e.g., 'Help me implement step 1: ...'
   - Ensure that the IDE AI strictly adheres to the architectural guidelines defined in the project prompt AI rules.
   - Prioritize architectural compliance and code quality throughout the implementation.
   - Always verify the IDE AI's suggestions against the project architecture.
   - After each implementation step and at the end of each phase, run the application to verify correct implementation.
   - Before proceeding to the next step, confirm the completion of the current step and get IDE AI verification.
5. 'project_prompt.md' serves as a project-level configuration and architectural rulebook for the IDE AI.
   - It should be kept in the project directory for the IDE AI to reference throughout the development process.
"""

#######################################################################################
# IMPORTS
#######################################################################################
import os
import sys
import time
import threading
import logging
import argparse
import shutil
import tempfile
# Platform-specific imports for file locking
import platform
if platform.system() == 'Windows':
    import msvcrt
else:
    import fcntl
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import re
import json
import hashlib

import anthropic
from dotenv import load_dotenv

#######################################################################################
# GLOBAL CONSTANTS
#######################################################################################
# Constants for retry logic in API calls
MAX_RETRIES = 3
BASE_DELAY = 5  # seconds
DEFAULT_MAX_TOKENS = 4096  # Updated default for Claude-3
MAX_MODEL_TOKENS = 8192  # Maximum tokens allowed for Claude 3.5 Haiku
TOKENS_PER_CHAR = 0.3  # Rough estimation ratio for fallback calculation
TOKEN_SAFETY_MARGIN = 1.2  # 20% safety margin for token estimation
MAX_CACHE_SIZE_MB = 100  # 100MB max cache size
CACHE_RETENTION_DAYS = 30  # Retain cache entries for 30 days by default

#######################################################################################
# LOGGING SETUP
#######################################################################################
class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages."""
    COLORS = {
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[91m', # Red
        'INFO': '\033[92m',     # Green
        'DEBUG': '\033[94m',    # Blue
        'RESET': '\033[0m'      # Reset color
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        message = super().format(record)
        return f"{color}{message}{self.COLORS['RESET']}"

class CustomFilter(logging.Filter):
    def filter(self, record):
        if "Token estimation failed: Messages.count_tokens() takes 1 positional argument but 2 were given" in record.getMessage():
            return False
        return True

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger()
handler = logger.handlers[0]
handler.setFormatter(ColoredFormatter("%(levelname)s: %(message)s"))
handler.addFilter(CustomFilter())

#######################################################################################
# UTILITY CLASSES
#######################################################################################
class FileLock:
    """File locking mechanism for concurrent access."""
    def __init__(self, file_path):
        self.file_path = file_path
        self.lock_file = Path(f"{file_path}.lock")
        self.lock_fd = None
        self.is_windows = platform.system() == 'Windows'

    def __enter__(self):
        try:
            # Create lock file if it doesn't exist
            if not self.lock_file.exists():
                self.lock_file.touch()
            
            # Open the lock file and acquire an exclusive lock
            self.lock_fd = open(self.lock_file, 'w')
            
            if self.is_windows:
                # Windows file locking
                file_handle = msvcrt.get_osfhandle(self.lock_fd.fileno())
                # Lock from current position to end of file (0 bytes beyond current)
                msvcrt.locking(self.lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                # Unix file locking
                fcntl.flock(self.lock_fd, fcntl.LOCK_EX)
            return self
        except Exception as e:
            logging.warning(f"Failed to acquire lock for {self.file_path}: {e}")
            # Clean up if we failed
            if self.lock_fd:
                self.lock_fd.close()
                self.lock_fd = None
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.lock_fd:
            try:
                # Release the lock
                if self.is_windows:
                    # Windows unlock
                    msvcrt.locking(self.lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    # Unix unlock
                    fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            except Exception as e:
                logging.warning(f"Error releasing lock: {e}")
            finally:
                # Close the file
                self.lock_fd.close()
                self.lock_fd = None
                # Try to remove the lock file
                try:
                    if self.lock_file.exists():
                        self.lock_file.unlink()
                except Exception as e:
                    logging.debug(f"Could not remove lock file: {e}")

class ProgressSpinner:
    """Progress spinner for console."""
    def __init__(self):
        self.spinning = False
        self.counter = 0
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.current_message = ""
        self.term_width = shutil.get_terminal_size().columns
        self.lock = threading.Lock()

    def spin(self):
        while self.spinning:
            with self.lock:
                sys.stdout.write(f"\r{self.spinner_chars[self.counter]} {self.current_message}")
                sys.stdout.flush()
            self.counter = (self.counter + 1) % len(self.spinner_chars)
            time.sleep(0.1)

    def start(self, message=""):
        self.current_message = message
        self.spinning = True
        threading.Thread(target=self.spin, daemon=True).start()

    def update_message(self, message):
        # Get current terminal width for better clearing
        self.term_width = shutil.get_terminal_size().columns
        
        with self.lock:
            # Clear the current line completely before writing the new message
            sys.stdout.write("\r" + " " * (self.term_width - 1) + "\r")
            self.current_message = message
            sys.stdout.write(f"{self.spinner_chars[self.counter]} {self.current_message}")
            sys.stdout.flush()

    def stop(self):
        self.spinning = False
        time.sleep(0.2)  # Give the spinner thread time to notice and exit
        
        # Get current terminal width for better clearing
        self.term_width = shutil.get_terminal_size().columns
        
        # Clear the current line completely
        sys.stdout.write("\r" + " " * (self.term_width - 1) + "\r")
        sys.stdout.flush()

#######################################################################################
# ENVIRONMENT FUNCTIONS
#######################################################################################
def load_environment() -> Tuple[str, str, Path, Path]:
    """Load API key, model, and paths from environment."""
    script_dir = Path(__file__).resolve().parent
    env_path = script_dir / '.env'

    if env_path.exists():
        logging.info(f"Loading environment variables from: {env_path}")
        load_dotenv(dotenv_path=env_path)
    else:
        logging.info(f"No .env file found at: {env_path}")
        logging.info("Creating .env file with required variables...")
        with open(env_path, 'w') as f:
            f.write("CLAUDE_MODEL=claude-3-5-haiku-latest\n")
            f.write("ANTHROPIC_API_KEY=your-api-key-here\n")
        logging.info("Please update the .env file with your API key and run the script again.")
        sys.exit(1)

    model = os.getenv('CLAUDE_MODEL')
    api_key = os.getenv('ANTHROPIC_API_KEY')

    if not model:
        model = "claude-3-5-haiku-latest"
        logging.info(f"No CLAUDE_MODEL specified, defaulting to {model}")
    
    if not api_key or api_key == 'your-api-key-here':
        logging.error("Please set ANTHROPIC_API_KEY in your .env file")
        sys.exit(1)
        
    # Get PRD path from environment or use default
    prd_path_str = os.getenv('PRD_PATH')
    if prd_path_str:
        prd_path = Path(prd_path_str)
    else:
        # Fallback to user home directory if environment variable is not set
        prd_path = Path.home() / "prds"
    
    # Ensure PRD directory exists
    try:
        prd_path.mkdir(exist_ok=True, parents=True)
    except Exception as e:
        logging.error(f"Error creating PRD directory at {prd_path}: {e}")
        fallback_prd_path = Path.cwd() / "prds"
        logging.info(f"Falling back to current working directory: {fallback_prd_path}")
        fallback_prd_path.mkdir(exist_ok=True, parents=True)
        prd_path = fallback_prd_path
    
    # Get cache path from environment or use default
    cache_path_str = os.getenv('CACHE_PATH')
    if cache_path_str:
        cache_path = Path(cache_path_str)
    else:
        # Fallback to user home directory if environment variable is not set
        cache_path = Path.home() / ".prd_generator" / "cache"
    
    # Ensure cache directory exists
    try:
        cache_path.mkdir(exist_ok=True, parents=True)
    except Exception as e:
        logging.error(f"Error creating cache directory at {cache_path}: {e}")
        fallback_cache_path = Path.cwd() / ".cache"
        logging.info(f"Falling back to current working directory: {fallback_cache_path}")
        fallback_cache_path.mkdir(exist_ok=True, parents=True)
        cache_path = fallback_cache_path
    
    logging.info(f"Using PRD output path: {prd_path}")
    logging.info(f"Using cache directory: {cache_path}")

    return model, api_key, prd_path, cache_path

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the script header."""
    clear_screen()
    print("\n📄 Dynamic PRD Generator with Anthropic Claude\n")
    print("This script generates a Product Requirements Document with a dynamic number of phases.")
    print("Follow the prompts to provide project details...\n")

#######################################################################################
# PRD GENERATOR CLASS
#######################################################################################
class PRDGenerator:
    """PRD Generator Class."""
    
    #---------------------------------------------------------------------------------
    # Initialization
    #---------------------------------------------------------------------------------
    def __init__(self, model: str, api_key: str, prd_path: Path, cache_path: Path, clear_cache: bool = False):
        try:
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model
            self.prd_path = prd_path
            self.cache_path = cache_path
            self.spinner = ProgressSpinner()
            self.sections: List[str] = []
            self.project_info: Dict = {} # Initialize project_info here
            self.api_calls = 0
            self.cache_hits = 0
            self.input_tokens_saved = 0
            self.output_tokens_saved = 0
            self.total_cache_entries = 0
            self.tokens_used_input = 0  # Track input tokens used when not using cache
            self.tokens_used_output = 0  # Track output tokens used when not using cache
            
            # Clear cache if requested
            if clear_cache:
                self._clear_cache()
            
            # Load cache statistics and manage cache size
            self._init_cache()
        except Exception as e:
            logging.error(f"Error initializing Anthropic client: {e}")
            sys.exit(1)
    
    #---------------------------------------------------------------------------------
    # Cache Management Methods
    #---------------------------------------------------------------------------------
    def _clear_cache(self):
        """Clear the cache directory"""
        try:
            if self.cache_path.exists():
                logging.info(f"Clearing cache directory: {self.cache_path}")
                # Remove all files in the cache directory
                for file in self.cache_path.glob("*"):
                    file.unlink()
                logging.info("Cache cleared successfully")
        except Exception as e:
            logging.error(f"Error clearing cache: {e}")

    def _get_cache_size(self) -> int:
        """Get the total size of the cache in bytes."""
        total_size = 0
        try:
            for file in self.cache_path.glob("*.json"):
                total_size += file.stat().st_size
        except Exception as e:
            logging.warning(f"Error calculating cache size: {e}")
        return total_size
    
    def _prune_cache(self):
        """Prune the cache by removing old or excess entries."""
        try:
            # Strategy 1: Remove entries older than CACHE_RETENTION_DAYS
            now = time.time()
            files_by_age = []
            
            # Collect files with their timestamps
            for file in self.cache_path.glob("*.json"):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    timestamp = cache_data.get('timestamp', 0)
                    age_days = (now - timestamp) / (60 * 60 * 24)
                    
                    if age_days > CACHE_RETENTION_DAYS:
                        # Remove files older than retention period
                        file.unlink()
                        logging.debug(f"Removed old cache entry: {file.name}")
                    else:
                        files_by_age.append((file, timestamp))
                except Exception as e:
                    logging.warning(f"Error processing cache file {file}: {e}")
                    # If we can't read it, better to remove it
                    try:
                        file.unlink()
                    except:
                        pass
            
            # If we still need to reduce cache size
            if self._get_cache_size() > (MAX_CACHE_SIZE_MB * 1024 * 1024):
                # Sort by timestamp (oldest first)
                files_by_age.sort(key=lambda x: x[1])
                
                # Remove oldest files until we're under the limit
                for file, _ in files_by_age:
                    try:
                        file.unlink()
                        logging.debug(f"Removed cache entry due to cache size limit: {file.name}")
                        
                        # Check if we're under the limit now
                        if self._get_cache_size() <= (MAX_CACHE_SIZE_MB * 1024 * 1024):
                            break
                    except Exception as e:
                        logging.warning(f"Error removing cache file {file}: {e}")
            
            # Recalculate cache statistics
            self.total_cache_entries = len(list(self.cache_path.glob("*.json")))
            logging.info(f"Cache pruned. Remaining entries: {self.total_cache_entries}")
            
        except Exception as e:
            logging.error(f"Error pruning cache: {e}")

    def _init_cache(self):
        """Initialize the cache system"""
        try:
            # Count existing cache entries
            cache_files = list(self.cache_path.glob("*.json"))
            self.total_cache_entries = len(cache_files)
            logging.info(f"Found {self.total_cache_entries} existing cache entries")
            
            # Check cache size and prune if necessary
            cache_size = self._get_cache_size()
            cache_size_mb = cache_size / (1024 * 1024)
            logging.info(f"Current cache size: {cache_size_mb:.2f}MB")
            
            if cache_size > (MAX_CACHE_SIZE_MB * 1024 * 1024):
                logging.info(f"Cache size ({cache_size_mb:.2f}MB) exceeds limit ({MAX_CACHE_SIZE_MB}MB). Pruning old entries.")
                self._prune_cache()
        except Exception as e:
            logging.warning(f"Error initializing cache: {e}")
            self.total_cache_entries = 0

    def _get_cache_key(self, prompt: str, model: str) -> str:
        """Generate a cache key for a prompt and model."""
        hash_input = f"{prompt}:{model}".encode('utf-8')
        return hashlib.md5(hash_input).hexdigest()

    def _check_cache(self, prompt: str) -> Tuple[bool, str, int, int]:
        """Check if a response is in the cache."""
        cache_key = self._get_cache_key(prompt, self.model)
        cache_file = self.cache_path / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                # Use file locking for concurrent access
                with FileLock(cache_file):
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    input_tokens = cache_data.get('input_tokens', 0)
                    output_tokens = cache_data.get('output_tokens', 0)
                    return True, cache_data['response'], input_tokens, output_tokens
            except Exception as e:
                logging.warning(f"Error reading cache file {cache_file}: {e}")
        
        return False, "", 0, 0

    def _save_to_cache(self, prompt: str, response: str, input_tokens: int, output_tokens: int):
        """Save a response to the cache."""
        try:
            cache_key = self._get_cache_key(prompt, self.model)
            cache_file = self.cache_path / f"{cache_key}.json"
            
            cache_data = {
                'prompt': prompt,
                'response': response,
                'model': self.model,
                'timestamp': time.time(),
                'input_tokens': input_tokens,
                'output_tokens': output_tokens
            }
            
            # Use a temporary file and atomic rename for reliability
            temp_file = cache_file.with_suffix('.tmp')
            
            # Use file locking for concurrent access
            with FileLock(cache_file):
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
                # Atomic rename
                temp_file.replace(cache_file)
            
            self.total_cache_entries += 1
            
            # Check cache size periodically and prune if necessary
            if self.total_cache_entries % 10 == 0:  # Check every 10 new entries
                cache_size = self._get_cache_size()
                if cache_size > (MAX_CACHE_SIZE_MB * 1024 * 1024):
                    logging.info(f"Cache size ({cache_size/(1024*1024):.2f}MB) exceeds limit. Pruning.")
                    self._prune_cache()
                    
        except Exception as e:
            logging.warning(f"Error saving to cache: {e}")
    
    #---------------------------------------------------------------------------------
    # Token Management Methods
    #---------------------------------------------------------------------------------
    def _estimate_tokens(self, prompt: str) -> int:
        """Estimate token count for a prompt with fallback to character-based estimation."""
        try:
            # First try the official method if available
            estimated_tokens = 0
            if hasattr(self.client.messages, 'count_tokens'):
                estimated_tokens = self.client.messages.count_tokens(prompt)
            else:
                # Fallback to character-based estimation
                char_count = len(prompt)
                estimated_tokens = int(char_count * TOKENS_PER_CHAR)
                logging.info(f"Using character-based token estimation: {estimated_tokens} tokens")
            
            # Apply safety margin
            estimated_tokens = int(estimated_tokens * TOKEN_SAFETY_MARGIN)
            logging.info(f"Estimated tokens with safety margin: {estimated_tokens}")
            
            return estimated_tokens
            
        except Exception as e:
            logging.warning(f"Token estimation failed: {e}. Using default token limit.")
            # Still apply safety margin to the default
            return int(DEFAULT_MAX_TOKENS * TOKEN_SAFETY_MARGIN)
    
    #---------------------------------------------------------------------------------
    # API Interaction Methods
    #---------------------------------------------------------------------------------
    def _generate_section(self, prompt: str, identifier: str) -> str:
        """Generate PRD section with retry logic and caching."""
        retries = 0
        messages = [{"role": "user", "content": prompt}]
        
        # Check cache first
        is_cached, cached_response, input_tokens, output_tokens = self._check_cache(prompt)
        if is_cached:
            self.cache_hits += 1
            self.input_tokens_saved += input_tokens
            self.output_tokens_saved += output_tokens
            logging.info(f"Cache hit for section '{identifier}'. Tokens saved: {input_tokens} input, {output_tokens} output")
            # Clear any partial output and show a complete message for cache hits
            self.spinner.stop()
            sys.stdout.write(f"\rRetrieving '{identifier}' from cache (Tokens saved: {input_tokens + output_tokens})..." + " " * 20 + "\n")
            sys.stdout.flush()
            time.sleep(0.5)  # Small delay to ensure message is visible
            return cached_response
        
        # Estimate required tokens
        estimated_tokens = self._estimate_tokens(prompt)
        # Double the input tokens as safety margin but don't exceed model's max limit
        max_tokens = min(max(estimated_tokens * 2, DEFAULT_MAX_TOKENS), MAX_MODEL_TOKENS)
        
        logging.info(f"Section {identifier}: Estimated input tokens: {estimated_tokens}, Max output tokens: {max_tokens}")
        
        while retries < MAX_RETRIES:
            try:
                self.spinner.update_message(f"Generating section: {identifier}... (Est. tokens: {estimated_tokens})")
                response = self.client.messages.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                completion = ""
                for block in response.content:
                    if block.type == "text":
                        completion += block.text
                if completion.strip():
                    # Increment API call counter
                    self.api_calls += 1
                    
                    # Calculate token usage
                    input_tokens = estimated_tokens
                    output_tokens = self._estimate_tokens(completion)
                    
                    # Track token usage for statistics
                    self.tokens_used_input += input_tokens
                    self.tokens_used_output += output_tokens
                    
                    # Save to cache
                    self._save_to_cache(prompt, completion, input_tokens, output_tokens)
                    
                    return completion
                else:
                    raise ValueError("Empty completion received.")
            except anthropic.APIError as e:
                retries += 1
                delay = BASE_DELAY * retries
                logging.error(f"API error generating section '{identifier}': {e}. Retrying in {delay} seconds...")
                self.spinner.update_message(f"API error: {e}. Retrying in {delay}s... (Attempt {retries}/{MAX_RETRIES})")
                time.sleep(delay)
            except anthropic.APIStatusError as e:
                retries += 1
                delay = BASE_DELAY * retries
                logging.error(f"API status error generating section '{identifier}': {e}. Retrying in {delay} seconds...")
                self.spinner.update_message(f"API status error: {e}. Retrying in {delay}s... (Attempt {retries}/{MAX_RETRIES})")
                time.sleep(delay)
            except anthropic.RateLimitError as e:
                retries += 1
                delay = BASE_DELAY * retries * 2  # Longer delay for rate limits
                logging.error(f"Rate limit error generating section '{identifier}': {e}. Retrying in {delay} seconds...")
                self.spinner.update_message(f"Rate limit error. Retrying in {delay}s... (Attempt {retries}/{MAX_RETRIES})")
                time.sleep(delay)
            except Exception as e:
                retries += 1
                delay = BASE_DELAY * retries
                logging.error(f"Error generating section '{identifier}': {e}. Retrying in {delay} seconds...")
                self.spinner.update_message(f"Error: {e}. Retrying in {delay}s... (Attempt {retries}/{MAX_RETRIES})")
                time.sleep(delay)
        
        # If we get here, all retries failed
        logging.error(f"Failed to generate section '{identifier}' after {MAX_RETRIES} retries.")
        return f"<!-- Failed to generate section: {identifier} after {MAX_RETRIES} attempts -->"
    
    #---------------------------------------------------------------------------------
    # User Input Methods
    #---------------------------------------------------------------------------------
    def get_multiline_input(self, prompt: str) -> str:
        """Get multiline input from user with validation."""
        print(f"\n{prompt} (Press Enter twice to finish):")
        lines = []
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            try:
                while True:
                    line = input()
                    if line.strip() == "" and lines:
                        break
                    lines.append(line)
                
                # Validate input
                if not lines:
                    attempt += 1
                    remaining = max_attempts - attempt
                    print(f"No input received; please provide some details. {remaining} attempts remaining.")
                    continue
                
                # Input is valid, return it
                user_input = "\n".join(lines)
                
                # Additional validation for potentially harmful inputs
                if len(user_input) > 10000:
                    print("Input is too long. Please provide a shorter description.")
                    lines = []
                    attempt += 1
                    continue
                    
                return user_input
                
            except KeyboardInterrupt:
                print("\nInput interrupted. Starting over.")
                lines = []
                attempt += 1
                continue
            except Exception as e:
                print(f"\nError during input: {e}. Please try again.")
                lines = []
                attempt += 1
                continue
        
        # If we get here, all attempts failed
        print("Maximum attempts reached. Using default placeholder.")
        return "Generic project with minimal details provided."

    def get_project_info(self) -> dict:
        """Collect project information from user with validation."""
        info = {}
        
        # Get project name with validation
        max_attempts = 3
        for attempt in range(max_attempts):
            name = input("\nEnter project name: ").strip()
            if name and len(name) <= 100:
                info['name'] = name
                break
            else:
                remaining = max_attempts - attempt - 1
                if not name:
                    print(f"Project name cannot be empty. {remaining} attempts remaining.")
                else:
                    print(f"Project name is too long (max 100 characters). {remaining} attempts remaining.")
                
                if remaining == 0:
                    print("Using default project name: 'My Project'")
                    info['name'] = "My Project"

        # Removed the "Describe your project goal" heading and just kept the helpful guidance
        print("\nBefore describing your goal, please consider:\n")
        print("- Target Platform/OS: (e.g., Windows, Web, Mobile, Chrome Extension)")
        print("- Preferred Programming Language/Frameworks: (Optional)")
        print("- Are there any specific technologies to be used or avoided?")
        print("- Desired Design Style: (e.g., Sleek, Modern, Minimalistic, etc.)")
        print("If you don't have specific preferences, the PRD will attempt to fill in any gaps in the app/service design.\n")
        info['goal'] = self.get_multiline_input("Enter your project goal")

        # No longer using hard-coded tech keywords
        # Technologies will be detected by AI during recommendation phase
        info['tech_specified'] = False  # Default assumption, will be refined by AI analysis

        return info

    def _generate_tech_recommendations(self, project_info: dict) -> dict:
        """Generate technology stack recommendations based on project requirements."""
        # Validate input
        if not project_info or not isinstance(project_info, dict):
            logging.error("Invalid project info provided to _generate_tech_recommendations")
            return {"tech_stack": {}, "vision_info": {}}
            
        if 'name' not in project_info or 'goal' not in project_info:
            logging.warning("Project info missing required fields (name or goal)")
            # Set defaults for missing fields
            if 'name' not in project_info:
                project_info['name'] = "Unnamed Project"
            if 'goal' not in project_info:
                project_info['goal'] = "Create a software application with minimal details provided."
        
        self.spinner.start("Generating technology recommendations...")
        
        tech_prompt = (
            f"Based on the following project information, provide recommended technology choices:\n\n"
            f"Project Name: {project_info['name']}\n"
            f"Project Goal: {project_info['goal']}\n\n"
            f"As an expert AI assistant, analyze the project requirements and determine if any specific technologies "
            f"are mentioned or implied in the project goal. Then recommend the most appropriate modern technology stack "
            f"that would be best suited for this specific project.\n\n"
            f"Please provide concise, specific technology recommendations for this project in the following categories. "
            f"Choose modern, well-supported technologies with strong community backing. "
            f"Do not include explanations, just the specific technology names and versions where appropriate:\n\n"
            f"Programming Language:\n"
            f"Frontend Framework (if applicable):\n"
            f"Backend Framework (if applicable):\n"
            f"Database (if applicable):\n"
            f"UI Framework (if applicable):\n"
            f"State Management (if applicable):\n"
            f"Testing Framework:\n"
            f"Data Persistence Strategy:\n"
            f"Networking/API Approach:\n"
            f"Styling/UI Design System:\n\n"
            f"Note: If the project doesn't need a particular technology (e.g., no database needed for a simple tool), "
            f"respond with 'N/A' for that category. Consider modern industry best practices and performance considerations."
        )
        
        tech_response = self._generate_section(tech_prompt, "tech_recommendations")
        self.spinner.stop()
        
        # Parse the response into a structured format
        tech_stack = {}
        current_key = None
        
        # List of all expected keys for exact matching
        expected_keys = [
            "Programming Language", 
            "Frontend Framework", 
            "Backend Framework", 
            "Database", 
            "UI Framework", 
            "State Management", 
            "Testing Framework",
            "Data Persistence Strategy",
            "Networking/API Approach",
            "Styling/UI Design System"
        ]
        
        # Create a mapping of lowercase variations to the exact expected keys
        key_mapping = {k.lower(): k for k in expected_keys}
        
        for line in tech_response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a category header
            if ":" in line:
                parts = line.split(":", 1)
                key_candidate = parts[0].strip()
                
                # Try to match with expected keys (case insensitive)
                matched_key = None
                for expected_key in expected_keys:
                    if key_candidate.lower() == expected_key.lower():
                        matched_key = expected_key
                        break
                        
                # If not directly matched, try to find a partial match
                if not matched_key:
                    for lower_key, exact_key in key_mapping.items():
                        if lower_key in key_candidate.lower():
                            matched_key = exact_key
                            break
                
                if matched_key:
                    current_key = matched_key
                    value = parts[1].strip() if len(parts) > 1 else ""
                    tech_stack[current_key] = value
                else:
                    # If we don't recognize this key, but it has content, add to current key if one exists
                    if current_key and len(parts) > 1 and parts[1].strip():
                        tech_stack[current_key] += " " + line
            elif current_key and line:
                # Append to current key if this is a continuation
                tech_stack[current_key] += " " + line
        
        # Determine if specific technologies were detected
        project_info['tech_specified'] = any(
            value and value.lower() not in ["n/a", "none", "not applicable"] 
            for value in tech_stack.values()
        )
        
        # Generate project vision and benefits
        vision_prompt = (
            f"Based on the following project information, provide a concise project vision and benefits:\n\n"
            f"Project Name: {project_info['name']}\n"
            f"Project Goal: {project_info['goal']}\n\n"
            f"Please provide:\n"
            f"1. A one-sentence project vision statement that captures the essence of the project\n"
            f"2. The problem the project solves (1-2 sentences)\n"
            f"3. The intended users of the project (be specific)\n"
            f"4. Three core objectives of the project (what the project aims to achieve)\n"
            f"5. The primary target audience (1 sentence, demographic details)\n"
            f"6. Three specific success criteria for the project (measurable outcomes)\n\n"
            f"Format each response as a simple bullet point starting with a hyphen (-). Keep responses concise, specific, and direct."
        )
        
        self.spinner.start("Generating project vision and benefits...")
        vision_response = self._generate_section(vision_prompt, "project_vision")
        self.spinner.stop()
        
        # Parse vision response
        vision_info = {
            "vision_statement": "",
            "problem_solved": "",
            "intended_users": "",
            "core_objectives": [],
            "target_audience": "",
            "success_criteria": []
        }
        
        # Parse the response line by line
        lines = vision_response.strip().split('\n')
        bullet_points = [line.strip() for line in lines if line.strip().startswith('-')]
        
        # More robust parsing - assign each bullet point to the appropriate category
        if len(bullet_points) >= 1:
            vision_info["vision_statement"] = bullet_points[0][1:].strip()
        
        if len(bullet_points) >= 2:
            vision_info["problem_solved"] = bullet_points[1][1:].strip()
            
        if len(bullet_points) >= 3:
            vision_info["intended_users"] = bullet_points[2][1:].strip()
        
        # Extract core objectives (next 3 bullet points)
        for i in range(3, min(6, len(bullet_points))):
            vision_info["core_objectives"].append(bullet_points[i][1:].strip())
            
        if len(bullet_points) >= 7:
            vision_info["target_audience"] = bullet_points[6][1:].strip()
            
        # Extract success criteria (last 3 bullet points or whatever is left)
        for i in range(7, len(bullet_points)):
            vision_info["success_criteria"].append(bullet_points[i][1:].strip())
        
        return {
            "tech_stack": tech_stack,
            "vision_info": vision_info
        }

    def _ensure_complete_tech_stack(self, project_info: dict) -> dict:
        """Ensure all tech stack categories have values by asking AI for missing ones."""
        tech_stack = project_info.get("tech_recommendations", {})
        
        # Required tech stack categories
        required_categories = [
            "Programming Language", 
            "Frontend Framework",
            "Backend Framework",
            "Database",
            "UI Framework",
            "State Management",
            "Testing Framework",
            "Data Persistence Strategy",
            "Networking/API Approach",
            "Styling/UI Design System"
        ]
        
        # Check for missing categories or N/A values
        missing_categories = []
        for category in required_categories:
            if category not in tech_stack or not tech_stack[category] or tech_stack[category].lower() in ["n/a", "none"]:
                missing_categories.append(category)
        
        # If we have missing categories, ask AI to fill them in
        if missing_categories:
            self.spinner.start(f"Getting recommendations for {len(missing_categories)} missing tech categories...")
            
            # Create a prompt to get recommendations for missing categories
            missing_prompt = (
                f"Based on the following project information, provide specific technology recommendations for ONLY these missing categories:\n\n"
                f"Project Name: {project_info['name']}\n"
                f"Project Goal: {project_info['goal']}\n\n"
                f"For each of the following categories, provide a specific technology recommendation that would work well for this project. "
                f"Be specific and concise (e.g., 'React 18.2' rather than just 'a modern JavaScript framework').\n\n"
            )
            
            # List the missing categories
            for category in missing_categories:
                missing_prompt += f"{category}:\n"
                
            missing_prompt += (
                f"\nProvide only the specific technology recommendation for each category, no explanations. "
                f"If a category truly isn't applicable, respond with 'Not applicable for this project'."
            )
            
            missing_response = self._generate_section(missing_prompt, "missing_tech_recommendations")
            self.spinner.stop()
            
            # Parse the response to update the tech stack
            current_category = None
            for line in missing_response.split("\n"):
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this is a category header
                matched_category = None
                for category in missing_categories:
                    if f"{category}:" in line:
                        matched_category = category
                        current_category = category
                        value = line.split(":", 1)[1].strip() if ":" in line else ""
                        if value and value.lower() not in ["n/a", "none", "not applicable", "not applicable for this project"]:
                            tech_stack[current_category] = value
                        break
                        
                # If not a category and we have a current category, append to it
                if not matched_category and current_category and line:
                    if current_category in tech_stack:
                        tech_stack[current_category] += " " + line
                    else:
                        tech_stack[current_category] = line
            
        # Update the tech recommendations
        project_info["tech_recommendations"] = tech_stack
        return project_info
        
    def _ensure_complete_vision_info(self, project_info: dict) -> dict:
        """Ensure all vision info components are filled in by asking AI for missing ones."""
        vision_info = project_info.get("vision_info", {})
        
        # Check for missing vision components
        missing_components = []
        
        if not vision_info.get("vision_statement"):
            missing_components.append("vision_statement")
            
        if not vision_info.get("problem_solved"):
            missing_components.append("problem_solved")
            
        if not vision_info.get("intended_users"):
            missing_components.append("intended_users")
            
        if not vision_info.get("core_objectives") or len(vision_info.get("core_objectives", [])) < 3:
            missing_components.append("core_objectives")
            
        if not vision_info.get("target_audience"):
            missing_components.append("target_audience")
            
        if not vision_info.get("success_criteria") or len(vision_info.get("success_criteria", [])) < 3:
            missing_components.append("success_criteria")
            
        # If we have missing components, ask AI to fill them in
        if missing_components:
            self.spinner.start(f"Getting {len(missing_components)} missing project vision components...")
            
            # Create a prompt to get the missing components
            component_names = {
                "vision_statement": "Project Vision Statement",
                "problem_solved": "Problem Being Solved",
                "intended_users": "Intended Users",
                "core_objectives": "Core Objectives (3 specific objectives)",
                "target_audience": "Target Audience",
                "success_criteria": "Success Criteria (3 measurable outcomes)"
            }
            
            missing_prompt = (
                f"Based on the following project information, provide ONLY these missing project vision components:\n\n"
                f"Project Name: {project_info['name']}\n"
                f"Project Goal: {project_info['goal']}\n\n"
                f"For each of the following components, provide a specific, detailed response:\n\n"
            )
            
            # List the missing components
            for component in missing_components:
                missing_prompt += f"{component_names[component]}:\n"
                
            missing_prompt += (
                f"\nFormat each response clearly with the component name followed by your answer. "
                f"Be specific, clear, and focused on this particular project."
            )
            
            missing_response = self._generate_section(missing_prompt, "missing_vision_components")
            self.spinner.stop()
            
            # Parse the response to update the vision info
            current_component = None
            current_content = []
            
            # Create a mapping between prompts and our internal component names
            component_mapping = {name.lower(): key for key, name in component_names.items()}
            
            for line in missing_response.split("\n"):
                line = line.strip()
                if not line:
                    # When we hit an empty line, save current component if we have one
                    if current_component and current_content:
                        if current_component == "core_objectives" or current_component == "success_criteria":
                            # For list components, we need to parse the bullet points
                            if "core_objectives" not in vision_info:
                                vision_info["core_objectives"] = []
                            if "success_criteria" not in vision_info:
                                vision_info["success_criteria"] = []
                                
                            # Join all content lines then split by bullet points or numbers
                            content_text = " ".join(current_content)
                            bullet_items = re.findall(r'(?:^|\n)[\s-]*([^-\n][^\n]*)', content_text)
                            clean_items = [item.strip() for item in bullet_items if item.strip()]
                            
                            if current_component == "core_objectives":
                                vision_info["core_objectives"].extend(clean_items)
                            else:
                                vision_info["success_criteria"].extend(clean_items)
                        else:
                            # For single value components, join the lines
                            vision_info[current_component] = " ".join(current_content)
                            
                        current_content = []
                    continue
                
                # Check if this line starts a new component
                matched_component = None
                for prompt_text, component_key in component_mapping.items():
                    if line.lower().startswith(prompt_text.lower()):
                        # Save previous component if we were tracking one
                        if current_component and current_content:
                            if current_component == "core_objectives" or current_component == "success_criteria":
                                # For list components, we need to parse the bullet points
                                if "core_objectives" not in vision_info:
                                    vision_info["core_objectives"] = []
                                if "success_criteria" not in vision_info:
                                    vision_info["success_criteria"] = []
                                    
                                # Join all content lines then split by bullet points or numbers
                                content_text = " ".join(current_content)
                                bullet_items = re.findall(r'(?:^|\n)[\s-]*([^-\n][^\n]*)', content_text)
                                clean_items = [item.strip() for item in bullet_items if item.strip()]
                                
                                if current_component == "core_objectives":
                                    vision_info["core_objectives"].extend(clean_items)
                                else:
                                    vision_info["success_criteria"].extend(clean_items)
                            else:
                                # For single value components, join the lines
                                vision_info[current_component] = " ".join(current_content)
                                
                        # Start new component
                        matched_component = component_key
                        current_component = component_key
                        # Extract content after the colon
                        if ":" in line:
                            content = line.split(":", 1)[1].strip()
                            if content:
                                current_content = [content]
                            else:
                                current_content = []
                        else:
                            current_content = []
                        break
                
                # If not a new component and we have a current component, add this line to it
                if not matched_component and current_component:
                    current_content.append(line)
            
            # Save the last component if we were tracking one
            if current_component and current_content:
                if current_component == "core_objectives" or current_component == "success_criteria":
                    # For list components, we need to parse the bullet points
                    if "core_objectives" not in vision_info:
                        vision_info["core_objectives"] = []
                    if "success_criteria" not in vision_info:
                        vision_info["success_criteria"] = []
                        
                    # Join all content lines then split by bullet points or numbers
                    content_text = " ".join(current_content)
                    bullet_items = re.findall(r'(?:^|\n)[\s-]*([^-\n][^\n]*)', content_text)
                    clean_items = [item.strip() for item in bullet_items if item.strip()]
                    
                    if current_component == "core_objectives":
                        vision_info["core_objectives"].extend(clean_items)
                    else:
                        vision_info["success_criteria"].extend(clean_items)
                else:
                    # For single value components, join the lines
                    vision_info[current_component] = " ".join(current_content)
        
        # Update the vision info
        project_info["vision_info"] = vision_info
        return project_info

    def generate_prd_sections(self, project_info: dict) -> List[str]:
        """Generate PRD sections dynamically."""
        sections = []
        
        # First, generate tech recommendations and project vision
        recommendations = self._generate_tech_recommendations(project_info)
        project_info["tech_recommendations"] = recommendations["tech_stack"]
        project_info["vision_info"] = recommendations["vision_info"]
        
        # Ensure we have complete tech stack and vision info
        project_info = self._ensure_complete_tech_stack(project_info)
        project_info = self._ensure_complete_vision_info(project_info)
        
        base_guidelines = (
            "Follow these specific guidelines when creating the PRD:\n"
            "1. Architectural Patterns (Choose most appropriate):\n"
            "   - Clean Architecture: Separation of concerns with clear boundaries\n"
            "   - Microservices: Independent, deployable services communicating via APIs\n"
            "   - Event-Driven Architecture: Services communicating via events/messages\n"
            "   - Domain-Driven Design (DDD): Model based on business domain\n"
            "   - CQRS/Event Sourcing: Separate read and write operations\n\n"
            "2. Implementation Approach:\n"
            "   - Start with simple, working application first\n"
            "   - Verify each major step with a working build by starting the app\n" 
            "   - Add complexity incrementally, one step at a time\n"
            "   - Keep each phase testable\n"
            "   - Use feature flags for progressive delivery when appropriate\n\n"
            "3. Code Structure:\n"
            "   - Clear naming conventions\n"
            "   - Consistent patterns\n"
            "   - Proper error handling\n"
            "   - Comprehensive logging\n"
            "   - Dependency injection for loose coupling\n"
            "   - Interface-based programming\n\n"
            "4. Security Best Practices:\n"
            "   - OWASP Top 10 vulnerability protection\n"
            "   - Avoid hardcoding credentials; use environment variables or secure configuration\n"
            "   - Input validation and sanitization\n"
            "   - Secure error handling (avoid exposing sensitive info)\n"
            "   - OAuth 2.0/OIDC for modern authentication when needed\n"
            "   - Zero-trust architecture principles\n"
            "   - Supply chain security (dependency scanning)\n\n"
            "5. Testing Requirements:\n"
            "   - Unit tests for core functionality\n"
            "   - Integration tests for components\n"
            "   - End-to-end tests for features\n"
            "   - Contract testing for service boundaries\n"
            "   - Property-based testing where appropriate\n"
            "   - Visual regression testing for UI components\n"
            "   - Performance testing for critical paths\n"
            "   - Security testing (SAST/DAST)\n\n"
            "6. DevOps Integration:\n"
            "   - CI/CD pipeline configuration\n"
            "   - Infrastructure as Code (IaC)\n"
            "   - GitOps workflows when applicable\n"
            "   - Containerization strategy\n"
            "   - Observability practices (logging, metrics, tracing)\n"
            "   - Monitoring and alerting configuration\n"
        )

        ai_guidelines_ref = (
            "\n\n**Reference AI Development Assistant Guidelines:**\n"
            "The generated PRD and phase implementation plans are designed to be used with an AI development assistant. "
            "Ensure the AI assistant adheres to the following principles throughout the development process:\n\n"
            "**AI Development Assistant Guidelines Summary:**\n"
            "- **Accuracy & Honesty:** Avoid fabrication, mark opinions clearly, cite sources when available.\n"
            "- **Troubleshooting:** Provide numbered, clear steps; favor systematic debugging; start simple.\n"
            "- **Security:** No hardcoded credentials; follow security standards; input validation; OWASP compliance.\n"
            "- **Dev & Prod:** Don't disrupt working features; targeted changes; prioritize security, maintainability, performance.\n"
            "- **Performance:** Identify bottlenecks before optimizing; balance optimization with maintainability.\n"
            "- **Code & Docs:** Clear syntax highlighting; helpful comments; reference documentation.\n"
            "- **Style & Tone:** Concise, clear, straightforward communication; avoid unnecessary explanation.\n"
            "- **Design Philosophy:** Interface-based programming; dependency inversion; composition over inheritance.\n"
            "- **Cross-Platform:** Awareness of path differences and environment-specific compatibility issues.\n"
            "- **UI/UX:** Modern, accessible UI with WCAG compliance; consistent design system usage.\n"
            "- **API Design:** RESTful/GraphQL best practices; versioning; security; consistent error handling.\n"
            "- **Data Privacy:** Data minimization; privacy by design; GDPR/CCPA considerations; encryption.\n"
            "- **Clean Architecture:** Separate interfaces/implementations; factories; public APIs only.\n"
            "- **Pattern Continuity:** Maintain existing patterns and architecture.\n"
            "- **Verification:** Start app and run tests to verify each phase and step.\n"
            "- **Error Handling:** Provide clear recovery steps and explanations for common errors.\n"
            "- **Edge Cases:** Consider boundary conditions, network failures, and resource constraints.\n\n"
            "**Troubleshooting Scenarios:**\n"
            "- If application fails to build: Verify dependencies, check syntax, review compiler errors systematically.\n"
            "- If tests fail: Identify specific failing test, examine test context, verify inputs/mocks, check assertions.\n"
            "- If runtime errors occur: Check logs, validate input data, examine error messages, trace execution flow.\n"
            "- If performance issues arise: Profile application, identify bottlenecks, optimize critical paths first.\n"
            "- If security vulnerabilities are found: Address OWASP issues first, verify input validation, check auth flows.\n\n"
            "**Common Edge Cases to Handle:**\n"
            "- Network connectivity loss during operations\n"
            "- Invalid/unexpected user input formats\n"
            "- Resource exhaustion (memory, disk space, database connections)\n"
            "- Concurrent access conflicts\n"
            "- API rate limiting and throttling\n"
            "- Authentication/authorization failures\n"
            "- Third-party service unavailability\n"
        )


        # Generate overview section
        overview_prompt = (
            f"Generate a comprehensive overview for a Software Product Requirements Document (PRD) for '{project_info['name']}'.\n\n"
            f"Project Goal:\n{project_info['goal']}\n\n"
            f"Include:\n"
            f"1. Executive Summary: Brief description of the product/project and its purpose.\n"
            f"2. Vision: The problem being solved and who it's for.\n"
            f"3. Objectives: Specific business and technical goals.\n"
            f"4. Timeline: Break down the implementation into 5 distinct, logical phases of progressive complexity and feature development.\n"
            f"   - Each phase should have a clear name and short description.\n"
            f"   - Phase 1 should be a minimal working prototype or MVP.\n"
            f"   - Subsequent phases should build on previous phases, adding more features/refinements.\n"
            f"   - Later phases should include optimization, security hardening, testing, etc.\n\n"
            f"IMPORTANT FORMAT REQUIREMENT: For each phase in the timeline, use the EXACT format: '#### Phase N: Phase Name' (with 4 hashtags), followed by the description on the next lines. Example:\n\n"
            f"#### Phase 1: Initial Setup\nDescription of initial setup phase...\n\n"
            f"#### Phase 2: Core Functionality\nDescription of core functionality phase...\n\n"
            f"Format everything in Markdown. Focus on architecture and design, not implementation details."
        )
        self.spinner.start("Generating Project Overview...")
        overview_section = self._generate_section(overview_prompt, "overview")
        
        # Stop spinner completely and print a clean status message
        self.spinner.stop()
        sys.stdout.write("\rOverview generation complete." + " " * 40 + "\n")
        sys.stdout.flush()
        
        sections.append(overview_section)
        
        # Extract phase descriptions from the generated overview
        phase_descriptions = self._extract_phases_from_overview(overview_section)

        # Generate each phase section based on extracted dynamic phases
        for i, phase_info in enumerate(phase_descriptions, 1):
            phase_name = phase_info['name']
            phase_description = phase_info['description']
            
            # Start spinner directly without the redundant print message
            self.spinner.stop()
            self.spinner.start(f"Generating Phase {i}: {phase_name}...")
            
            phase_prompt = (
                f"Create Phase {i} implementation plan for {project_info['name']}.\n\n"
                f"Project Context:\n{project_info['goal']}\n"
                f"Phase: {phase_name}\n"
                f"Phase Description: {phase_description}\n\n"
                f"**Design Requirement**: Sleek, modern, minimalistic UI. Clear layout, effective typography, consistent visual style.\n"
                f"Follow Clean Architecture principles.\n"
            )

            # Add technology stack to the prompt
            if project_info.get("tech_recommendations"):
                tech_recommendations = project_info["tech_recommendations"]
                phase_prompt += f"**Technology Stack**:\n"
                
                # Add each technology if it's specified
                for tech_key, tech_value in tech_recommendations.items():
                    if tech_value and tech_value.lower() not in ["n/a", "none"]:
                        phase_prompt += f"- {tech_key}: {tech_value}\n"
                
                phase_prompt += "\n"

            phase_prompt += (
                f"{base_guidelines}\n"
                f"{ai_guidelines_ref}\n\n"
                f"PHASE {i} REQUIREMENTS:\n"
                f"- Phase {i-1} complete and verified (if not Phase 1)\n"
                f"- All previous tests passing\n"
                f"- Follow implementation steps in order, one incremental step at a time\n"
                f"- After each implementation step, start the app and verify the implementation\n"
                f"- For each step, implement comprehensive tests following these testing principles:\n"
                f"  • Unit tests for all business logic and core functionality\n"
                f"  • Integration tests for component interactions\n"
                f"  • UI/UX tests for user interfaces\n"
                f"  • Contract tests for service boundaries\n"
                f"  • Performance tests for critical paths\n"
                f"  • Security tests for sensitive operations\n"
                f"- Include CI/CD configuration for automated testing and deployment\n"
                f"- Implement observability with logging, metrics, and distributed tracing\n"
                f"- Update README.md with completed step details and verification status\n"
                f"- Confirm step completion with verification\n"
                f"- Maintain consistent architecture\n"
                f"- Get IDE AI verification before proceeding\n\n"
                "Provide:\n"
                "1. Implementation Steps:\n"
                "   - Concrete, verifiable tasks\n"
                "   - Component specifications\n"
                "   - Code structure\n"
                "   - Interface definitions\n"
                "   - Error handling approach\n"
                "   - Edge case management\n\n"
                "2. Completion Criteria:\n"
                "   - Functionality to be implemented and verified by starting the app\n"
                "   - Test requirements and coverage expectations\n"
                "   - Performance metrics (if relevant)\n"
                "   - Security compliance criteria\n"
                "   - Quality checks\n\n"
                "3. Sample Code (Illustrative):\n"
                "   - Core interfaces\n"
                "   - Key implementations\n"
                "   - Test examples (unit, integration, UI)\n"
                "   - Error handling patterns\n"
                "   - Configuration examples (avoid hardcoding, use placeholders)\n\n"
                "4. Validation Steps (Verification Checklist):\n"
                "   - Steps to start the app and verify functionality after each implementation step\n"
                "   - Test scenarios with expected outcomes\n"
                "   - Security validation procedures\n"
                "   - Performance validation methodology\n"
                "   - Success criteria for each step and phase\n"
                "   - Error case validation\n"
                "   - Integration points to verify\n\n"
                # New section for step-specific tests based on technology stack
                "5. Step-Specific Tests:\n"
                "   - For EACH implementation step, provide detailed, technology-specific test code\n"
                "   - Tests should use the correct testing framework from the technology stack\n"
                "   - Include unit tests, integration tests, and UI tests as appropriate for each step\n"
                "   - Tests should verify both successful operations and error handling\n"
                "   - Include tests for edge cases and boundary conditions\n"
                "   - Provide commands to run tests and expected outcomes\n"
                "   - Tests should follow testing best practices for the specific technology stack\n\n"
                # New section for DevOps and CI/CD
                "6. DevOps Integration:\n"
                "   - CI/CD pipeline configuration appropriate for the technology stack\n"
                "   - Automated testing setup in the pipeline\n"
                "   - Deployment strategies (blue-green, canary, etc. as appropriate)\n"
                "   - Infrastructure as Code definitions when applicable\n"
                "   - Monitoring and observability setup\n"
                "   - Alerting configuration for critical failures\n\n"
                f"For IDE AI Usage:\n"
                f"- Copy ENTIRE phase document content to IDE AI chat.\n"
                f"- Paste directly into IDE AI chat interface.\n"
                f"- For each implementation step, ask IDE AI for assistance (e.g., 'Help me implement step 1...').\n"
                f"- After implementing each step, ask IDE AI for the corresponding tests (e.g., 'Help me test step 1...').\n"
                f"- AI MUST follow architectural guidelines in the project prompt.\n"
                f"- Verify each step by starting the app AND running the tests.\n"
                f"- Confirm step completion and get AI verification before next step.\n"
                f"- **Design Focus**: Sleek, modern, minimalistic UI design.\n"
                f"- Ensure architectural integrity, encapsulation, consistent tech stack, and modern UI design.\n"
                f"- Address all edge cases and error conditions.\n"
                f"- Implement proper logging and observability.\n"
                f"Format in Markdown. Concrete, testable implementation steps. No implementation without explicit confirmation."
            )

            # After generating each phase content
            phase_content = self._generate_section(phase_prompt, f"phase_{i}")
            sections.append(phase_content)
            
            # Status after phase completion
            self.spinner.stop()
            sys.stdout.write(f"\rCompleted Phase {i}: {phase_name}" + " " * 30 + "\n")
            sys.stdout.flush()

        self.sections = sections
        return sections


    def _extract_phases_from_overview(self, overview_section: str) -> List[Dict[str, str]]:
        """Extract phase names and descriptions from the overview section."""
        if not overview_section or not isinstance(overview_section, str):
            logging.warning("Invalid overview section provided to _extract_phases_from_overview")
            # Return a default phase structure if extraction fails
            return self._get_default_phases()
            
        phase_descriptions: List[Dict[str, str]] = []
        
        # Try multiple phase header patterns
        # Pattern 1: #### Phase N: Name
        # Pattern 2: ### Phase N: Name
        # Pattern 3: ## Phase N: Name
        # Pattern 4: Phase N: Name
        # Pattern 5: **Phase N:** Name
        patterns = [
            re.compile(r"####\s+Phase\s+\d+:\s*(.*)"),
            re.compile(r"###\s+Phase\s+\d+:\s*(.*)"),
            re.compile(r"##\s+Phase\s+\d+:\s*(.*)"),
            re.compile(r"Phase\s+\d+:\s*(.*)"),
            re.compile(r"\*\*Phase\s+\d+:\*\*\s*(.*)")
        ]

        lines = overview_section.splitlines()
        i = 0
        current_phase_info = None
        
        # First try finding phases with our supported patterns
        while i < len(lines):
            line = lines[i].strip()
            matched = False
            
            for pattern in patterns:
                phase_match = pattern.match(line)
                if phase_match:
                    matched = True
                    if current_phase_info: # Save previous phase if exists
                        phase_descriptions.append(current_phase_info)

                    phase_name = phase_match.group(1).strip()
                    current_phase_info = {'name': phase_name, 'description': ""} # Start new phase info
                    i += 1
                    description_lines = []
                    
                    # Read until next phase heading or empty line followed by a heading
                    while i < len(lines):
                        next_line = lines[i].strip()
                        # Check if we've hit another phase heading
                        is_next_phase = False
                        for p in patterns:
                            if p.match(next_line):
                                is_next_phase = True
                                break
                        
                        if is_next_phase:
                            break
                        
                        description_lines.append(next_line)
                        i += 1
                    
                    current_phase_info['description'] = " ".join(description_lines).strip() # Join description lines and set
                    break  # Found a match with this pattern, break the pattern loop
            
            if not matched:
                i += 1  # Only increment if no match was found

        # Add the last phase if it was captured
        if current_phase_info:
            phase_descriptions.append(current_phase_info)

        # If we still have no phases, try to find sections that look like phases
        if not phase_descriptions:
            logging.warning("Could not find phases with standard patterns. Trying alternative extraction...")
            phase_descriptions = self._extract_phases_alternative(overview_section)

        # If still no phases, use defaults
        if not phase_descriptions:
            logging.warning("Could not automatically extract phase descriptions from overview. Using default phases.")
            return self._get_default_phases()

        return phase_descriptions
        
    def _extract_phases_alternative(self, overview_section: str) -> List[Dict[str, str]]:
        """Alternative phase extraction when standard patterns fail."""
        phases = []
        
        # Look for any heading that contains the word 'phase'
        phase_pattern = re.compile(r"#+\s+.*phase.*", re.IGNORECASE)
        # Or numbered lists that might represent phases
        numbered_list_pattern = re.compile(r"^\d+\.\s+(.*)", re.IGNORECASE)
        
        lines = overview_section.splitlines()
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for headings with "phase"
            if phase_pattern.match(line):
                phase_name = line.strip("#").strip()
                i += 1
                description = []
                while i < len(lines) and not lines[i].strip().startswith("#"):
                    description.append(lines[i].strip())
                    i += 1
                phases.append({
                    "name": phase_name,
                    "description": " ".join(description).strip()
                })
                continue
            
            # Check for numbered lists
            numbered_match = numbered_list_pattern.match(line)
            if numbered_match and i < len(lines) - 1:
                phase_name = numbered_match.group(1).strip()
                i += 1
                description = []
                # Get next paragraph as description
                while i < len(lines) and lines[i].strip() and not numbered_list_pattern.match(lines[i].strip()):
                    description.append(lines[i].strip())
                    i += 1
                phases.append({
                    "name": f"Phase {len(phases) + 1}: {phase_name}",
                    "description": " ".join(description).strip()
                })
                continue
                
            i += 1
            
        return phases
            
    def _get_default_phases(self) -> List[Dict[str, str]]:
        """Return default phases when extraction fails."""
        return [
            {'name': 'Initial Setup and Core Structure', 'description': 'Set up project files and implement basic architecture.'},
            {'name': 'Core Functionality', 'description': 'Implement the main features and functionality.'},
            {'name': 'UI Enhancement and Settings Page', 'description': 'Add a settings page and improve the user interface.'},
            {'name': 'Testing and Optimization', 'description': 'Test functionality across different scenarios and optimize performance.'},
            {'name': 'Deployment and Documentation', 'description': 'Package the extension for distribution and document the project.'}
        ]

    def save_prd(self, sections: List[str], project_name: str, project_info: dict = None):
        """Save PRD sections into Markdown files."""
        if not sections:
            logging.error("No PRD content to save.")
            return
            
        # Validate project name
        if not project_name or not isinstance(project_name, str):
            logging.warning("Invalid project name provided. Using default.")
            project_name = "unnamed_project"
            
        # Clean project name for filesystem safety
        project_name = re.sub(r'[^\w\s-]', '', project_name).strip()
        if not project_name:
            project_name = "unnamed_project"
        
        # Use the stored project_info if none is provided
        if project_info is None:
            project_info = self.project_info
            
        if not project_info or not isinstance(project_info, dict):
            logging.warning("Invalid project info. Using defaults.")
            project_info = {
                'name': project_name,
                'tech_recommendations': {},
                'vision_info': {
                    'vision_statement': '',
                    'problem_solved': '',
                    'intended_users': '',
                    'core_objectives': [],
                    'target_audience': '',
                    'success_criteria': []
                }
            }

        # Process sections to ensure proper formatting of code blocks and diagrams
        processed_sections = []
        for section in sections:
            # Fix issues with incorrectly formatted code blocks in bullet points
            # First, convert the section to lines for processing
            lines = section.split('\n')
            processed_lines = []
            in_code_block = False
            in_bullet_list = False
            bullet_indent_level = 0
            i = 0

            while i < len(lines):
                line = lines[i].rstrip()

                # Track bullet list context
                if re.match(r'^(\s*)-\s+', line):
                    bullet_match = re.match(r'^(\s*)-\s+', line)
                    current_indent = len(bullet_match.group(1))
                    in_bullet_list = True
                    bullet_indent_level = current_indent
                elif in_bullet_list and (not line.strip() or line.startswith(' ' * (bullet_indent_level + 2))):
                    # Still in the bullet list (blank line or continued indented content)
                    pass
                elif in_bullet_list:
                    # End of bullet list detected
                    in_bullet_list = False

                # Handle code block markers
                if line.strip() == '```' or line.strip().startswith('```'):
                    # Check if this is likely an incorrect code block in a bullet list
                    if in_bullet_list and i+1 < len(lines):
                        next_line = lines[i+1].rstrip()
                        if re.match(r'^(\s*)-\s+', next_line):
                            # This is likely an incorrect code block start in the middle of a bullet list
                            # Skip this code block marker
                            i += 1
                            continue

                    in_code_block = not in_code_block
                    processed_lines.append(line)
                elif line.strip() == '``' or line.strip() == '`':
                    # Incomplete code block markers - likely mistakes
                    if in_bullet_list:
                        # Skip these in bullet lists
                        i += 1
                        continue
                    processed_lines.append(line)
                else:
                    processed_lines.append(line)

                i += 1

            # Ensure all code blocks are properly closed
            if in_code_block:
                processed_lines.append('```')

            # Join the lines back together
            processed_sections.append('\n'.join(processed_lines))

        # Additional cleanup pass to fix any remaining issues
        final_sections = []
        for section in processed_sections:
            # Remove single bullet points wrapped in code blocks
            section = re.sub(r'```\s*\n\s*-\s+([^\n]+)\n\s*```', r'- \1', section)

            # Remove lone backticks in bullet lists
            section = re.sub(r'(^\s*-\s+.*)`\s*$', r'\1', section, flags=re.MULTILINE)

            # Fix any remaining misplaced code blocks
            section = re.sub(r'\n```\s*\n-\s+', r'\n\n- ', section)
            section = re.sub(r'-\s+(.*)\n```\s*\n(?!\s*```)', r'- \1\n\n', section)

            final_sections.append(section)

        sections = final_sections

        self.spinner.update_message("Creating PRD files...")
        dir_name = self.prd_path / f"{project_name.lower().replace(' ', '_')}_prd"
        
        # Backup existing directory if it exists
        if dir_name.exists():
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_dir = self.prd_path / f"{project_name.lower().replace(' ', '_')}_prd_bak_{timestamp}"
            self.spinner.update_message(f"Backing up existing directory to {backup_dir.name}...")
            shutil.copytree(dir_name, backup_dir)
            shutil.rmtree(dir_name)
            logging.info(f"Backed up existing directory to {backup_dir}")
            print(f"\rBacked up existing directory to: {backup_dir}")
        
        try:
            dir_name.mkdir(exist_ok=True, parents=True)

            # Create phase_00.md with initial project overview and basic instructions
            instructions_file = dir_name / "phase_00.md"
            with instructions_file.open('w', encoding='utf-8') as f:
                f.write(f"# {project_name} - IDE AI Instructions\n\n")
                f.write("## Project Definition\n\n")
                
                # Filter out or modify the general technology recommendations in the overview
                overview_lines = sections[0].split('\n')
                in_tech_section = False
                tech_section_start = False
                filtered_overview = []
                
                for line in overview_lines:
                    # Detect technology section
                    if "Technology Recommendations" in line or "Tech Stack" in line:
                        in_tech_section = True
                        tech_section_start = True
                        filtered_overview.append(line)  # Keep the heading
                        
                        # Add the specific tech stack instead of general categories
                        filtered_overview.append("\nUsing the following specific technology stack:\n")
                        
                        tech_recommendations = project_info["tech_recommendations"]
                        
                        # Add technology stack details
                        if "Programming Language" in tech_recommendations:
                            filtered_overview.append(f"- **Programming Language**: {tech_recommendations['Programming Language']}")
                        
                        if "Frontend Framework" in tech_recommendations:
                            filtered_overview.append(f"- **Frontend Framework**: {tech_recommendations['Frontend Framework']}")
                        
                        filtered_overview.append(f"- **Architecture Pattern**: Clean Architecture")
                        
                        if "State Management" in tech_recommendations:
                            filtered_overview.append(f"- **State Management**: {tech_recommendations['State Management']}")
                        
                        if "Styling/UI Design System" in tech_recommendations:
                            filtered_overview.append(f"- **Styling**: {tech_recommendations['Styling/UI Design System']}")
                        
                        if "Data Persistence Strategy" in tech_recommendations:
                            filtered_overview.append(f"- **Data Persistence**: {tech_recommendations['Data Persistence Strategy']}")
                        
                        if "Backend Framework" in tech_recommendations:
                            filtered_overview.append(f"- **Backend**: {tech_recommendations['Backend Framework']}")
                        
                        if "Database" in tech_recommendations:
                            filtered_overview.append(f"- **Database**: {tech_recommendations['Database']}")
                        
                        if "Networking/API Approach" in tech_recommendations:
                            filtered_overview.append(f"- **Networking**: {tech_recommendations['Networking/API Approach']}")
                        
                        if "Testing Framework" in tech_recommendations:
                            filtered_overview.append(f"- **Testing**: {tech_recommendations['Testing Framework']}")
                        
                        continue
                    
                    # Skip general tech categories but keep other parts of the tech section
                    if in_tech_section:
                        if line.startswith('###') or line.startswith('##'):
                            # Found the next section heading
                            in_tech_section = False
                            filtered_overview.append(line)
                        elif not line.strip().lower().startswith(('**backend', '**database', '**infrastructure', '**ai', '**payment', '**monitoring')):
                            # Skip general categories and options but keep other content
                            if not (tech_section_start and not line.strip()):  # Skip first empty line after heading
                                filtered_overview.append(line)
                        tech_section_start = False
                    else:
                        filtered_overview.append(line)
                
                # Write the filtered overview
                f.write('\n'.join(filtered_overview))
                f.write("\n\n## AI Development Assistant Guidelines\n\n")

                # Include the full AI Assistant Guidelines in phase_00.md
                f.write("## AI Assistant Workflow Process\n\n")
                f.write("The generated PRD is designed for a step-by-step implementation with an AI assistant:\n\n")
                f.write("1. **Project Configuration**: Use the project prompt as the project-wide AI rulebook that ensures the AI assistant follows consistent guidelines throughout all phases of development. This file defines:\n")
                f.write("   - Project architecture requirements\n")
                f.write("   - Clean Architecture principles\n")
                f.write("   - Coding standards and patterns\n")
                f.write("   - Security practices\n")
                f.write("   - Interface design principles\n")
                f.write("   - Testing requirements\n")
                f.write("   - The AI must adhere to these rules for all implementation phases\n\n")
                f.write("2. **Initial Context Setting**: Share `phase_00.md` with your IDE AI assistant first to give it an overview of the application you want to develop.\n\n")
                f.write("3. **Implementation by Phases**: For each phase:\n")
                f.write("   - Provide the content of `phase_01.md` to the AI\n")
                f.write("   - Implement one section at a time, in order\n")
                f.write("   - Implement and run the tests for each step\n")  # Add this line
                f.write("   - Verify each step works by starting the application\n")
                f.write("   - Get AI verification before proceeding to the next step or phase\n")
                f.write("   - Once phase 1 is complete and verified, move on to `phase_02.md`, and so on\n")
                f.write("   - The AI must follow the rules in the project prompt for all implementations\n\n")
                f.write("4. **Step-by-Step Approach**: Each phase document breaks implementation into small, verifiable steps. Take one step at a time and verify functionality before moving to the next.\n\n")
                
                f.write("## IDE AI Assistant Guidelines\n\n")
                f.write("The IDE AI Assistant should:\n\n")
                f.write("1. Always prioritize understanding over immediate implementation.\n")
                f.write("2. Use the information in the project prompt as guiding principles.\n")
                f.write("3. Strictly adhere to the architectural guidelines defined in the project prompt.\n")
                f.write("4. Maintain proper encapsulation between modules as defined in the architecture.\n")
                f.write("5. Follow a systematic development process: understand, plan, implement, test, verify.\n")
                f.write("6. Explain reasoning for architectural choices when implementing complex features.\n")
                f.write("7. Suggest refactoring when code deviates from the defined architecture.\n")
                f.write("8. Provide step-by-step development guidance following Step-by-Step approach.\n")
                f.write("9. Include comprehensive tests as part of the implementation process.\n\n")
                
                f.write("## IDE AI DO NOT GUIDELINES\n\n")
                f.write("The IDE AI Assistant should NOT:\n\n")
                f.write("1. Skip ahead to later phases before current phase is fully implemented and verified.\n")
                f.write("2. Implement multiple steps at once, bypassing the Step-by-Step approach.\n")
                f.write("3. Generate code that deviates from the Architecture Pattern defined in the PRD.\n")
                f.write("4. Make assumptions about feature implementations without referring to detailed requirements.\n")
                f.write("5. Leave security vulnerabilities or performance issues unaddressed.\n\n")
                f.write("Refer to the project-wide project prompt configured for the detailed technical architecture.\n\n")

            # Create project_prompt.md with focused project information
            project_prompt_file = dir_name / "project_prompt.md"
            with project_prompt_file.open('w', encoding='utf-8') as f:
                f.write(f"# {project_name} - Project Overview and Architecture\n\n") # Updated title

                # Extract project vision and core goals from the overview
                overview_lines = sections[0].split('\n')

                # Try to extract the 1-2 sentence description
                description = ""
                for line in overview_lines[:20]:
                    if line.strip() and not line.startswith('#') and not line.startswith('- ') and len(line.split()) >= 5:
                        description = line.strip()
                        break

                # Extract phase information
                phase_descriptions = self._extract_phases_from_overview(sections[0])

                # Extract tech stack info
                tech_stack = []
                in_tech_section = False
                for line in overview_lines:
                    if "Technology Recommendations" in line or "Tech Stack" in line:
                        in_tech_section = True
                        continue
                    if in_tech_section and line.startswith('##'):
                        in_tech_section = False
                        break
                    if in_tech_section and line.strip().startswith('- '):
                        tech_stack.append(line.strip())

                # Vision and Core Goals
                f.write("## Project Vision and Core Goals\n")
                
                # Add vision statement
                if description:
                    f.write(f"- {description}\n")
                else:
                    f.write(f"- {project_info['vision_info']['vision_statement']}\n")
                
                # Add problem being solved
                f.write(f"- {project_info['vision_info']['problem_solved']}\n")
                
                # Add intended users
                f.write(f"- {project_info['vision_info']['intended_users']}\n")
                
                # Add core objectives
                for objective in project_info['vision_info']['core_objectives']:
                    f.write(f"- {objective}\n")
                
                # Add target audience
                f.write(f"- {project_info['vision_info']['target_audience']}\n")
                
                # Add success criteria
                for criterion in project_info['vision_info']['success_criteria']:
                    # Only write non-empty criteria
                    if criterion and criterion.strip():
                        f.write(f"- {criterion}\n")

                f.write("\n")

                # Key Architectural Principles
                f.write("## Key Architectural Principles (Clean Architecture)\n") # Clean Architecture in heading
                f.write("- Clean separation of concerns (presentation, domain, data layers)\n")
                f.write("- Dependency injection for loose coupling\n")
                f.write("- Testable components with clear interfaces\n")
                f.write("- Single responsibility principle for all modules\n")
                f.write("- Data validation at domain layer boundary\n\n")

                # Main Component Interactions
                f.write("## Main Component Interactions\n")
                # Extract mermaid diagram
                mermaid_diagram = None
                in_mermaid = False
                mermaid_lines = []
                for line in overview_lines:
                    if line.strip() == "```mermaid":
                        in_mermaid = True
                        mermaid_lines.append(line.strip())
                        continue
                    if in_mermaid:
                        mermaid_lines.append(line)
                        if line.strip() == "```":
                            in_mermaid = False
                            break

                if mermaid_lines:
                    mermaid_diagram = "\n".join(mermaid_lines)
                    f.write(mermaid_diagram)
                else:
                    # Default diagram
                    f.write("```mermaid\n")
                    f.write("flowchart TB\n")
                    f.write("    subgraph Presentation Layer\n")
                    f.write("        UI[UI Components] --> VM[View Models]\n")
                    f.write("    end\n")
                    f.write("    subgraph Application Layer\n")
                    f.write("        VM --> UC[Use Cases]\n")
                    f.write("    end\n")
                    f.write("    subgraph Domain Layer\n")
                    f.write("        UC --> E[Entities]\n")
                    f.write("        UC --> R[Repository Interfaces]\n")
                    f.write("    end\n")
                    f.write("    subgraph Infrastructure Layer\n")
                    f.write("        RI[Repository Implementations] --> R\n")
                    f.write("        RI --> DS[Data Sources]\n")
                    f.write("    end\n")
                    f.write("    Presentation Layer --> Application Layer\n")
                    f.write("    Application Layer --> Domain Layer\n")
                    f.write("    Domain Layer --> Infrastructure Layer\n")
                    f.write("```\n")

                f.write("\n")

                # Tech Stack Summary
                f.write("## Tech Stack Summary\n")
                if tech_stack:
                    for tech in tech_stack[:6]:
                        f.write(f"{tech}\n")
                else:
                    tech_recommendations = project_info["tech_recommendations"]
                    
                    # Programming Language
                    if "Programming Language" in tech_recommendations:
                        f.write(f"- **Programming Language**: {tech_recommendations['Programming Language']}\n")
                    
                    # Frontend Framework
                    if "Frontend Framework" in tech_recommendations:
                        f.write(f"- **Frontend Framework**: {tech_recommendations['Frontend Framework']}\n")
                    
                    # Architecture Pattern is always Clean Architecture
                    f.write(f"- **Architecture Pattern**: Clean Architecture\n")
                    
                    # State Management
                    if "State Management" in tech_recommendations:
                        f.write(f"- **State Management**: {tech_recommendations['State Management']}\n")
                    
                    # Styling/UI Design
                    if "Styling/UI Design System" in tech_recommendations:
                        f.write(f"- **Styling**: {tech_recommendations['Styling/UI Design System']}\n")
                    
                    # Data Persistence
                    if "Data Persistence Strategy" in tech_recommendations:
                        f.write(f"- **Data Persistence**: {tech_recommendations['Data Persistence Strategy']}\n")
                    
                    # Backend Framework (if applicable)
                    if "Backend Framework" in tech_recommendations:
                        f.write(f"- **Backend**: {tech_recommendations['Backend Framework']}\n")
                    
                    # Database (if applicable)
                    if "Database" in tech_recommendations:
                        f.write(f"- **Database**: {tech_recommendations['Database']}\n")
                    
                    # Networking/API
                    if "Networking/API Approach" in tech_recommendations:
                        f.write(f"- **Networking**: {tech_recommendations['Networking/API Approach']}\n")
                    
                    # Testing Framework
                    if "Testing Framework" in tech_recommendations:
                        f.write(f"- **Testing**: {tech_recommendations['Testing Framework']}\n")

                f.write("\n")

                # Phase Overview
                f.write("## Phase Overview\n")
                if phase_descriptions:
                    for i, phase in enumerate(phase_descriptions, 1):
                        phase_name = phase['name']
                        f.write(f"{i}. **Phase {i}: {phase_name}**")
                        if i > 1:
                            f.write(f" - *Depends on*: Phase {i-1} completion\n")
                        else:
                            f.write("\n")
                else:
                    # Default phases
                    f.write("1. **Phase 1: Foundation/Prototype** - Project setup, architecture, basic UI\n")
                    f.write("2. **Phase 2: Core Functionality** - [Key feature set] implementation\n")
                    f.write("   - *Depends on*: Phase 1 completion\n")
                    f.write("3. **Phase 3: Enhanced Features** - [Additional features] implementation\n")
                    f.write("   - *Depends on*: Phase 2 completion\n")
                    f.write("4. **Phase 4: Polish & Optimization** - Performance, UI refinement, testing\n")
                    f.write("   - *Depends on*: Phase 3 completion\n")

                f.write("\n")

                # Implementation Guidelines
                f.write("## Implementation Guidelines\n")
                f.write("- Implement one phase at a time, or one step at a time.\n") # Step-by-Step guideline
                f.write("- Verify each phase is complete (start app and check) before next phase.\n") # Verification guideline
                f.write("- Maintain architectural integrity (Clean Architecture) throughout.\n") # Modified guideline
                f.write("- Test each component as implemented.\n")
                f.write("- Document key design decisions.\n")
                f.write("- Update README.md after each implementation step with:\n")  # New README guidelines
                f.write("  - Summary of completed functionality\n")
                f.write("  - Updated project status\n")
                f.write("  - New features or changes\n")
                f.write("  - Known issues or limitations\n")
                f.write("  - Updated setup/installation instructions if needed\n\n")

                f.write("## README.md Management\n")  # New README.md section
                f.write("The README.md file should be treated as a living document that reflects the current state of the project. After each implementation step:\n\n")
                f.write("1. **Update Project Status**\n")
                f.write("   - Current phase and step completion status\n")
                f.write("   - Recently implemented features\n")
                f.write("   - Verification status of new features\n")
                f.write("   - Test coverage and results summary\n\n")
                f.write("2. **Document Changes**\n")
                f.write("   - New dependencies or requirements\n")
                f.write("   - Configuration changes\n")
                f.write("   - API updates or changes\n")
                f.write("   - New environment variables\n")
                f.write("   - Architecture decisions and rationale\n\n")
                f.write("3. **Update Instructions**\n")
                f.write("   - Setup and installation steps\n")
                f.write("   - Build and run commands\n")
                f.write("   - Testing procedures\n")
                f.write("   - Deployment procedures\n")
                f.write("   - Troubleshooting guides\n")
                f.write("   - Environment configuration\n\n")
                f.write("4. **Track Progress**\n")
                f.write("   - Completed features checklist\n")
                f.write("   - Known issues and workarounds\n")
                f.write("   - Upcoming features or tasks\n")
                f.write("   - Dependencies and requirements\n")
                f.write("   - Performance metrics and benchmarks\n\n")
                f.write("5. **Security and Compliance**\n")
                f.write("   - Security features implemented\n")
                f.write("   - Authentication/authorization model\n")
                f.write("   - Data privacy considerations\n")
                f.write("   - Regulatory compliance status\n")
                f.write("   - Vulnerability assessment results\n\n")
                f.write("6. **API Documentation**\n")
                f.write("   - Endpoint specifications\n")
                f.write("   - Request/response formats\n")
                f.write("   - Authentication requirements\n")
                f.write("   - Rate limiting information\n")
                f.write("   - Example requests/responses\n\n")

            # Create phase files
            for i, phase in enumerate(sections[1:], 1):
                phase_file = dir_name / f"phase_{i:02d}.md"

                # Modify Phase 1 - generic initialization instructions
                if i == 1:
                    project_init_note = (
                        "### ⚠️ IMPORTANT: Project Initialization\n"
                        "Start by setting up the basic project structure and tooling.\n"
                        "This includes:\n"
                        "- Create project directory (if needed).\n"
                        "- Initialize repository (e.g., `git init`).\n"
                        "- Set up dev environment with SDKs/runtimes.\n"
                        "- Install dependencies/packages.\n"
                        "- Create initial config files.\n"
                        "- Set up build process to compile and run project.\n"
                        "- Configure linting and code formatting tools.\n"
                        "- Set up testing framework and write initial tests.\n"
                        "- Create CI/CD pipeline configuration.\n"
                        "- Initialize observability setup (logging, metrics).\n"
                        "- Configure development environment variables.\n"
                        "- Initialize README.md with comprehensive project information.\n"
                        "Refer to 'Tech Stack Summary' in the project prompt for the technology-specific setup.\n\n"
                        "**README.md Initial Setup**:\n"
                        "Create a comprehensive README.md that includes:\n"
                        "- Project name and description\n"
                        "- Technology stack and requirements\n"
                        "- Architecture overview with diagram\n"
                        "- Setup and installation instructions\n"
                        "- Environment configuration details\n"
                        "- Build and run commands\n"
                        "- Testing instructions and commands\n"
                        "- API documentation (if applicable)\n"
                        "- Development guidelines and coding standards\n"
                        "- Project structure overview with explanations\n"
                        "- Contributing guidelines\n"
                        "- License information\n"
                        "- Troubleshooting common issues\n"
                        "- Performance considerations\n"
                        "- Security guidelines\n\n"
                        "**CI/CD Setup**:\n"
                        "Configure a CI/CD pipeline that includes:\n"
                        "- Automated builds on code changes\n"
                        "- Static code analysis\n"
                        "- Automated testing (unit, integration, etc.)\n"
                        "- Security scanning\n"
                        "- Deployment automation\n"
                        "- Environment-specific configurations\n"
                        "- Notification system for build/test failures\n\n"
                        "**Verification**: After setup, start the app to ensure basic project setup is correct and the app runs without errors. Run the initial test suite to verify testing framework is properly configured.\n\n"
                    )

                    if "## 1. Implementation Steps" in phase:
                        parts = phase.split("## 1. Implementation Steps")
                        if len(parts) > 1:
                            phase = parts[0] + "## 1. Implementation Steps\n\n" + project_init_note + parts[1]

                # Replace software_architecture.md with project_prompt.md
                phase = phase.replace("software_architecture.md", "project_prompt.md")

                with phase_file.open('w', encoding='utf-8') as f:
                    try:
                        f.write(f"# Phase {i} Implementation Plan\n\n")

                        # Architecture reminder - removed Cosmos Pattern reference
                        f.write("## 🔧 Technical Architecture\n\n")
                        f.write("### Technology Stack Overview\n\n")
                        
                        tech_recommendations = project_info.get("tech_recommendations", {})
                        
                        # Programming Language
                        if "Programming Language" in tech_recommendations:
                            f.write(f"- **Programming Language**: {tech_recommendations['Programming Language']}\n")
                        
                        # Frontend Framework
                        if "Frontend Framework" in tech_recommendations:
                            f.write(f"- **Frontend Framework**: {tech_recommendations['Frontend Framework']}\n")
                        
                        # Architecture Pattern is always Clean Architecture
                        f.write(f"- **Architecture Pattern**: Clean Architecture\n")
                        
                        # State Management
                        if "State Management" in tech_recommendations:
                            f.write(f"- **State Management**: {tech_recommendations['State Management']}\n")
                        
                        # Styling/UI Design
                        if "Styling/UI Design System" in tech_recommendations:
                            f.write(f"- **Styling**: {tech_recommendations['Styling/UI Design System']}\n")
                        
                        # Data Persistence
                        if "Data Persistence Strategy" in tech_recommendations:
                            f.write(f"- **Data Persistence**: {tech_recommendations['Data Persistence Strategy']}\n")
                        
                        # Backend Framework (if applicable)
                        if "Backend Framework" in tech_recommendations:
                            f.write(f"- **Backend**: {tech_recommendations['Backend Framework']}\n")
                        
                        # Database
                        if "Database" in tech_recommendations:
                            f.write(f"- **Database**: {tech_recommendations['Database']}\n")
                        
                        # Networking/API
                        if "Networking/API Approach" in tech_recommendations:
                            f.write(f"- **Networking**: {tech_recommendations['Networking/API Approach']}\n")
                        
                        # Testing Framework
                        if "Testing Framework" in tech_recommendations:
                            f.write(f"- **Testing**: {tech_recommendations['Testing Framework']}\n")

                        f.write("\n## 📚 Technical Guidelines\n\n")
                        f.write("- Follow clean architecture principles with clear separation of concerns.\n") # Clean Architecture
                        f.write("- Ensure proper dependency injection and interface-based design.\n") # DI
                        f.write("- Use appropriate design patterns to solve common problems.\n") # Design Patterns
                        f.write("- Prioritize testability in all components.\n") # Testing
                        f.write("- Ensure proper error handling and user feedback.\n") # Error handling
                        f.write("- Maintain architectural integrity throughout.\n") # Architecture

                        f.write("\n## Implementation Requirements\n\n")
                        f.write("- Review README.md first to understand current project status and development progress\n")  # New instruction
                        if i > 1:
                            f.write(f"- Phase {i-1} complete and verified (by starting app)\n")
                            f.write("- All previous tests passing\n")
                        f.write("- Follow implementation steps in order, one incremental step at a time.\n")
                        f.write("- After each step, start the app and verify implementation.\n")
                        f.write("- For each step, implement the specified tests and ensure they pass.\n")  # Add this line
                        f.write("- Update README.md with completed step details and verification status.\n")
                        f.write("- Confirm step completion.\n")
                        f.write("- Maintain consistent architecture.\n")
                        f.write("- Get IDE AI verification before proceeding.\n\n")
                        f.write(phase)
                        f.write("\n\n## Completion Checklist\n\n")
                        f.write("- [ ] All implementation steps completed (Step-by-Step)\n")
                        f.write("- [ ] After each step, app started and functionality verified\n")
                        f.write("- [ ] For each step, all tests implemented and passing\n")  # Add this new checklist item
                        f.write("- [ ] README.md updated with completed step details\n")
                        f.write("- [ ] All tests passing\n")
                        f.write("- [ ] Code reviewed and documented\n")
                        f.write("- [ ] Architectural compliance verified\n")
                        f.write("- [ ] IDE AI verification received\n")
                    except Exception as e:
                        logging.error(f"Error writing to phase file {phase_file}: {e}")
                        raise

            self.spinner.stop()
            print(f"\nPRD files created in '{dir_name}' directory.")
            
        except Exception as e:
            self.spinner.stop()
            logging.error(f"Error saving PRD files: {e}")

def main():
    """Main function to run the PRD generator."""
    try:
        # Set up command-line argument parsing
        parser = argparse.ArgumentParser(description='Generate a Product Requirements Document (PRD) with dynamic phases')
        parser.add_argument('--clear-cache', action='store_true', help='Clear the cache before running')
        parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
        args = parser.parse_args()
        
        # Configure logging level based on verbose flag
        if args.verbose:
            logger.setLevel(logging.INFO)
            logging.info("Verbose logging enabled")
        
        # Print header only once at the start
        print_header()
        
        try:
            model, api_key, prd_path, cache_path = load_environment()
        except Exception as e:
            logging.critical(f"Failed to load environment: {e}")
            print("\nError: Failed to load environment settings. Please check your .env file.")
            sys.exit(1)
            
        try:
            generator = PRDGenerator(model, api_key, prd_path, cache_path, args.clear_cache)
        except Exception as e:
            logging.critical(f"Failed to initialize generator: {e}")
            print("\nError: Failed to initialize the PRD generator. Check your API credentials.")
            sys.exit(1)
            
        try:
            project_info = generator.get_project_info()
            # Store project_info in the generator for later use
            generator.project_info = project_info
        except KeyboardInterrupt:
            print("\n\nPRD generation cancelled during project info collection.")
            sys.exit(0)
        except Exception as e:
            logging.error(f"Error collecting project information: {e}")
            print("\nError: Failed to collect project information. Using minimal defaults.")
            project_info = {'name': 'Unnamed Project', 'goal': 'Create a software application.', 'tech_specified': False}
            generator.project_info = project_info
            
        try:
            prd_sections = generator.generate_prd_sections(project_info)
        except KeyboardInterrupt:
            print("\n\nPRD generation cancelled during section generation.")
            sys.exit(0)
        except Exception as e:
            logging.error(f"Error generating PRD sections: {e}")
            print("\nError: Failed to generate PRD sections.")
            sys.exit(1)

        if prd_sections:
            try:
                generator.save_prd(prd_sections, project_info['name'], project_info)
            except Exception as e:
                logging.error(f"Error saving PRD files: {e}")
                print(f"\nError: Failed to save PRD files: {e}")
                sys.exit(1)
                
            print("\nPRD generation complete! Implementation guide in generated directory.")
            
            # Display cache statistics
            print("\n📊 Generation Statistics")
            print(f"• Output directory: {generator.prd_path / project_info['name'].lower().replace(' ', '_')}_prd")
            print(f"• Cache directory: {generator.cache_path}")
            print(f"• Model: {generator.model}")
            print(f"• API calls: {generator.api_calls}")
            print(f"• Cache hits: {generator.cache_hits}")
            print(f"• Total cache entries: {generator.total_cache_entries}")
            
            # Show tokens used for API calls
            if generator.api_calls > 0:
                input_tokens_formatted = f"{generator.tokens_used_input:,}"
                output_tokens_formatted = f"{generator.tokens_used_output:,}"
                total_tokens_formatted = f"{generator.tokens_used_input + generator.tokens_used_output:,}"
                print(f"• Tokens used: {input_tokens_formatted} input, {output_tokens_formatted} output")
                print(f"• Total tokens used: {total_tokens_formatted}")
            
            # Show tokens saved from cache hits
            if generator.cache_hits > 0:
                input_tokens_formatted = f"{generator.input_tokens_saved:,}"
                output_tokens_formatted = f"{generator.output_tokens_saved:,}"
                total_tokens_formatted = f"{generator.input_tokens_saved + generator.output_tokens_saved:,}"
                print(f"• Tokens saved: {input_tokens_formatted} input, {output_tokens_formatted} output")
                print(f"• Total tokens saved: {total_tokens_formatted}")
                
            print("\nIMPORTANT:")
            print("1. Configure project-wide AI rules with project_prompt.md")
            print("2. Share phase_00.md with IDE AI assistant for project overview")
            print("3. Start implementation: copy phase_01.md content to IDE AI chat")
            print("4. Follow phase documents in order for implementation, and take one incremental step at a time and verify after each step.")
            print("5. Get IDE AI verification before moving to next step or phase.")

    except KeyboardInterrupt:
        print("\n\nPRD generation cancelled.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()