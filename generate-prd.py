#!/usr/bin/env python3
"""
Enhanced Dynamic Phased PRD Generator Script using Anthropic Messages API

This script generates a PRD (Product Requirements Document) with a dynamic number of phases
based on user input, using the Anthropic API.

The PRD is designed to be technology-agnostic by default, focusing on clean architecture,
design principles, and phased implementation. Technology-specific suggestions will be made
only if the user explicitly specifies a technology stack in their project goal.

Workflow for IDE AI Integration:

1. Generate PRD using this script.
2. Locate the generated PRD directory (e.g., <project_name>_prd).
3. Open 'instructions.md' and provide its content as the initial prompt to your IDE AI chat.
   - Wait for the IDE AI to acknowledge and understand the project instructions.
4. For each phase of implementation:
   - Open the corresponding 'phase_XX.md' file (e.g., 'phase_01.md').
   - Copy the ENTIRE content of the phase document.
   - Paste the content into the IDE AI chat interface.
   - For each implementation step in the phase document, ask the IDE AI for assistance,
     e.g., 'Help me implement step 1: ...'
   - Ensure that the IDE AI strictly adheres to the architectural guidelines defined in 'project_prompt.md'
   - Prioritize architectural compliance and code quality throughout the implementation.
   - Always verify the IDE AI's suggestions against the project architecture.
   - After each implementation step and at the end of each phase, run the application to verify correct implementation.
   - Before proceeding to the next step, confirm the completion of the current step and get IDE AI verification.
5. 'project_prompt.md' serves as a project-level configuration and architectural rulebook for the IDE AI.
   - It should be kept in the project directory for the IDE AI to reference throughout the development process.
"""

import os
import sys
import time
import threading
import logging
import hashlib
import json
import datetime
import signal
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import re

try:
    import requests
except ImportError:
    print("Error: The 'requests' library is required but not installed.")
    print("Please install it using: pip install requests")
    sys.exit(1)

try:
    import anthropic
except ImportError:
    print("Error: The 'anthropic' library is required but not installed.")
    print("Please install it using: pip install anthropic")
    sys.exit(1)

from dotenv import load_dotenv

# Default output path for PRD documents
PRD_OUTPUT_PATH = Path(os.getenv("PRD_OUTPUT_PATH", "D:/Workspaces/prds"))

# Cache directory for API responses
CACHE_DIR = Path(os.getenv("PRD_CACHE_DIR", "D:/Workspaces/prds/cache"))

# Load and validate the output paths
def validate_paths():
    """Validate the output paths and create them if possible."""
    paths = {
        "PRD_OUTPUT_PATH": PRD_OUTPUT_PATH,
        "CACHE_DIR": CACHE_DIR
    }
    
    for name, path in paths.items():
        try:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                logging.info(f"Created {name} directory at {path}")
            if not os.access(path, os.W_OK):
                logging.warning(f"{name} path {path} is not writable. May cause permissions issues.")
        except Exception as e:
            logging.error(f"Error validating {name} path ({path}): {e}")
            if name == "PRD_OUTPUT_PATH":
                alternative = Path.home() / "prd_documents"
                logging.warning(f"Will attempt to use alternative path: {alternative}")
                # Update the global variable
                globals()[name] = alternative
                if not alternative.exists():
                    try:
                        alternative.mkdir(parents=True, exist_ok=True)
                    except Exception as alt_err:
                        logging.error(f"Cannot create alternative path: {alt_err}")
    
    return paths

# Global variable to store the generator for signal handlers
active_generator = None

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    """Handle interrupt signals gracefully."""
    print("\nReceived interrupt signal. Cleaning up...")
    print("Saving cache before exit...")
    # We'll rely on the general exception handling in main() instead
    # of trying to access potentially non-existent objects here
    sys.exit(0)

# Register signal handlers for multiple signal types
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination request

# Handle SIGABRT if available on this platform
try:
    signal.signal(signal.SIGABRT, signal_handler)  # Abnormal termination
except AttributeError:
    pass  # SIGABRT not available on this platform

# Configure logging for better error tracking with colors
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

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger()
handler = logger.handlers[0]
handler.setFormatter(ColoredFormatter("%(levelname)s: %(message)s"))
handler.addFilter(CustomFilter())

# Constants for retry logic in API calls
MAX_RETRIES = 3
BASE_DELAY = 5  # seconds
DEFAULT_MAX_TOKENS = 4096  # Updated default for Claude-3
TOKENS_PER_CHAR = 0.3  # Rough estimation ratio for fallback calculation

def load_environment() -> Tuple[str, str]:
    """Load API key and model from environment."""
    script_dir = Path(__file__).resolve().parent
    env_path = script_dir / '.env'

    if env_path.exists():
        logging.info(f"Loading environment variables from: {env_path}")
        load_dotenv(dotenv_path=env_path)
    else:
        logging.info(f"No .env file found at: {env_path}")
        logging.info("Creating .env file with required variables...")
        with open(env_path, 'w') as f:
            f.write("CLAUDE_MODEL=claude-3-opus-20240229\n")
            f.write("ANTHROPIC_API_KEY=your-api-key-here\n")
        logging.info("Please update the .env file with your API key and run the script again.")
        sys.exit(1)

    model = os.getenv('CLAUDE_MODEL')
    api_key = os.getenv('ANTHROPIC_API_KEY')

    if not model or not api_key or api_key == 'your-api-key-here':
        logging.error("Please set CLAUDE_MODEL and ANTHROPIC_API_KEY in your .env file")
        sys.exit(1)
        
    # Security warning for sensitive credentials
    file_permissions = os.stat(env_path).st_mode & 0o777
    if file_permissions != 0o600 and os.name != 'nt':  # Not applicable on Windows
        logging.warning(f"Security risk: .env file permissions ({oct(file_permissions)}) are too permissive")
        logging.warning("Consider restricting access with: chmod 600 .env")
        
        # Prompt user to fix permissions
        user_response = input("Restrict .env file permissions now? (y/N): ").strip().lower()
        if user_response == 'y':
            try:
                os.chmod(env_path, 0o600)
                logging.info("File permissions updated to 600 (read/write for owner only)")
            except Exception as e:
                logging.error(f"Failed to update file permissions: {e}")

    return model, api_key

class ProgressSpinner:
    """Progress spinner for console."""
    def __init__(self):
        self.spinning = False
        self.counter = 0
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.current_message = ""
        self.thread = None

    def spin(self):
        while self.spinning:
            sys.stdout.write(f"\r{self.spinner_chars[self.counter]} {self.current_message}")
            sys.stdout.flush()
            self.counter = (self.counter + 1) % len(self.spinner_chars)
            time.sleep(0.1)

    def start(self, message=""):
        if self.spinning:
            self.update_message(message)
            return
            
        self.current_message = message
        self.spinning = True
        self.thread = threading.Thread(target=self.spin, daemon=True)
        self.thread.start()

    def update_message(self, message):
        self.current_message = message
        sys.stdout.write("\r" + " " * (len(self.spinner_chars) + len(self.current_message) + 2) + "\r")
        sys.stdout.write(f"{self.spinner_chars[self.counter]} {self.current_message}")
        sys.stdout.flush()

    def stop(self):
        """Safely stop the spinner thread."""
        if not self.spinning:
            return
            
        self.spinning = False
        
        # Wait for thread to complete (max 1 second)
        if self.thread and self.thread.is_alive():
            try:
                self.thread.join(timeout=1.0)
            except Exception:
                pass  # Ignore join errors
                
        # Clear the spinner line
        sys.stdout.write("\r" + " " * (len(self.spinner_chars) + len(self.current_message) + 2) + "\r")
        sys.stdout.flush()
        
        # Reset thread reference
        self.thread = None


class PRDGenerator:
    """PRD Generator Class."""
    def __init__(self, model: str, api_key: str):
        try:
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model
            self.spinner = ProgressSpinner()
            self.sections: List[str] = []
            self.project_info: Dict = {} # Initialize project_info here
            
            # Ensure cache directory exists
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            logging.info(f"Cache directory: {CACHE_DIR}")
            
            # Initialize cache-related attributes
            self.response_cache = {}
            self.session_cache = {}
            self.api_calls_made = 0
            self.cache_hits = []
            self.tokens_saved = {
                'input': 0,
                'output': 0
            }
            
            # Load existing response cache if available
            try:
                self.response_cache = self._load_cache()
            except Exception as cache_err:
                logging.warning(f"Error loading cache: {cache_err}. Starting with empty cache.")
                self.response_cache = {}
                
        except Exception as e:
            logging.error(f"Error initializing Anthropic client: {e}")
            sys.exit(1)

    def _cache_key(self, prompt: str, model: str) -> str:
        """Generate a deterministic cache key based on prompt and model."""
        # Use a hash of the prompt and model as the cache key
        content = f"{prompt}_{model}".encode('utf-8')
        return hashlib.md5(content).hexdigest()
        
    def _load_cache(self) -> Dict:
        """Load the response cache from disk."""
        cache_file = CACHE_DIR / "response_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                    logging.info(f"Loaded cache with {len(cache)} entries")
                    return cache
            except Exception as e:
                logging.warning(f"Failed to load cache: {e}. Starting with empty cache.")
        return {}
        
    def _save_cache(self) -> None:
        """Save the response cache to disk with retry mechanism."""
        cache_file = CACHE_DIR / "response_cache.json"
        lock_file = CACHE_DIR / "response_cache.lock"
        max_retries = 3
        retry_count = 0
        
        # Import here to avoid potential platform-specific import errors
        import os
        try:
            import fcntl
            has_fcntl = True
        except ImportError:
            # Windows doesn't have fcntl
            has_fcntl = False
            import msvcrt
            
        while retry_count < max_retries:
            lock_fd = None
            try:
                # Acquire a lock on the lock file
                if has_fcntl:  # Unix-based systems
                    lock_fd = open(lock_file, 'w')
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:  # Windows
                    # Try to create/open the lock file in exclusive mode
                    try:
                        lock_fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    except FileExistsError:
                        # If file exists, another process has the lock
                        logging.warning("Cache is locked by another process. Retrying...")
                        retry_count += 1
                        time.sleep(0.5 * retry_count)
                        continue
                
                # First save to a temporary file
                temp_file = cache_file.with_suffix('.tmp')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(self.response_cache, f, ensure_ascii=False, indent=2)
                
                # Use os.replace for atomic operation across platforms
                os.replace(temp_file, cache_file)
                
                logging.info(f"Saved cache with {len(self.response_cache)} entries")
                return
                
            except Exception as e:
                retry_count += 1
                wait_time = 0.5 * retry_count  # Exponential backoff
                logging.warning(f"Failed to save cache (attempt {retry_count}/{max_retries}): {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            finally:
                # Release the lock
                if lock_fd is not None:
                    if has_fcntl:  # Unix
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                        lock_fd.close()
                    else:  # Windows
                        os.close(lock_fd)
                        try:
                            os.remove(lock_file)
                        except Exception as e:
                            logging.warning(f"Failed to remove lock file: {e}")
        
        logging.error(f"Failed to save cache after {max_retries} attempts")
            
    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get a cached response if available."""
        if cache_key in self.response_cache:
            cache_entry = self.response_cache[cache_key]
            logging.info(f"Cache hit: {cache_key[:8]}... (from {cache_entry['timestamp']})")
            return cache_entry['response']
        return None
        
    def _cache_response(self, cache_key: str, response: str) -> None:
        """Cache a response."""
        self.response_cache[cache_key] = {
            'response': response,
            'timestamp': datetime.datetime.now().isoformat(),
        }
        # Save cache after each new entry to prevent data loss
        self._save_cache()

    def _estimate_tokens(self, prompt: str) -> int:
        """Estimate token count for a prompt with fallback to character-based estimation."""
        try:
            # First try the official method if available
            if hasattr(self.client.messages, 'count_tokens'):
                return self.client.messages.count_tokens(prompt)
            
            # Fallback to character-based estimation
            char_count = len(prompt)
            # Use a more conservative estimation with a safety margin (1.2x)
            estimated_tokens = int(char_count * TOKENS_PER_CHAR * 1.2)
            logging.info(f"Using character-based token estimation: {estimated_tokens} tokens")
            return estimated_tokens
        except Exception as e:
            logging.warning(f"Token estimation failed: {e}. Using conservative character-based estimation.")
            # Conservative fallback - character count with safety margin
            char_count = len(prompt)
            conservative_estimate = int(char_count * TOKENS_PER_CHAR * 1.5)  # 1.5x safety margin
            logging.info(f"Conservative token estimate: {conservative_estimate}")
            return min(conservative_estimate, DEFAULT_MAX_TOKENS)  # Cap at default max

    def _generate_section(self, prompt: str, identifier: str) -> str:
        """Generate a section of the PRD using Claude API."""
        if not prompt:
            logging.error(f"Empty prompt provided for section: {identifier}")
            return ""
            
        try:
            # Check cache first
            cache_key = self._cache_key(prompt, self.model)
            cached_response = self._get_cached_response(cache_key)
            
            if cached_response:
                if not hasattr(self, 'cache_hits'):
                    self.cache_hits = []
                self.cache_hits.append(cache_key)
                
                # Extract timestamp for display if available
                timestamp = self.response_cache.get(cache_key, {}).get('timestamp', 'Unknown')
                logging.debug(f"Cache hit: {cache_key[:8]}... (from {timestamp})")
                
                # Estimate token savings
                estimated_input_tokens = self._estimate_tokens(prompt)
                # Output tokens are typically ~1.5x input tokens
                estimated_output_tokens = int(estimated_input_tokens * 1.5)
                
                if not hasattr(self, 'tokens_saved'):
                    self.tokens_saved = {'input': 0, 'output': 0}
                self.tokens_saved['input'] += estimated_input_tokens  
                self.tokens_saved['output'] += estimated_output_tokens
                
                logging.debug(f"Cache hit saved approx. {estimated_input_tokens + estimated_output_tokens} tokens")
                
                self.spinner.update_message(f"Using cached response for section: {identifier}")
                time.sleep(0.5)  # Brief pause for UI feedback
                return cached_response
            
            # Estimate token count for logging
            estimated_tokens = self._estimate_tokens(prompt)
            logging.debug(f"Conservative token estimate: {estimated_tokens}")
            
            if identifier.startswith("phase_"):
                # For phase sections, estimate needed tokens
                logging.debug(f"Section {identifier}: Estimated input tokens: {estimated_tokens}, Max output tokens: {estimated_tokens * 1.5}")
            
            # No cache hit, make API call
            self.spinner.update_message(f"Generating section: {identifier}... (Est. tokens: {estimated_tokens})")
            
            # Use API to generate response
            prompt = self._optimize_prompt(prompt)
            
            messages = [{"role": "user", "content": prompt}]
            
            # Get API key from the client object or directly from the instance
            api_key = self.api_key if hasattr(self, 'api_key') else self.client.api_key
            
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "max_tokens": 4000,
                "temperature": 0.3,
                "messages": messages
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload
            )
            
            # Increment API call counter
            if not hasattr(self, 'api_calls_made'):
                self.api_calls_made = 0
            self.api_calls_made += 1
            
            logging.debug(f"HTTP Request: POST https://api.anthropic.com/v1/messages \"{response.status_code} {response.reason}\"")
            
            if response.status_code != 200:
                logging.error(f"API Error: {response.status_code} - {response.text}")
                return ""
                
            result = response.json()
            
            # Log token usage
            usage = {
                'input_tokens': result.get('usage', {}).get('input_tokens', 0),
                'output_tokens': result.get('output_tokens', 0)
            }
            logging.debug(f"Actual tokens used - Input: {usage['input_tokens']}, Output: {usage['output_tokens']}")
            
            content = result['content'][0]['text']
            
            # Refresh API key if needed
            if 'api key exceeds' in content.lower() or 'api key has been revoked' in content.lower():
                raise Exception("API key issue detected in response. Please check your API key.")
            
            # Cache the response
            self._cache_response(cache_key, content)
            
            return content
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}")
            return ""
        except json.JSONDecodeError:
            logging.error("Failed to parse API response")
            return ""
        except Exception as e:
            logging.error(f"Error generating section: {e}")
            return ""

    def _optimize_prompt(self, prompt: str) -> str:
        """Optimize the prompt to reduce token usage."""
        # Remove excessive newlines (more than 2 in a row)
        prompt = re.sub(r'\n{3,}', '\n\n', prompt)
        
        # Remove trailing/leading whitespace on each line
        prompt = '\n'.join(line.strip() for line in prompt.split('\n'))
        
        # Remove redundant spaces
        prompt = re.sub(r' {2,}', ' ', prompt)
        
        return prompt

    def get_multiline_input(self, prompt: str) -> str:
        """Get multi-line input from user with input validation."""
        print(f"{prompt} (Press Enter twice to finish):")
        
        # Read all input at once and split on double newlines
        all_lines = []
        
        # Maximum allowed input size
        MAX_INPUT_SIZE = 100 * 1024
        total_size = 0
        
        # Read the first line outside the loop
        first_line = input()
        if first_line:  # Only add non-empty first line
            all_lines.append(first_line)
            total_size += len(first_line.encode('utf-8'))
        
        # Now track explicitly if we've seen an empty line
        previous_empty = first_line == ""
        
        try:
            while True:
                line = input()
                
                if line == "":
                    break
                
                # Check size limit
                line_size = len(line.encode('utf-8'))
                if total_size + line_size > MAX_INPUT_SIZE:
                    print(f"\nWarning: Input exceeds size limit ({MAX_INPUT_SIZE // 1024}KB).")
                    break
                    
                # Add to collection (we'll include the first empty line)
                all_lines.append(line)
                total_size += line_size
                
        except KeyboardInterrupt:
            print("\nInput interrupted.")
        except Exception as e:
            logging.error(f"Error reading input: {e}")
            
        # Join the lines with newlines
        return "\n".join(all_lines)

    def get_project_info(self) -> dict:
        """Get project information from the user."""
        # Show the header
        print_header()
        
        info = {
            "version": 1.0,
            "creation_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "use_cache": True  # Default to using cache
        }

        # Get project name with validation for special characters
        while True:
            name = input("\nEnter project name: ").strip()
            # Check for empty name
            if not name:
                print("Project name cannot be empty. Please try again.")
                continue
                
            # Check for special characters that might cause issues in file paths
            invalid_chars = '<>:"/\\|?*'
            if any(char in name for char in invalid_chars):
                print(f"Project name contains invalid characters. Please avoid: {invalid_chars}")
                continue
                
            info['name'] = name
            break

        print("\nDescribe your project goal in detail:")
        print("Before describing your goal, please consider:\n")
        print("- Target Platform/OS: (e.g., Windows, Web, Mobile, Chrome Extension)")
        print("- Preferred Programming Language/Frameworks: (Optional)")
        print("- Are there any specific technologies to be used or avoided?")
        print("- Desired Design Style: (e.g., Sleek, Modern, Minimalistic, etc.)")
        print("If you don't have specific preferences, the PRD will focus on architecture and design principles.\n")
        
        info['goal'] = self.get_multiline_input("\nEnter your project goal")

        # Check if technology is specified in the goal
        tech_keywords = ['react', 'angular', 'vue', 'svelte', 'native', 'android', 'ios', 'swift',
                        'kotlin', '.net', 'django', 'flask', 'express', 'node', 'spring', 'rails',
                        'chrome extension', 'next.js', 'flutter']

        has_tech_specified = any(keyword in info['goal'].lower() for keyword in tech_keywords)
        info['tech_specified'] = has_tech_specified
        
        # Always use cache without prompting
        if hasattr(self, 'response_cache') and len(self.response_cache) > 0:
            # Silently use cache
            info['use_cache'] = True
            logging.debug(f"Using {len(self.response_cache)} cached responses automatically")
        else:
            # No cache entries found or cache not initialized
            info['use_cache'] = True
            if not hasattr(self, 'response_cache'):
                self.response_cache = {}
        
        # Store the project info in the instance
        self.project_info = info
            
        return info

    def generate_prd_sections(self, project_info: dict) -> List[str]:
        """Generate all PRD sections."""
        sections = []
        self.cache_hits = []
        self.api_calls_made = 0
        self.tokens_saved = {'input': 0, 'output': 0}

        # 1. Generate overview prompt and generate overview section
        overview_prompt = self._generate_overview_prompt(project_info)
        
        # Get the estimated token count for the prompt
        estimated_tokens = self._estimate_tokens(overview_prompt)
        
        self.spinner.start(f"Generating section: overview... (Est. tokens: {estimated_tokens})")
        overview_section = self._generate_section(overview_prompt, "overview")
        sections.append(overview_section)
        self.spinner.stop()  # Stop spinner temporarily to show phases

        # 2. Extract phases from overview and generate each phase
        phases = self._extract_phases_from_overview(overview_section)
        
        # Print the phases in a clean, user-friendly format
        if phases:
            print("\n📋 Implementation Phases Identified:")
            print("─" * 50)
            for i, phase in enumerate(phases, 1):
                print(f"{i}. {phase['name']}")
            print("─" * 50 + "\n")
        else:
            print("\n⚠️ No implementation phases identified. Using default phases.\n")

        # Log more detailed phase info only when in debug mode
        for i, phase in enumerate(phases, 1):
            logging.debug(f"Phase {i}: {phase['name']}")
            logging.debug(f"Description: {phase['description'][:100]}...")
        
        # Special handling for strange output
        if len(phases) == 1 and phases[0]['name'] == "Email Translator":
            print("\n⚠️ WARNING: Only one generic phase was identified. This may not be correct.")
            print("Review the Implementation Phases section in your PRD overview.")
            print("Format should be: '1. **Phase Name**: Description'\n")

        # Initialize checkpoint tracking
        checkpoint_file = CACHE_DIR / f"checkpoint_{hashlib.md5(project_info['name'].encode()).hexdigest()}.json"
        checkpoints = {}
        
        # Try to load existing checkpoints
        try:
            if checkpoint_file.exists():
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoints = json.load(f)
                    # Verify checkpoints match current project
                    if checkpoints.get('project_name') == project_info['name']:
                        sections = checkpoints.get('sections', [])
                        logging.info(f"Loaded {len(sections)} sections from checkpoint")
                    else:
                        # If project name doesn't match, don't use the checkpoint
                        checkpoints = {'project_name': project_info['name'], 'sections': sections}
            else:
                # Initialize new checkpoint
                checkpoints = {'project_name': project_info['name'], 'sections': sections}
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            # Initialize new checkpoint
            checkpoints = {'project_name': project_info['name'], 'sections': sections}

        # Define checkpoint save function
        def save_checkpoint():
            """Save current progress to checkpoint file."""
            try:
                checkpoints['sections'] = sections
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoints, f)
                logging.info(f"Saved checkpoint with {len(sections)} sections")
            except Exception as e:
                logging.error(f"Failed to save checkpoint: {e}")

        # 3. Generate phase-specific sections
        start_phase = len(sections)  # Skip phases we already have from checkpoint
        for i, phase in enumerate(phases, 1):
            if i <= start_phase - 1:
                # Skip phases we already have
                continue
                
            # Format progress indicator and start spinner
            progress = f"{i}/{len(phases)}"
            self.spinner.start(f"Generating phase {progress}: {phase['name']}...")
            
            # Generate section for this phase
            phase_prompt = self._generate_phase_prompt(project_info, phase, i, len(phases))
            phase_section = self._generate_section(phase_prompt, f"phase_{i}")
            sections.append(phase_section)
            
            # Save checkpoint after each phase
            save_checkpoint()

        # Remove checkpoint file after successful completion
        try:
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                logging.info("Removed checkpoint file after successful completion")
        except Exception as e:
            logging.error(f"Failed to remove checkpoint file: {e}")

        self.spinner.stop()
        return sections

    def _generate_overview_prompt(self, project_info: dict) -> str:
        """Generate a dynamic overview prompt based on project information."""
        # Extract the essential information to minimize prompt size
        project_name = project_info['name']
        project_goal = project_info.get('goal', '')
        tech_specified = project_info.get('tech_specified', False)
        
        # Create a shorter base guidelines version to save tokens
        base_guidelines = (
            "Follow these guidelines:\n"
            "- Clean Architecture: separation of concerns, DI, interfaces, testability\n"
            "- Incremental implementation: start simple, verify steps, add complexity gradually\n"
            "- Code Structure: clear naming, consistent patterns, error handling, logging\n"
            "- Security: no hardcoded credentials, input validation, secure error handling\n"
            "- Testing: unit/integration/e2e tests with clear criteria"
        )

        # Generate the project overview section with an optimized prompt
        overview_prompt = (
            f"Generate a high-level project overview for {project_name}.\n\n"
            f"Project Details:\n"
            f"Name: {project_name}\n"
            f"Goal: {project_goal}\n"
            f"Design: Sleek, modern, minimalistic.\n\n"
        )

        # Conditionally add technology suggestion only if needed
        if not tech_specified:
            overview_prompt += (
                f"No specific technology specified; focus on architecture and design principles.\n"
                f"Suggest general technology categories that would fit the project.\n\n"
            )

        overview_prompt += (
            f"{base_guidelines}\n\n"
            "1. Provide a concise 1-2 sentence description of the project.\n"
            "2. Suggest 4-5 implementation phases, starting with a minimal working prototype.\n"
            "3. Include: Project Overview (core functionality, architecture, components, boundaries)\n"
            "4. Suggest general technology categories and architecture diagram (using Mermaid)\n"
            "5. Format in Markdown. Focus on architecture and design, not implementation details."
        )
        
        return overview_prompt

    def _generate_phase_prompt(self, project_info: dict, phase: dict, phase_number: int, total_phases: int) -> str:
        """Generate a dynamic phase prompt based on project information and phase details."""
        project_name = project_info['name']
        
        # Create a more concise and focused phase prompt
        phase_prompt = (
            f"Create implementation plan for Phase {phase_number}: {phase['name']} of {project_name}.\n\n"
            f"Phase Description: {phase.get('description', 'No description provided')}\n\n"
            f"Requirements:\n"
            f"1. Implementation steps (small, incremental, verifiable)\n"
            f"2. Component specifications using Clean Module Structure\n"
            f"3. Interface definitions\n"
            f"4. Testing criteria\n\n"
            f"Key points:\n"
            f"- Verify each step by running the app\n"
            f"- Follow Clean Architecture principles\n"
            f"- Maintain consistent architecture\n"
            f"- Format in Markdown\n"
        )
        
        return phase_prompt

    def _extract_phases_from_overview(self, overview_section: str) -> List[Dict[str, str]]:
        """Extract phase information from the overview section."""
        if not overview_section:
            logging.error("Empty overview section provided for phase extraction")
            return []
            
        try:
            phases = []
            
            # First, look for a section specifically about implementation phases
            implementation_section_match = re.search(r'(?i)(?:^|\n)#+\s*(?:implementation|development)\s+(?:phases|stages|steps)[^\n]*(?:\n+(?:(?!\n#).)*)(?=\n#|\Z)', overview_section, re.DOTALL)
            
            if implementation_section_match:
                implementation_section = implementation_section_match.group(0)
                # Debug logging to see the actual implementation section content
                logging.debug("FOUND IMPLEMENTATION SECTION:")
                logging.debug("-" * 50)
                logging.debug(implementation_section)
                logging.debug("-" * 50)
                
                # Direct pattern for numbered lists with bold phase names (e.g., "1. **Foundation Phase**:")
                numbered_bold_pattern = re.compile(r'(?:^|\n)(\d+)\.\s+\*\*([^*\n:]+)[^*\n]*\*\*:?\s*([^\n]*)', re.MULTILINE)
                matches = numbered_bold_pattern.finditer(implementation_section)
                
                for match in matches:
                    phase_number = match.group(1)
                    phase_name = match.group(2).strip()
                    description = match.group(3).strip()
                    
                    # Debug logging for each match
                    logging.debug(f"MATCHED PHASE: {phase_number}. **{phase_name}**: {description}")
                    
                    # If there's more content until the next phase, include it in the description
                    start_pos = match.end()
                    next_phase = re.search(r'(?:^|\n)\d+\.\s+\*\*', implementation_section[start_pos:], re.MULTILINE)
                    next_heading = re.search(r'(?:^|\n)#+\s+', implementation_section[start_pos:], re.MULTILINE)
                    
                    # Find the end position of the current phase
                    if next_phase:
                        phase_end = start_pos + next_phase.start()
                    elif next_heading:
                        phase_end = start_pos + next_heading.start()
                    else:
                        phase_end = len(implementation_section)
                    
                    additional_description = implementation_section[start_pos:phase_end].strip()
                    if additional_description:
                        description = (description + " " + additional_description).strip()
                    
                    # Clean up description
                    description = re.sub(r'\s+', ' ', description).strip()
                    
                    if phase_name and len(phase_name) > 2:  # Ensure phase name is meaningful
                        phases.append({
                            'name': phase_name,
                            'description': description
                        })
            else:
                logging.debug("No implementation section found. Searching entire overview.")
                
            # If no phases found with the numbered bold pattern, try the Phase X: pattern
            if not phases and implementation_section_match:
                implementation_section = implementation_section_match.group(0)
                
                # Look for phase patterns within the implementation section
                phase_pattern = re.compile(r'(?:(?:^|\n)#+\s+|\n)(?:Phase|Stage|Step)\s+(\d+)[:\.]?\s+([^\n]+)', re.MULTILINE | re.IGNORECASE)
                matches = phase_pattern.finditer(implementation_section)
                
                for match in matches:
                    phase_number = match.group(1)
                    phase_name = match.group(2).strip()
                    
                    # Find the description by getting content until the next phase or heading
                    start_pos = match.end()
                    next_phase = re.search(r'(?:^|\n)(?:#+\s+|\n)(?:Phase|Stage|Step)\s+\d+[:\.]?', implementation_section[start_pos:], re.MULTILINE | re.IGNORECASE)
                    next_heading = re.search(r'(?:^|\n)#+\s+', implementation_section[start_pos:], re.MULTILINE)
                    
                    # Find the end position of the current phase
                    if next_phase:
                        phase_end = start_pos + next_phase.start()
                    elif next_heading:
                        phase_end = start_pos + next_heading.start()
                    else:
                        phase_end = len(implementation_section)
                    
                    description = implementation_section[start_pos:phase_end].strip()
                    
                    # Clean up description
                    description = re.sub(r'\s+', ' ', description).strip()
                    
                    if phase_name and len(phase_name) > 2:  # Ensure phase name is meaningful
                        phases.append({
                            'name': phase_name,
                            'description': description
                        })
            
            # If no phases found with the implementation section approach, try the numbered list approach
            if not phases:
                # Updated regex to look for numbered items that might be phases
                phase_pattern = re.compile(r'(?:^|\n)\d+\.\s+([A-Za-z][\w\s/\-&]{2,})\b[^\n]*?((?:\n\s+[\-\*]\s[^\n]+)+|:\s*([^\n]+))', re.MULTILINE)
                
                matches = phase_pattern.finditer(overview_section)
                for match in matches:
                    phase_name = match.group(1).strip()
                    description = match.group(2).strip() if match.group(2) else ""
                    
                    # Clean up the description - remove bullet points and extra spaces
                    description = re.sub(r'^\s*[\-\*]\s*', '', description, flags=re.MULTILINE)
                    description = re.sub(r'\s+', ' ', description).strip()
                    
                    if phase_name and len(phase_name) > 2:  # Ensure phase name is meaningful
                        phases.append({
                            'name': phase_name,
                            'description': description
                        })
            
            # If still no phases, look for headings that might be phases
            if not phases:
                # Look for headings or sections that likely represent phases
                semantic_pattern = re.compile(r'(?:^|\n)#+\s+((?:Phase|Stage|Step|Implementation|Development)[\w\s/\-&:]{3,})', re.MULTILINE | re.IGNORECASE)
                matches = semantic_pattern.finditer(overview_section)
                
                for match in matches:
                    phase_name = match.group(1).strip()
                    
                    # Skip if this is just the "Implementation Phases" heading itself
                    if re.search(r'(?i)(?:implementation|development)\s+(?:phases|stages|steps)', phase_name):
                        continue
                        
                    # Extract text following the heading until next heading or end
                    start_pos = match.end()
                    next_heading = re.search(r'(?:^|\n)#+\s+', overview_section[start_pos:], re.MULTILINE)
                    
                    if next_heading:
                        description = overview_section[start_pos:start_pos + next_heading.start()].strip()
                    else:
                        description = overview_section[start_pos:].strip()
                        
                    if phase_name and not any(p['name'] == phase_name for p in phases):
                        phases.append({
                            'name': phase_name,
                            'description': description
                        })
            
            # Add debug logging to show detected phases
            logging.info(f"Extracted phases: {[p['name'] for p in phases]}")
            
            # If still no phases, use default phases
            if not phases:
                logging.warning("Could not extract phases from overview. Using default phases.")
                return [
                    {"name": "Foundation/Prototype", "description": "Project setup, architecture, basic UI"},
                    {"name": "Core Functionality", "description": "Implementation of key features"},
                    {"name": "Enhanced Features", "description": "Additional features and improvements"},
                    {"name": "Polish & Optimization", "description": "Performance, UI refinement, testing"}
                ]
                
            # Add meaningful descriptions for empty descriptions
            for phase in phases:
                if not phase['description'] or len(phase['description']) < 10:
                    phase['description'] = f"Implementation of {phase['name']} features and functionality"
                    
            return phases
        except Exception as e:
            logging.error(f"Error extracting phases from overview: {e}")
            # Return a default set of phases as fallback
            return [
                {"name": "Foundation/Prototype", "description": "Project setup, architecture, basic UI"},
                {"name": "Core Functionality", "description": "Implementation of key features"},
                {"name": "Enhanced Features", "description": "Additional features and improvements"},
                {"name": "Polish & Optimization", "description": "Performance, UI refinement, testing"}
            ]

    def _backup_existing_folder(self, folder_path: Path) -> bool:
        """
        Backup an existing folder by renaming it with a timestamp suffix.
        Returns True if backup succeeded, False otherwise.
        """
        if not folder_path.exists():
            return True  # No backup needed
            
        # Create backup name with timestamp to allow multiple backups
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = Path(f"{folder_path}_{timestamp}")
        
        try:
            self.spinner.update_message(f"Backing up existing folder to {backup_path.name}...")
            folder_path.rename(backup_path)
            logging.info(f"Backed up existing folder to {backup_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to backup existing folder: {e}")
            return False
            
    def save_prd(self, sections: List[str], project_name: str):
        """Save PRD sections into Markdown files."""
        if not sections:
            logging.error("No PRD content to save.")
            return

        # Process sections to ensure proper formatting
        final_sections = []
        for section in sections:
            # Fix common formatting issues
            
            # 1. Ensure code blocks are properly closed
            # Count opening and closing code block markers
            open_markers = len(re.findall(r'```(?!\s*```)', section))
            close_markers = len(re.findall(r'(?<!`)```(?!\s*`)', section))
            
            processed_section = section
            
            # Add missing closing code block markers if needed
            if open_markers > close_markers:
                processed_section += '\n' + '```' * (open_markers - close_markers)
            
            # 2. Fix incorrectly formatted code blocks in bullet points
            # This regex finds code blocks within bullet points and ensures proper formatting
            bullet_code_pattern = r'(\s*-\s+.*?)(```\s*\n.*?```)'
            processed_section = re.sub(bullet_code_pattern, r'\1\n\2', processed_section, flags=re.DOTALL)
            
            # 3. Fix incomplete code markers
            processed_section = re.sub(r'(?<!\`)``(?!\`)', '```', processed_section)
            processed_section = re.sub(r'(?<!\`)`(?!\`)', '`', processed_section)
            
            final_sections.append(processed_section)

        sections = final_sections

        self.spinner.update_message("Creating PRD files...")
        
        # Create the output directory if it doesn't exist
        if not self._safe_create_directory(PRD_OUTPUT_PATH):
            return
            
        # Create project-specific directory within the output path
        dir_name = PRD_OUTPUT_PATH / f"{project_name.lower().replace(' ', '_')}_prd"
        
        # Backup existing folder if it exists
        if dir_name.exists():
            backup_success = self._backup_existing_folder(dir_name)
            if not backup_success:
                user_response = input(f"\nFolder {dir_name} already exists and couldn't be backed up. Overwrite? (y/N): ").strip().lower()
                if user_response != 'y':
                    print("PRD generation cancelled.")
                    return
                print("Continuing with overwrite...")
            else:
                print(f"\nExisting PRD folder was backed up. Continuing with new generation.")
        
        # Create the project directory
        if not self._safe_create_directory(dir_name):
            return

        try:
            # Create instructions.md with initial project overview and basic instructions
            instructions_file = dir_name / "instructions.md"
            try:
                with instructions_file.open('w', encoding='utf-8') as f:
                    f.write(f"# {project_name} - IDE AI Instructions\n\n")
                    f.write("## Project Definition\n\n")
                    f.write(sections[0])
                    f.write("\n\n## AI Development Assistant Guidelines\n\n")

                    # Include the full AI Assistant Guidelines in instructions.md
                    f.write("As your AI Development Assistant, please adhere to the following guidelines:\n\n")
                    f.write("### 1. Accuracy and Honesty\n")
                    f.write("- If you lack confirmed information, say explicitly that you don't have the info rather than fabricating details.\n")
                    f.write("- When your response isn't based on confirmed sources, begin with 'I think…', 'I believe…', or 'My understanding…' to clearly mark opinions or assumptions.\n\n")

                    f.write("### 2. Troubleshooting and Problem-Solving\n")
                    f.write("- Address issues directly with clear, numbered steps for resolution.\n")
                    f.write("- Favor systematic debugging approaches over quick fixes when appropriate.\n")
                    f.write("- Start with simple solutions before suggesting complex ones.\n")
                    f.write("- Help users interpret error messages and logs effectively.\n")
                    f.write("- Ensure that each point is mentioned only once without unnecessary repetition.\n\n")

                    f.write("### 3. Security Best Practices\n")
                    f.write("- Always suggest configurations or code that avoid hardcoding credentials.\n")
                    f.write("- Separate configurations from sensitive credentials storage.\n")
                    f.write("- Prioritize solutions that follow industry security standards.\n")
                    f.write("- Be explicit about potential security implications of suggested solutions.\n")
                    f.write("- Recommend proper input validation and sanitization to prevent injection attacks.\n")
                    f.write("- Suggest appropriate error handling that doesn't expose sensitive details.\n\n")

                    f.write("### 4. Development and Production Considerations\n")
                    f.write("- When updating features, be extremely careful not to disrupt existing functionality that's confirmed as working.\n")
                    f.write("- Have a clear chain of thought for the objective and ensure modifications are strictly targeted to requested changes.\n")
                    f.write("- When multiple solutions exist, prioritize: (1) security, (2) maintainability, (3) performance, unless specified otherwise.\n")
                    f.write("- For reusable modules, always implement the interface-implementation pattern when supported by the language.\n")
                    f.write("- Recommend appropriate test strategies alongside implementation code.\n")
                    f.write("- Include guidance on observability (logging, metrics, monitoring) when relevant.\n\n")

                    f.write("### 5. Performance Optimization\n")
                    f.write("- Focus on identifying performance bottlenecks before recommending optimizations.\n")
                    f.write("- Balance between avoiding premature optimization and addressing necessary performance considerations.\n")
                    f.write("- Provide guidelines for resource-intensive operations (caching, pagination, asynchronous processing).\n\n")

                    f.write("### 6. Code and Documentation\n")
                    f.write("- Format code with appropriate syntax highlighting and indentation.\n")
                    f.write("- Include helpful comments in code examples, especially for complex operations.\n")
                    f.write("- Reference relevant documentation when applicable.\n")
                    f.write("- When suggesting libraries or tools, briefly mention their advantages and limitations.\n")
                    f.write("- Document interfaces thoroughly, clearly defining contracts, expected behaviors, and implementation requirements.\n")
                    f.write("- Ensure commands are generated for Windows PowerShell (e.g., using ; instead of &&, like cd src-tauri ; cargo check).\n\n")

                    f.write("### 7. Style and Tone\n")
                    f.write("- Keep responses concise, clear, and straightforward.\n")
                    f.write("- Use humor where appropriate, particularly when it helps illustrate a concept.\n")
                    f.write("- Avoid redundant explanations and keep answers direct.\n")
                    f.write("- Adapt formality to match the complexity and seriousness of the topic.\n")
                    f.write("- Ensure all responses are in Australian English.\n\n")

                    f.write("### 8. Design Philosophy\n")
                    f.write("- Use interface-based programming for all reusable modules.\n")
                    f.write("- Decouple implementations from interfaces to enable easy substitution.\n")
                    f.write("- Design interfaces based on behavior rather than implementation details.\n")
                    f.write("- Follow dependency inversion principle by depending on abstractions (interfaces) rather than concrete implementations.\n")
                    f.write("- Promote composition over inheritance when designing object relationships.\n")
                    f.write("- Encourage single responsibility principle for modules and functions.\n\n")

                    f.write("### 9. Cross-Platform Awareness\n")
                    f.write("- While focused on Windows 11, be aware of potential cross-platform issues when relevant.\n")
                    f.write("- Include awareness of path differences and environment-specific issues in suggestions.\n\n")

                    f.write("### 10. UI/UX Considerations (when applicable)\n")
                    f.write("- Suggest modern, accessible UI components following established design patterns.\n")
                    f.write("- Consider accessibility and internationalization in user interface recommendations.\n\n")

                    f.write("### 11. API Design Principles\n")
                    f.write("- Follow RESTful API best practices including proper resource naming, HTTP methods, and status codes.\n")
                    f.write("- Design GraphQL schemas with clear types, resolvers, and queries that minimize overfetching.\n")
                    f.write("- Implement consistent error handling and response formats across all API endpoints.\n")
                    f.write("- Use API versioning strategies (URI path, query parameter, or header-based) to maintain backward compatibility.\n")
                    f.write("- Document APIs using standards like OpenAPI/Swagger for REST or GraphQL Schema Documentation.\n")
                    f.write("- Implement proper rate limiting, authentication, and authorization mechanisms.\n")
                    f.write("- Design APIs for optimal caching at various levels (client, CDN, server).\n")
                    f.write("- Consider using hypermedia (HATEOAS) for REST APIs to improve discoverability.\n")
                    f.write("- Structure request/response payloads efficiently, with clear validation requirements.\n")
                    f.write("- Develop comprehensive API testing strategies including contract testing.\n\n")

                    f.write("### 12. Data Privacy\n")
                    f.write("- Implement data minimization by collecting and processing only necessary information.\n")
                    f.write("- Design systems with privacy by default and privacy by design principles.\n")
                    f.write("- Include mechanisms for data subject access requests (DSARs) and the right to be forgotten.\n")
                    f.write("- Apply appropriate anonymization techniques (k-anonymity, differential privacy) for analytics data.\n")
                    f.write("- Implement data retention policies and automated purging mechanisms.\n")
                    f.write("- Use pseudonymization for data that requires identification capabilities without exposing PII.\n")
                    f.write("- Create clear audit trails for data access and modifications to sensitive information.\n")
                    f.write("- Design explicit consent mechanisms with granular permission controls.\n")
                    f.write("- Ensure cross-border data transfer compliance with relevant regulations (GDPR, CCPA, etc.).\n")
                    f.write("- Implement data breach detection, notification, and response procedures.\n")
                    f.write("- Protect sensitive data with appropriate encryption in transit and at rest.\n\n")

                    f.write("### 13. Clean Architecture Module Structure (Clean Module Structure)\n")
                    f.write("Follow the Clean Module Structure for module structure to maintain Clean Architecture principles. This involves separating interfaces from implementations, using factory patterns, and ensuring modules are independently testable. Refer to `project_prompt.md` for language-specific directory structure examples.\n\n")

                    f.write("### 14. Pattern Reference\n")
                    f.write("The 'Clean Module Structure' refers to the Clean Architecture Module Structure defined in section 13. When instructed to use this pattern, apply the appropriate language-specific directory structure and follow all principles outlined in that section.\n\n")

                    f.write("### 15. Pattern and Architecture Continuity\n")
                    f.write("- Maintain all existing design patterns and software architecture (including the Clean Module Structure) when making any changes or updates.\n")
                    f.write("- Never deviate from established architectural patterns even for minor modifications.\n")
                    f.write("- Ensure all new code follows the same architectural principles as the existing codebase.\n")
                    f.write("- When extending functionality, preserve the interface-implementation separation and module structure.\n")
                    f.write("- Validate that changes preserve the integrity of the overall system architecture.\n")
                    f.write("- If architectural changes are necessary, explicitly discuss the rationale before implementation.\n\n")


                    f.write("### Development Workflow\n")
                    f.write("- Implement features iteratively according to the phase plan, one step at a time.\n") # Baby steps workflow
                    f.write("- After each implementation step, start the app and verify the implementation.\n") # Verification workflow
                    f.write("- Write tests before or alongside implementation.\n")
                    f.write("- Document all key decisions and architectural choices.\n")
                    f.write("- Review code quality and test coverage before completing each phase.\n")
                    f.write("- Maintain consistency with established patterns (Clean Module Structure).\n\n") # Clean Module Structure in workflow

                    f.write("## IDE AI Instructions\n\n")
                    f.write("As the AI assistant for this project:\n")
                    f.write("1. Follow one phase at a time as provided by the developer, step by step.\n") # Baby steps for AI
                    f.write("2. After each implementation step, provide instructions to start the app and verify the implemented functionality.\n") # Verification for AI
                    f.write("3. Strictly adhere to the architectural guidelines defined in project_prompt.md and Clean Module Structure.\n") # Clean Module Structure for AI
                    f.write("4. Maintain proper encapsulation between modules as defined in the architecture (Clean Module Structure).\n") # Clean Module Structure for AI
                    f.write("5. Use consistent technology stacks throughout the development process.\n")
                    f.write("6. Only implement features defined in the current phase.\n")
                    f.write("7. Suggest refactoring when code deviates from the defined architecture or Clean Module Structure.\n") # Clean Module Structure for AI refactoring
                    f.write("8. Maintain consistency with previously established patterns and conventions.\n")
                    f.write("9. Provide complete implementations rather than skeleton code.\n")
                    f.write("10. Include appropriate tests for all new functionality.\n")
                    f.write("11. Validate that each phase is complete before moving to the next, including starting the app and verifying.\n\n") # Verification for AI phase completion

                    f.write("## DO NOT Begin Implementation Yet\n")
                    f.write("Wait for Phase 1 document before starting implementation.\n")
                    f.write("Refer to project_prompt.md for detailed technical architecture and Clean Module Structure.\n") # Clean Module Structure in DO NOT
            except IOError as e:
                logging.error(f"Failed to write to {instructions_file}: {e}")
                return

            # Create project_prompt.md with focused project information
            project_prompt_file = dir_name / "project_prompt.md"
            try:
                with project_prompt_file.open('w', encoding='utf-8') as f:
                    f.write(f"# {project_name} - Project Architecture Blueprint\n\n")
                    
                    # Extract project vision and core goals from the overview
                    overview_lines = sections[0].split('\n')

                    # Try to extract the 1-2 sentence description
                    description = ""
                    for line in overview_lines[:20]:
                        if line.strip() and not line.startswith('#') and not line.startswith('- ') and len(line.split()) >= 5:
                            description = line.strip()
                            break

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

                    # Vision and Core Purpose (Concise)
                    f.write("## Core Purpose\n")
                    if description:
                        f.write(f"{description}\n\n")
                    else:
                        f.write("[Primary purpose of the application in 1-2 sentences]\n\n")

                    # Extract mermaid diagram for architecture visualization
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

                    # Key Architectural Principles - More concise but still comprehensive
                    f.write("## Architecture and Implementation Rules\n")
                    f.write("The following rules are **mandatory** for implementation:\n\n")
                    
                    f.write("### 1. Clean Architecture Layers\n")
                    f.write("- **Presentation**: UI components, view models (depends on Application)\n")
                    f.write("- **Application**: Use cases, services, controllers (depends on Domain)\n") 
                    f.write("- **Domain**: Entities, value objects, repository interfaces (no dependencies)\n")
                    f.write("- **Infrastructure**: Data sources, external services, repository implementations\n\n")

                    f.write("### 2. Core Principles\n")
                    f.write("- **Dependency Rule**: Dependencies point inward (outer layers depend on inner layers)\n")
                    f.write("- **Interface Segregation**: Interfaces define clear contracts between layers\n")
                    f.write("- **Dependency Injection**: Components receive dependencies rather than creating them\n")
                    f.write("- **Single Responsibility**: Each component has one reason to change\n")
                    f.write("- **Testability**: Business logic must be testable in isolation\n\n")

                    f.write("### 3. Module Structure (Mandatory)\n")
                    f.write("All modules MUST follow this structure:\n")
                    f.write("```\n")
                    f.write("<module_name>/\n")
                    f.write("├── public/           # Public interfaces defining the contract\n")
                    f.write("├── service/          # Interface and implementation separated\n")
                    f.write("├── factory/          # Factory for creating implementations\n")
                    f.write("└── controller/       # (If needed) Controller connecting to UI\n")
                    f.write("```\n\n")

                    # Main Component Interactions - Visual is worth 1000 words
                    f.write("### 4. Component Interactions\n")
                    if mermaid_lines:
                        f.write("\n".join(mermaid_lines))
                    else:
                        # Simpler default diagram
                        f.write("```mermaid\n")
                        f.write("flowchart TB\n")
                        f.write("    UI[UI Components] --> VM[View Models] --> UC[Use Cases]\n") 
                        f.write("    UC --> E[Entities]\n")
                        f.write("    UC --> RI[Repository Interfaces]\n")
                        f.write("    RI <-- DI[Dependency Injection] --> IMPL[Repository Implementations]\n")
                        f.write("    IMPL --> EXT[External Services/APIs]\n")
                        f.write("```\n")

                    f.write("\n")

                    # Tech Stack - Only the essential parts
                    f.write("## Tech Stack\n")
                    if tech_stack:
                        for tech in tech_stack[:4]:  # Limit to top 4 for brevity
                            f.write(f"{tech}\n")
                    else:
                        # Generic tech stack - more concise
                        f.write("- **Frontend**: [Framework/language]\n")
                        f.write("- **Backend**: [Backend technology if applicable]\n")
                        f.write("- **State Management**: [Approach]\n")
                        f.write("- **Data Storage**: [Solution]\n")

                    f.write("\n")

                    # Project Structure - Keep it simple and focused
                    f.write("## Project Structure\n")
                    f.write("```\n")
                    f.write(f"{project_name.lower().replace(' ', '_')}/\n")
                    f.write("├── src/\n")
                    f.write("│   ├── presentation/     # UI components, views\n") 
                    f.write("│   ├── application/      # Use cases, controllers\n")
                    f.write("│   ├── domain/           # Business logic, entities\n")
                    f.write("│   │   └── modules/      # Domain modules following structure in §3\n")
                    f.write("│   └── infrastructure/   # External integrations\n")
                    f.write("└── tests/                # Tests by layer\n")
                    f.write("```\n\n")

                    # Implementation Verification - Critical for correct implementation
                    f.write("## Implementation Verification\n")
                    f.write("- After implementing each step, start the application to verify functionality\n")
                    f.write("- Ensure all layers maintain proper separation of concerns\n")
                    f.write("- Validate that each module follows the mandatory structure in §3\n")
                    f.write("- Confirm all dependencies flow according to Clean Architecture rules\n")
                    f.write("- Test each component in isolation with appropriate mocks\n\n")

                    # Security and Data Guidelines - Crucial principles
                    f.write("## Security and Data Guidelines\n")
                    f.write("- Never hardcode credentials; use environment variables or secure stores\n")
                    f.write("- Validate all user inputs at domain boundaries\n")
                    f.write("- Implement proper error handling without exposing sensitive details\n")
                    f.write("- Use secure communication protocols (HTTPS, WSS)\n")
                    f.write("- Minimize data collection and implement data deletion capabilities\n")
            except IOError as e:
                logging.error(f"Failed to write to {project_prompt_file}: {e}")
                return

            # Create phase files
            for i, phase in enumerate(sections[1:], 1):
                phase_file = dir_name / f"phase_{i:02d}.md"
                phase_id = f"phase_{i:02d}"  # Create a consistent phase identifier
                
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
                        "Refer to 'Tech Stack Summary' in `project_prompt.md` for technology-specific setup.\n\n"
                        "**Verification**: After setup, start the app to ensure basic project setup is correct and the app runs without errors.\n\n" # Verification in Phase 1 init
                    )

                    # Use regex to find implementation steps section and insert initialization note
                    implementation_pattern = re.compile(r'(#+\s*(?:1\.|Implementation Steps).*?)(\n+)', re.IGNORECASE)
                    match = implementation_pattern.search(phase)
                    
                    if match:
                        # Insert after the heading and any newlines that follow it
                        phase = phase[:match.end()] + project_init_note + phase[match.end():]
                    else:
                        # If no implementation steps section found, add it at the beginning
                        phase = "## 1. Implementation Steps\n\n" + project_init_note + phase

                # Replace software_architecture.md with project_prompt.md
                phase = phase.replace("software_architecture.md", "project_prompt.md")

                try:
                    with phase_file.open('w', encoding='utf-8') as f:
                        f.write(f"# Phase {i} Implementation Plan\n\n")
                        f.write(f"<!-- Phase ID: {phase_id} -->\n\n")  # Add phase identifier as comment

                        # Architecture reminder - Clean Module Structure included
                        f.write("## ⚠️ IMPORTANT: Architectural Compliance (Clean Architecture, Clean Module Structure)\n")
                        f.write("Code MUST follow architecture in `project_prompt.md` and Clean Module Structure.\n")
                        f.write("These rules are foundational and cannot be modified.\n")
                        f.write("The file `instructions.md` contains AI development assistant guidelines.\n")
                        f.write("Refer to `project_prompt.md` for architecture and Clean Module Structure details.\n")
                        f.write("AI assistant MUST adhere to these guidelines.\n\n")

                        f.write("## Implementation Requirements\n\n")
                        if i > 1:
                            prev_phase_id = f"phase_{i-1:02d}"  # Previous phase identifier
                            f.write(f"- {prev_phase_id} complete and verified (by starting app)\n")  # Use phase identifier
                            f.write("- All previous tests passing\n")
                        f.write("- Follow implementation steps in order, step by step.\n") # Baby steps in phase files
                        f.write("- After each step, start the app and verify implementation.\n") # Verification step in phase files
                        f.write("- Confirm step completion.\n")
                        f.write("- Maintain consistent architecture and Clean Module Structure.\n") # Clean Module Structure in phase requirements
                        f.write("- Get IDE AI verification before proceeding.\n\n")
                        f.write(phase)
                        f.write("\n\n## Completion Checklist\n\n")
                        f.write("- [ ] All implementation steps completed (incremental steps)\n")
                        f.write("- [ ] After each step, app started and functionality verified\n") # Verification checklist
                        f.write("- [ ] All tests passing\n")
                        f.write("- [ ] Code reviewed and documented\n")
                        f.write("- [ ] Architectural compliance (Clean Architecture, Clean Module Structure) verified\n") # Clean Module Structure in checklist
                        f.write("- [ ] IDE AI verification received\n")
                except IOError as e:
                    logging.error(f"Failed to write to {phase_file}: {e}")
                    logging.warning(f"Skipping phase {i} due to file write error")
                    continue

            self.spinner.stop()
            
            print(f"\n✅ PRD successfully generated!")
            print(f"\n📁 Files created in '{dir_name}' directory.\n")
            
            print("\n📋 NEXT STEPS:")
            print("─" * 50)
            print("1. Share instructions.md with your IDE AI assistant")
            print("2. Reference project_prompt.md for architecture details")
            print("3. Start implementation with phase_01.md content")
            print("4. Follow each phase step by step, verifying as you go")
            print("5. Get AI verification before moving to next phase")
            print("─" * 50)
            
            # Display cache statistics in a cleaner format
            if hasattr(self, 'api_calls_made'):
                cache_hits = len(self.cache_hits) if hasattr(self, 'cache_hits') else 0
                print(f"\n📊 Generation Statistics")
                print(f"• Model: {self.model}")
                print(f"• API calls: {self.api_calls_made}")
                print(f"• Cache hits: {cache_hits}")
                print(f"• Total cache entries: {len(self.response_cache)}")
                
                # Show token statistics without cost information
                if hasattr(self, 'tokens_saved'):
                    input_tokens_saved = self.tokens_saved.get('input', 0)
                    output_tokens_saved = self.tokens_saved.get('output', 0)
                    total_tokens_saved = input_tokens_saved + output_tokens_saved
                    
                    if total_tokens_saved > 0:
                        print(f"• Tokens saved: {input_tokens_saved:,} input, {output_tokens_saved:,} output")
                        print(f"• Total tokens saved: {total_tokens_saved:,}")

        except Exception as e:
            # Use try-except to prevent errors if spinner is already stopped
            try:
                if hasattr(self, 'spinner') and self.spinner.spinning:
                    self.spinner.stop()
            except:
                pass
            logging.error(f"Error saving PRD files: {e}")

    def _safe_create_directory(self, directory_path: Path) -> bool:
        """
        Safely create a directory with proper error handling.
        Returns True on success, False on failure.
        """
        try:
            directory_path.mkdir(parents=True, exist_ok=True)
            return True
        except PermissionError:
            logging.error(f"Permission denied when creating directory {directory_path}. Check your file permissions.")
            return False
        except OSError as e:
            logging.error(f"Failed to create directory {directory_path}: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error creating directory {directory_path}: {e}")
            return False


def clear_screen():
    """Clear the terminal screen."""
    print("\033[H\033[J", end="")

def print_header():
    """Print the application header."""
    clear_screen()
    print("==================================================")
    print("           Dynamic Phased PRD Generator           ")
    print("==================================================")
    print("")
    
    # Set logger to WARNING for normal operation (only show warnings and errors by default)
    logging.getLogger().setLevel(logging.WARNING)

def main():
    """Main function to run the PRD generator."""
    try:
        # Signal handling for graceful exit
        signal.signal(signal.SIGINT, signal_handler)
        
        # Get API key and model from environment or user
        model, api_key = load_environment()
        
        # Initialize the generator with API key
        generator = PRDGenerator(model, api_key)
        
        # Get project info from user
        project_info = generator.get_project_info()
        
        # Generate PRD sections
        sections = generator.generate_prd_sections(project_info)
        
        if not sections:
            print("\nFailed to generate PRD. Please check the logs for errors.")
            return
            
        # Save PRD to files
        generator.save_prd(sections, project_info['name'])
        
    except KeyboardInterrupt:
        print("\nReceived SIGINT. Cleaning up...")
        if 'generator' in locals() and hasattr(generator, 'spinner'):
            try:
                if generator.spinner.spinning:
                    generator.spinner.stop()
            except:
                pass
                
        print("Saving cache before exit...")
        if 'generator' in locals() and hasattr(generator, '_save_cache'):
            generator._save_cache()
    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
        print(f"\nError: {e}. See logs for details.")
        if 'generator' in locals() and hasattr(generator, 'spinner'):
            try:
                if generator.spinner.spinning:
                    generator.spinner.stop()
            except:
                pass


if __name__ == "__main__":
    main()