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
3. Open 'phase_00.md' and provide its content as the initial prompt to your IDE AI chat.
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
import argparse
import shutil
from typing import List, Tuple, Dict
from pathlib import Path
import re
import json
import hashlib

import anthropic
from dotenv import load_dotenv

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
            f.write("CLAUDE_MODEL=claude-3-opus-20240229\n")
            f.write("ANTHROPIC_API_KEY=your-api-key-here\n")
        logging.info("Please update the .env file with your API key and run the script again.")
        sys.exit(1)

    model = os.getenv('CLAUDE_MODEL')
    api_key = os.getenv('ANTHROPIC_API_KEY')

    if not model or not api_key or api_key == 'your-api-key-here':
        logging.error("Please set CLAUDE_MODEL and ANTHROPIC_API_KEY in your .env file")
        sys.exit(1)
        
    # Get PRD path from environment or use default
    prd_path_str = os.getenv('PRD_PATH')
    if prd_path_str:
        prd_path = Path(prd_path_str)
    else:
        prd_path = Path.home() / "prds"
    
    # Ensure PRD directory exists
    prd_path.mkdir(exist_ok=True, parents=True)
    
    # Get cache path from environment or use default
    cache_path_str = os.getenv('CACHE_PATH')
    if cache_path_str:
        cache_path = Path(cache_path_str)
    else:
        cache_path = Path.home() / ".prd_generator" / "cache"
    
    # Ensure cache directory exists
    cache_path.mkdir(exist_ok=True, parents=True)
    
    logging.info(f"Using PRD output path: {prd_path}")
    logging.info(f"Using cache directory: {cache_path}")

    return model, api_key, prd_path, cache_path

class ProgressSpinner:
    """Progress spinner for console."""
    def __init__(self):
        self.spinning = False
        self.counter = 0
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.current_message = ""

    def spin(self):
        while self.spinning:
            sys.stdout.write(f"\r{self.spinner_chars[self.counter]} {self.current_message}")
            sys.stdout.flush()
            self.counter = (self.counter + 1) % len(self.spinner_chars)
            time.sleep(0.1)

    def start(self, message=""):
        self.current_message = message
        self.spinning = True
        threading.Thread(target=self.spin, daemon=True).start()

    def update_message(self, message):
        self.current_message = message
        sys.stdout.write("\r" + " " * (len(self.spinner_chars) + len(self.current_message) + 2) + "\r")
        sys.stdout.write(f"{self.spinner_chars[self.counter]} {self.current_message}")
        sys.stdout.flush()

    def stop(self):
        self.spinning = False
        sys.stdout.write("\r" + " " * (len(self.spinner_chars) + len(self.current_message) + 2) + "\r")
        sys.stdout.flush()


class PRDGenerator:
    """PRD Generator Class."""
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
            
            # Load cache statistics
            self._init_cache()
        except Exception as e:
            logging.error(f"Error initializing Anthropic client: {e}")
            sys.exit(1)
            
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

    def _init_cache(self):
        """Initialize the cache system"""
        try:
            # Count existing cache entries
            cache_files = list(self.cache_path.glob("*.json"))
            self.total_cache_entries = len(cache_files)
            logging.info(f"Found {self.total_cache_entries} existing cache entries")
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
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            self.total_cache_entries += 1
        except Exception as e:
            logging.warning(f"Error saving to cache: {e}")

    def _estimate_tokens(self, prompt: str) -> int:
        """Estimate token count for a prompt with fallback to character-based estimation."""
        try:
            # First try the official method if available
            if hasattr(self.client.messages, 'count_tokens'):
                return self.client.messages.count_tokens(prompt)
            
            # Fallback to character-based estimation
            char_count = len(prompt)
            estimated_tokens = int(char_count * TOKENS_PER_CHAR)
            logging.info(f"Using character-based token estimation: {estimated_tokens} tokens")
            return estimated_tokens
        except Exception as e:
            logging.warning(f"Token estimation failed: {e}. Using default token limit.")
            return DEFAULT_MAX_TOKENS

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
            self.spinner.update_message(f"Using cached response for: {identifier}... (Tokens saved: {input_tokens + output_tokens})")
            return cached_response
        
        # Estimate required tokens
        estimated_tokens = self._estimate_tokens(prompt)
        max_tokens = max(estimated_tokens * 2, DEFAULT_MAX_TOKENS)  # Double the input tokens as safety margin
        
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
            except Exception as e:
                retries += 1
                delay = BASE_DELAY * retries
                logging.error(f"Error generating section '{identifier}': {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
        logging.error(f"Failed to generate section '{identifier}' after {MAX_RETRIES} retries.")
        return f"<!-- Failed to generate section: {identifier} -->"

    def get_multiline_input(self, prompt: str) -> str:
        """Get multiline input from user."""
        print(f"\n{prompt} (Press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if line.strip() == "" and lines:
                break
            lines.append(line)
        if not lines:
            print("No input received; please provide some details.")
            return self.get_multiline_input(prompt)
        return "\n".join(lines)

    def get_project_info(self) -> dict:
        """Collect project information from user."""
        # No need to call print_header() here, it's called in main()
        info = {}

        info['name'] = input("\nEnter project name: ").strip()

        print("\nDescribe your project goal in detail:")
        print("Before describing your goal, please consider:\n")
        print("- Target Platform/OS: (e.g., Windows, Web, Mobile, Chrome Extension)")
        print("- Preferred Programming Language/Frameworks: (Optional)")
        print("- Are there any specific technologies to be used or avoided?")
        print("- Desired Design Style: (e.g., Sleek, Modern, Minimalistic, etc.)")
        print("If you don't have specific preferences, the PRD will focus on architecture and design principles.\n")
        info['goal'] = self.get_multiline_input("Enter your project goal")

        # Check if technology is specified in the goal - keep this logic if needed for conditional prompts later
        tech_keywords = ['react', 'angular', 'vue', 'svelte', 'native', 'android', 'ios', 'swift',
                        'kotlin', '.net', 'django', 'flask', 'express', 'node', 'spring', 'rails',
                        'chrome extension', 'next.js', 'flutter']

        has_tech_specified = any(keyword in info['goal'].lower() for keyword in tech_keywords)
        info['tech_specified'] = has_tech_specified

        return info

    def generate_prd_sections(self, project_info: dict) -> List[str]:
        """Generate PRD sections dynamically."""
        sections = []
        base_guidelines = (
            "Follow these specific guidelines when creating the PRD:\n"
            "1. Maintain Clean Architecture:\n"
            "   - Clear separation of concerns\n"
            "   - Dependency injection\n"
            "   - Interface-based design\n"
            "   - Testable components\n\n"
            "2. Implementation Approach:\n"
            "   - Start with simple, working application first\n"
            "   - Verify each major step with a working build by starting the app\n" # Added verification by starting app
            "   - Add complexity incrementally, one baby step at a time\n" # Added baby step approach
            "   - Keep each phase testable\n\n"
            "3. Code Structure:\n"
            "   - Clear naming conventions\n"
            "   - Consistent patterns (Cosmos Pattern for modules)\n" # Added Cosmos Pattern
            "   - Proper error handling\n"
            "   - Comprehensive logging\n\n"
            "4. Security Best Practices:\n" # Added Security Guidelines
            "   - Avoid hardcoding credentials; use environment variables or secure configuration\n"
            "   - Input validation and sanitization\n"
            "   - Secure error handling (avoid exposing sensitive info)\n\n"
            "5. Testing Requirements:\n"
            "   - Unit tests for core functionality\n"
            "   - Integration tests for components\n"
            "   - End-to-end tests for features\n"
            "   - Clear test criteria"
        )

        ai_guidelines_ref = (
            "\n\n**Reference AI Development Assistant Guidelines:**\n"
            "The generated PRD and phase implementation plans are designed to be used with an AI development assistant. "
            "Ensure the AI assistant adheres to the following principles throughout the development process:\n\n"
            "**AI Development Assistant Guidelines Summary (Refer to phase_00.md for full details):**\n"
            "- **Accuracy & Honesty:** Avoid fabrication, mark opinions clearly.\n"
            "- **Troubleshooting:** Provide numbered, clear steps; favor systematic debugging; start simple.\n"
            "- **Security:** No hardcoded credentials; follow security standards; input validation.\n"
            "- **Dev & Prod:** Don't disrupt working features; targeted changes; prioritize security, maintainability, performance.\n"
            "- **Performance:** Identify bottlenecks before optimizing; balance optimization.\n"
            "- **Code & Docs:** Syntax highlighting; comments; reference docs; use Cosmos Pattern for modules.\n" # Added Cosmos Pattern here as well
            "- **Style & Tone:** Concise, clear, straightforward.\n"
            "- **Design Philosophy:** Interface-based programming; dependency inversion; composition over inheritance.\n"
            "- **Cross-Platform:** Awareness of cross-platform issues.\n"
            "- **UI/UX:** Modern, accessible UI; consider accessibility.\n"
            "- **API Design:** RESTful principles; GraphQL schemas; versioning; security.\n"
            "- **Data Privacy:** Data minimization; privacy by design; DSARs; encryption.\n"
            "- **Clean Architecture (Cosmos Pattern):** Separate interfaces/implementations; factories; public APIs only.\n" # Explicit Cosmos Pattern mention
            "- **Pattern Continuity:** Maintain existing patterns and architecture.\n"
            "- **Verification:** Start app to verify each phase and step.\n" # Added verification step

        )


        # Generate the project overview section
        overview_prompt = (
            f"Generate a high-level project overview for {project_info['name']}.\n\n"
            f"Project Details:\n"
            f"Name: {project_info['name']}\n"
            f"Goal: {project_info['goal']}\n"
            f"**Design**: Sleek, modern, minimalistic design.\n"
            f"\n\n"
        )

        # Conditionally add technology suggestion if no technology is specified
        if not project_info.get('tech_specified', False):
            overview_prompt += (
                f"Since no specific technology stack was explicitly specified, this PRD will focus on "
                f"architecture, design principles, and phased implementation, making it technology-agnostic. "
                f"The generated PRD will emphasize clean architecture, separation of concerns, and testability. "
                f"Implementation phases will be described conceptually, without specific technology commands.\n\n"
                f"If you intend to use a specific technology stack, ensure it aligns with the architectural guidelines "
                f"defined in this PRD. For web-based projects like Chrome extensions, consider HTML, CSS, JavaScript, "
                f"and frameworks like React, Vue, or Svelte. For cross-platform apps, options include Flutter, React Native, or native technologies. "
                f"Choose a stack that best fits your needs while adhering to the outlined architectural principles.\n\n"
            )

        overview_prompt += (
            f"{base_guidelines}\n"
            f"{ai_guidelines_ref}\n\n"
            "**First, provide a concise 1-2 sentence description of the '{project_info['name']}' app/extension.**\n\n"
            "**Then, suggest a phased implementation approach, with the following structure:**\n"
            "#### Phase 1: Minimal Working Prototype\n"
            "Start with a simple, working prototype that builds and runs. Focus on setting up the project with minimal functionality and verifying the setup by starting the app.\n\n" # Added verification in Phase 1
            "Then suggest 3-5 additional phases, each building on a working foundation from the previous phase. Use '#### Phase N: Phase Name' format for phase headings.\n\n"
            "After defining the phases, then Include:\n"
            "1. Project Overview\n"
            "   - Core functionality\n"
            "   - Target architecture (Clean Architecture, Cosmos Pattern for modules)\n" # Added Cosmos Pattern to architecture
            "   - Key components\n"
            "   - System boundaries\n\n"
            "2. Technology Recommendations (General, if no specific tech is provided)\n"
            "   - Suggest general technology categories (Frontend, Backend, Database, etc.)\n" # More general tech recommendations
            "   - Framework versions (if applicable)\n"
            "   - Key libraries (if applicable)\n"
            "   - Development tools\n"
            "   - Build requirements\n\n"
            "3. High-Level Design\n"
            "   - **Overall aesthetic**: Sleek, modern, minimalistic design\n"
            "   - Component diagram (use Mermaid format)\n"
            "   - Data flow\n"
            "   - Key interfaces\n"
            "   - Security considerations\n\n"
            "For the component diagram, use Mermaid format like this:\n"
            "```mermaid\n"
            "flowchart TB\n"
            "    subgraph Presentation Layer\n"
            "        UI[UI Components] --- VM[View Models]\n"
            "    end\n"
            "    subgraph Application Layer\n"
            "        VM --- UC[Use Cases]\n"
            "    end\n"
            "    subgraph Domain Layer\n"
            "        UC --- E[Entities]\n"
            "        UC --- R[Repository Interfaces]\n"
            "    end\n"
            "    subgraph Infrastructure Layer\n"
            "        RI[Repository Implementations] --- R\n"
            "        RI --> DS[Data Sources]\n"
            "    end\n"
            "    Presentation Layer --> Application Layer\n"
            "    Application Layer --> Domain Layer\n"
            "    Domain Layer --> Infrastructure Layer\n"
            "```\n\n"
            "IMPORTANT FORMATTING RULES:\n"
            "- No code blocks (```) for bullet points\n"
            "- Code blocks ONLY for code, diagrams, or multi-line technical content\n"
            "- Consistent bullet point formatting\n"
            "- Bold or italic for components/models, not code blocks\n\n"
            "IMPORTANT NOTE FOR IDE AI IMPLEMENTATION:\n"
            "- Overview document for UNDERSTANDING ONLY\n"
            "- NO implementation yet\n"
            "- NO project structure yet\n"
            "- NO repository initialization yet\n"
            "- Wait for Phase 1 document for implementation\n"
            "- Each phase has specific instructions\n"
            "- Confirm each phase completion before next phase\n"
            "Format in Markdown. Focus on architecture and design, not implementation details."
        )
        self.spinner.start("Generating Project Overview...")
        overview_section = self._generate_section(overview_prompt, "overview")
        sections.append(overview_section)

        # Extract phase descriptions from the generated overview
        phase_descriptions = self._extract_phases_from_overview(overview_section)

        # Generate each phase section based on extracted dynamic phases
        for i, phase_info in enumerate(phase_descriptions, 1):
            phase_name = phase_info['name']
            phase_description = phase_info['description']
            self.spinner.update_message(f"Generating Phase {i}: {phase_name}...")
            phase_prompt = (
                f"Create Phase {i} implementation plan for {project_info['name']}.\n\n"
                f"Project Context:\n{project_info['goal']}\n"
                f"Phase: {phase_name}\n"
                f"Phase Description: {phase_description}\n\n"
                f"**Design Requirement**: Sleek, modern, minimalistic UI. Clear layout, effective typography, consistent visual style.\n"
                f"Follow Clean Architecture principles and Cosmos Pattern for module structure.\n" # Emphasize architecture and Cosmos Pattern
            )

            # Conditionally add technology-specific guidance if tech was specified
            if project_info.get('tech_specified', False):
                if 'flutter' in project_info['goal'].lower():
                    phase_prompt += (
                        f"**Technology Focus**: Flutter development. Use Flutter widgets, follow best practices.\n"
                        f"**State Management**: BLoC or Provider based on complexity.\n"
                    )
                elif 'chrome extension' in project_info['goal'].lower() or 'next.js' in project_info['goal'].lower():
                    phase_prompt += (
                        f"**Technology Focus**: Chrome Extension development, potentially using web technologies (HTML, CSS, JavaScript, React/Next.js).\n"
                        f"**Component Framework**: Consider React or Vue for UI modularity.\n"
                    )
                # Add more conditional blocks for other technologies if needed

            phase_prompt += (
                f"{base_guidelines}\n"
                f"{ai_guidelines_ref}\n\n"
                f"PHASE {i} REQUIREMENTS:\n"
                f"- Phase {i-1} complete and verified (if not Phase 1)\n"
                f"- All previous tests passing\n"
                f"- Follow implementation steps in order, one baby step at a time\n" # Baby step approach
                f"- After each implementation step, start the app and verify the implementation.\n" # Verification after each step
                f"- Confirm completion of each step\n"
                f"- Maintain consistent architecture (Clean Architecture, Cosmos Pattern)\n" # Cosmos Pattern in phases too
                f"- Get IDE AI verification before proceeding\n\n"
                "Provide:\n"
                "1. Implementation Steps (Baby Steps):\n" # Baby Steps in heading
                "   - Concrete, verifiable tasks\n" # Verifiable tasks
                "   - Component specifications (using Cosmos Pattern)\n" # Cosmos Pattern for components
                "   - Code structure (Cosmos Pattern)\n" # Cosmos Pattern for code structure
                "   - Interface definitions\n\n"
                "2. Completion Criteria:\n"
                "   - Functionality to be implemented and verified by starting the app\n" # Verification by starting app
                "   - Test requirements\n"
                "   - Performance metrics (if relevant)\n"
                "   - Quality checks\n\n"
                "3. Sample Code (Illustrative):\n"
                "   - Core interfaces\n"
                "   - Key implementations\n"
                "   - Test examples\n"
                "   - Configuration examples (avoid hardcoding, use placeholders)\n\n" # Emphasize placeholders
                "4. Validation Steps (Verification Checklist):\n" # Verification Checklist
                "   - Steps to start the app and verify functionality after each implementation step\n" # Verification steps detailed
                "   - Test scenarios\n"                "   - Success criteria for each step and phase\n" # Success criteria for steps and phases
                "   - Error case examples\n"
                "   - Integration points to verify\n\n"
                f"For IDE AI Usage:\n"
                f"- Copy ENTIRE phase document content to IDE AI chat.\n"
                f"- Paste directly into IDE AI chat interface.\n"
                f"- For each implementation step, ask IDE AI for assistance (e.g., 'Help me implement step 1...').\n"
                f"- AI MUST follow architectural guidelines in `project_prompt.md` and Cosmos Pattern.\n" # Cosmos Pattern for AI
                f"- Prioritize architectural compliance and code quality.\n"
                f"- Verify AI suggestions against project architecture and Cosmos Pattern.\n" # Cosmos Pattern for AI verification
                f"- After each step, start the app and verify implementation.\n" # Verification after each step for AI
                f"- Confirm step completion and get AI verification before next step.\n"
                f"- Project Context: [From project_prompt.md]\n"
                f"- Current Phase Goal: [From phase description]\n"
                f"- IMPORTANT: Code MUST follow `project_prompt.md` architecture and Cosmos Pattern.\n" # Cosmos Pattern important note
                f"- **Design Focus**: Sleek, modern, minimalistic UI design.\n"
                f"- Ensure architectural integrity, encapsulation, consistent tech stack, and modern UI design.\n"
                f"Format in Markdown. Concrete, testable implementation steps. No implementation without explicit confirmation."
            )
            phase_section = self._generate_section(phase_prompt, f"phase_{i}")
            sections.append(phase_section)

        self.sections = sections
        return sections


    def _extract_phases_from_overview(self, overview_section: str) -> List[Dict[str, str]]:
        """Extract phase names and descriptions from the overview section."""
        phase_descriptions: List[Dict[str, str]] = []
        phase_pattern = re.compile(r"#### Phase \d+:\s*(.*)") # Matches "#### Phase N: Phase Name"

        lines = overview_section.splitlines()
        i = 0
        current_phase_info = None
        while i < len(lines):
            line = lines[i].strip()
            phase_match = phase_pattern.match(line)
            if phase_match:
                if current_phase_info: # Save previous phase if exists
                    phase_descriptions.append(current_phase_info)

                phase_name = phase_match.group(1).strip()
                current_phase_info = {'name': phase_name, 'description': ""} # Start new phase info
                i += 1
                description_lines = []
                while i < len(lines) and not lines[i].strip().startswith("#### Phase "): # Read description until next phase heading
                    description_lines.append(lines[i].strip())
                    i += 1
                current_phase_info['description'] = " ".join(description_lines).strip() # Join description lines and set
                continue # Skip incrementing i again as inner loop already did
            i += 1

        if current_phase_info: # Add the last phase if it was captured
            phase_descriptions.append(current_phase_info)


        if not phase_descriptions:
            logging.warning("Could not automatically extract phase descriptions from overview. Check overview format.")
            return []

        return phase_descriptions


    def save_prd(self, sections: List[str], project_name: str):
        """Save PRD sections into Markdown files."""
        if not sections:
            logging.error("No PRD content to save.")
            return

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
        try:
            dir_name.mkdir(exist_ok=True, parents=True)

            # Create phase_00.md with initial project overview and basic instructions
            instructions_file = dir_name / "phase_00.md"
            with instructions_file.open('w', encoding='utf-8') as f:
                f.write(f"# {project_name} - IDE AI Instructions\n\n")
                f.write("## Project Definition\n\n")
                f.write(sections[0])
                f.write("\n\n## AI Development Assistant Guidelines\n\n")

                # Include the full AI Assistant Guidelines in phase_00.md
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
                f.write("- Suggest appropriate error handling that doesn't expose sensitive information.\n\n")

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

                f.write("### 13. Clean Architecture Module Structure (Cosmos Pattern)\n")
                f.write("Follow the Cosmos Pattern for module structure to maintain Clean Architecture principles. This involves separating interfaces from implementations, using factory patterns, and ensuring modules are independently testable. Refer to `project_prompt.md` for language-specific directory structure examples.\n\n")

                f.write("### 14. Pattern Reference\n")
                f.write("The 'Cosmos Pattern' refers to the Clean Architecture Module Structure defined in section 13. When instructed to use this pattern, apply the appropriate language-specific directory structure and follow all principles outlined in that section.\n\n")

                f.write("### 15. Pattern and Architecture Continuity\n")
                f.write("- Maintain all existing design patterns and software architecture (including the Cosmos Pattern) when making any changes or updates.\n")
                f.write("- Never deviate from established architectural patterns even for minor modifications.\n")
                f.write("- Ensure all new code follows the same architectural principles as the existing codebase.\n")
                f.write("- When extending functionality, preserve the interface-implementation separation and module structure.\n")
                f.write("- Validate that changes preserve the integrity of the overall system architecture.\n")
                f.write("- If architectural changes are necessary, explicitly discuss the rationale before implementation.\n\n")


                f.write("### Development Workflow\n")
                f.write("- Implement features iteratively according to the phase plan, one baby step at a time.\n") # Baby steps workflow
                f.write("- After each implementation step, start the app and verify the implementation.\n") # Verification workflow
                f.write("- Write tests before or alongside implementation.\n")
                f.write("- Document all key decisions and architectural choices.\n")
                f.write("- Review code quality and test coverage before completing each phase.\n")
                f.write("- Maintain consistency with established patterns (Cosmos Pattern).\n\n") # Cosmos Pattern in workflow

                f.write("## IDE AI Instructions\n\n")
                f.write("As the AI assistant for this project:\n")
                f.write("1. Follow one phase at a time as provided by the developer, step by baby step.\n") # Baby steps for AI
                f.write("2. After each implementation step, provide instructions to start the app and verify the implemented functionality.\n") # Verification for AI
                f.write("3. Strictly adhere to the architectural guidelines defined in project_prompt.md and Cosmos Pattern.\n") # Cosmos Pattern for AI
                f.write("4. Maintain proper encapsulation between modules as defined in the architecture (Cosmos Pattern).\n") # Cosmos Pattern for AI
                f.write("5. Use consistent technology stacks throughout the development process.\n")
                f.write("6. Only implement features defined in the current phase.\n")
                f.write("7. Suggest refactoring when code deviates from the defined architecture or Cosmos Pattern.\n") # Cosmos Pattern for AI refactoring
                f.write("8. Maintain consistency with previously established patterns and conventions.\n")
                f.write("9. Provide complete implementations rather than skeleton code.\n")
                f.write("10. Include appropriate tests for all new functionality.\n")
                f.write("11. Validate that each phase is complete before moving to the next, including starting the app and verifying.\n\n") # Verification for AI phase completion

                f.write("## DO NOT Begin Implementation Yet\n")
                f.write("Wait for Phase 1 document before starting implementation.\n")
                f.write("Refer to project_prompt.md for detailed technical architecture and Cosmos Pattern.\n") # Cosmos Pattern in DO NOT

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
                if description:
                    f.write(f"- {description}\n")
                if not description:
                    f.write("- [Primary purpose of the application/extension in 1-2 sentences]\n")
                f.write("- [Key user benefits/value proposition]\n")
                f.write("- [Primary target audience]\n")
                f.write("- [Success criteria for the project]\n\n")

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
                    # Generic tech stack
                    f.write("- **Frontend**: [Frontend framework/language (e.g., React, Vue, Svelte, HTML/JS)]\n")
                    f.write("- **Architecture Pattern**: Clean Architecture, Cosmos Pattern for modules\n") # Cosmos Pattern in tech stack
                    f.write("- **State Management**: [State management approach (e.g., Context API, Redux, Zustand)]\n")
                    f.write("- **Styling**: [UI design system/approach (e.g., CSS frameworks, styled components)]\n")
                    f.write("- **Data Persistence**: [Local storage solution (e.g., browser local storage, IndexedDB)]\n")
                    f.write("- **Networking**: [API communication approach (e.g., Fetch API, Axios)]\n")

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
                f.write("- Implement one phase at a time, step by baby step.\n") # Baby steps guideline
                f.write("- Verify each phase is complete (start app and check) before next phase.\n") # Verification guideline
                f.write("- Maintain architectural integrity (Clean Architecture, Cosmos Pattern) throughout.\n") # Cosmos Pattern guideline
                f.write("- Test each component as implemented.\n")
                f.write("- Document key design decisions.\n")

                # Folder Structure - Cosmos Pattern Directory Structure
                f.write("\n## Cosmos Pattern - Recommended Folder Structure\n\n") # Cosmos Pattern Folder Structure heading
                # Generic Cosmos Pattern folder structure, adaptable
                f.write("```\n")
                f.write(f"{project_name.lower().replace(' ', '_')}/\n")
                f.write("├── src/                     # Source code root\n")
                f.write("│   ├── domain/              # Domain layer - business logic and entities\n")
                f.write("│   │   └── <module_name>/   # Modules following Cosmos Pattern\n") # Cosmos Pattern Modules
                f.write("│   │       ├── public/\n")
                f.write("│   │       ├── service/\n")
                f.write("│   │       ├── factory/\n")
                f.write("│   │       └── <module>_controller.rs (or language equivalent)\n")
                f.write("│   ├── application/         # Application layer - use cases/services\n")
                f.write("│   ├── infrastructure/      # Infrastructure layer - external dependencies, data sources\n")
                f.write("│   └── presentation/        # Presentation layer - UI components, views\n")
                f.write("├── assets/                  # Static assets\n")
                f.write("├── tests/                   # Tests\n")
                f.write("└── docs/                    # Documentation\n")
                f.write("```\n\n")
                f.write("Refer to `phase_00.md` for language-specific examples of Cosmos Pattern directory structure.\n") # Reference to phase_00.md

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
                        "Refer to 'Tech Stack Summary' in `project_prompt.md` for technology-specific setup.\n\n"
                        "**Verification**: After setup, start the app to ensure basic project setup is correct and the app runs without errors.\n\n" # Verification in Phase 1 init
                    )

                    if "## 1. Implementation Steps" in phase:
                        parts = phase.split("## 1. Implementation Steps")
                        if len(parts) > 1:
                            phase = parts[0] + "## 1. Implementation Steps\n\n" + project_init_note + parts[1]

                # Replace software_architecture.md with project_prompt.md
                phase = phase.replace("software_architecture.md", "project_prompt.md")

                with phase_file.open('w', encoding='utf-8') as f:
                    f.write(f"# Phase {i} Implementation Plan\n\n")

                    # Architecture reminder - Cosmos Pattern included
                    f.write("## ⚠️ IMPORTANT: Architectural Compliance (Clean Architecture, Cosmos Pattern)\n") # Cosmos Pattern in phase files
                    f.write("Code MUST follow architecture in `project_prompt.md` and Cosmos Pattern for modules.\n") # Cosmos Pattern reminder
                    f.write("These rules are foundational and cannot be modified.\n")
                    f.write("Review `phase_00.md` for guidance and `project_prompt.md` for architecture and Cosmos Pattern.\n") # Cosmos Pattern in review
                    f.write("AI assistant MUST adhere to these guidelines.\n\n")

                    f.write("## Implementation Requirements\n\n")
                    if i > 1:
                        f.write(f"- Phase {i-1} complete and verified (by starting app)\n") # Verification in requirements
                        f.write("- All previous tests passing\n")
                    f.write("- Follow implementation steps in order, baby step by baby step.\n") # Baby steps in phase files
                    f.write("- After each step, start the app and verify implementation.\n") # Verification step in phase files
                    f.write("- Confirm step completion.\n")
                    f.write("- Maintain consistent architecture and Cosmos Pattern.\n") # Cosmos Pattern in phase requirements
                    f.write("- Get IDE AI verification before proceeding.\n\n")
                    f.write(phase)
                    f.write("\n\n## Completion Checklist\n\n")
                    f.write("- [ ] All implementation steps completed (baby steps)\n") # Baby steps checklist
                    f.write("- [ ] After each step, app started and functionality verified\n") # Verification checklist
                    f.write("- [ ] All tests passing\n")
                    f.write("- [ ] Code reviewed and documented\n")
                    f.write("- [ ] Architectural compliance (Clean Architecture, Cosmos Pattern) verified\n") # Cosmos Pattern in checklist
                    f.write("- [ ] IDE AI verification received\n")

            self.spinner.stop()
            print(f"\nPRD files created in '{dir_name}' directory.")
            
            # Remove duplicate statistics display since we show it in main()
            print("\nIMPORTANT:")
            print("1. Share phase_00.md with IDE AI assistant")
            print("2. Reference project_prompt.md for project overview, architecture, and Cosmos Pattern")
            print("3. Start implementation: copy phase_01.md content to IDE AI chat")
            print("4. Follow phase documents in order for implementation, baby step by baby step, verifying after each step.")
            print("5. Get IDE AI verification before moving to next step or phase.")

        except Exception as e:
            self.spinner.stop()
            logging.error(f"Error saving PRD files: {e}")


def clear_screen():
    """Clear the terminal screen."""
    print("\033[H\033[J", end="")

def print_header():
    """Print header for the tool."""
    clear_screen()
    print("=" * 50)
    print("Dynamic Phased PRD Generator".center(50))
    print("=" * 50)
    print("\nGenerates detailed, implementable PRD with dynamic phases, designed for AI assistance.") # Updated header description

def main():
    """Main function to run the PRD generator."""
    try:
        # Set up command-line argument parsing
        parser = argparse.ArgumentParser(description='Generate a Product Requirements Document (PRD) with dynamic phases')
        parser.add_argument('--clear-cache', action='store_true', help='Clear the cache before running')
        args = parser.parse_args()
        
        # Print header only once at the start
        print_header()
        
        model, api_key, prd_path, cache_path = load_environment()
        generator = PRDGenerator(model, api_key, prd_path, cache_path, args.clear_cache)
        project_info = generator.get_project_info()
        # Store project_info in the generator for later use
        generator.project_info = project_info
        prd_sections = generator.generate_prd_sections(project_info)

        if prd_sections:
            generator.save_prd(prd_sections, project_info['name'])
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
            print("1. Share phase_00.md with IDE AI assistant")
            print("2. Reference project_prompt.md for project overview, architecture, and Cosmos Pattern")
            print("3. Start implementation: copy phase_01.md content to IDE AI chat")
            print("4. Follow phase documents in order for implementation, baby step by baby step, verifying after each step.")
            print("5. Get IDE AI verification before moving to next step or phase.")

    except KeyboardInterrupt:
        print("\n\nPRD generation cancelled.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()