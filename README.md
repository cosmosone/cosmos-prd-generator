# Dynamic Phased PRD Generator

A Python tool that automatically generates comprehensive, implementation-ready Product Requirement Documents (PRDs) with dynamic phases. Designed specifically for AI-assisted development workflows.

## Features

- **AI-Powered Document Generation**: Uses Claude APIs to create detailed, contextual PRD content
- **Dynamic Phase Extraction**: Automatically identifies implementation phases from project descriptions
- **Multi-File Output**: Creates structured files for each implementation phase
- **Intelligent Caching**: Saves API responses to reduce token usage and improve performance
- **Token Usage Tracking**: Monitors and displays all token usage, both for API calls and cache savings
- **Clean Architecture Module Structure Support**: Generates documents with a clean architecture design
- **Cross-Platform Support**: Works on Windows, macOS, and Linux

## Prerequisites

- Python 3.7+
- An Anthropic API key for Claude

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/cosmos-prd-generator.git
   cd cosmos-prd-generator
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with the following content:
   ```
   CLAUDE_MODEL=claude-3-5-haiku-latest
   ANTHROPIC_API_KEY=your-api-key-here
   # PRD_PATH=C:/Temp/prds  # Optional: Path to store PRD files
   # CACHE_PATH=C:/Temp/cache  # Optional: Path to store API cache
   ```

4. Replace `your-api-key-here` with your actual Anthropic API key.
   
   **Note about Windows paths**: When specifying file paths in the .env file, you can use either forward slashes `/` or backslashes `\` on Windows (e.g., `C:/Temp/prds` or `C:\Temp\prds`), but forward slashes are recommended for best compatibility.

## Usage

Run the generator:
```
python generate-prd.py
```

### Command-Line Options

- `--clear-cache`: Clear the cache before running the generator
  ```
  python generate-prd.py --clear-cache
  ```
  This is useful when you want to force regeneration of all content without using cached responses.

### Interactive Prompts

The script will ask you for:
1. Project name
2. Detailed project goal/description

### Output Files

The generator creates a directory with the following files:

- **project_prompt.md**: Project-wide AI rules and guidelines for the AI to follow throughout the development
- **phase_00.md**: The first file to share with your IDE AI assistant, containing the project overview
- **phase_01.md, phase_02.md, etc.**: Step-by-step implementation plans for each development phase

### Directory Structure

Files are saved to the following location:
```
<PRD_PATH>/[project_name]_prd/
```

Where `<PRD_PATH>` is:
- The path specified in your `.env` file's `PRD_PATH` variable, if set (e.g., C:/Temp/prds)
- Your user home directory + "/prds" if not specified (e.g., C:/Users/username/prds on Windows or /home/username/prds on Linux)

### AI Workflow Process

The generated PRD is designed for a step-by-step implementation with an AI assistant:

1. **Project Configuration**: Use `project_prompt.md` as the project-wide AI rulebook that ensures the AI assistant follows consistent guidelines throughout all phases of development. This file defines:
   - Project architecture requirements
   - Clean Architecture principles
   - Coding standards and patterns
   - Security practices
   - Interface design principles
   - Testing requirements
   - The AI must adhere to these rules for all implementation phases

2. **Initial Context Setting**: Share `phase_00.md` with your IDE AI assistant first to give it an overview of the application you want to develop.

3. **Implementation by Phases**: For each phase:
   - Provide the content of `phase_01.md` to the AI
   - Implement one section at a time, in order
   - Verify each step works by starting the application
   - Get AI verification before proceeding to the next step or phase
   - Once phase 1 is complete and verified, move on to `phase_02.md`, and so on
   - The AI must follow the rules in `project_prompt.md` for all implementations

4. **Step-by-Step Approach**: Each phase document breaks implementation into small, verifiable steps. Take one phase at a time or one step at a time depending on the target app and verify functionality before moving to the next.

### Statistics Output

Upon completion, the generator displays detailed statistics:

```
📊 Generation Statistics
• Output directory: C:\Temp\prds\test_prd
• Cache directory: C:\Users\username\.prd_generator\cache
• Model: claude-3-5-haiku-latest
• API calls: 3
• Cache hits: 4
• Total cache entries: 7
• Tokens used: 15,284 input, 22,876 output
• Total tokens used: 38,160
• Tokens saved: 12,488 input, 18,732 output
• Total tokens saved: 31,220
```

## Features in Detail

### Intelligent Caching

The tool automatically caches responses to save time and reduce token usage. Statistics including tokens used, tokens saved, and API calls made are displayed on completion. The cache is stored in a `.prd_generator/cache` folder in your home directory by default, or in a custom location specified by the `CACHE_PATH` environment variable.

### Token Usage Tracking

The tool tracks and displays all token usage, both for API calls and cache savings. This helps you understand the resource impact and potential cost savings from using the cache.

### Phase Extraction

The tool analyses your project description to identify implementation phases. For best results, format phases in your description using:
```
## Implementation Phases

1. **Foundation Phase**: Description...
2. **Core Functionality Phase**: Description...
```

### Clean Architecture Module Structure Support

Generated documents include the Clean Architecture Module Structure pattern:
```
<module_name>/
├── public/           # Public interfaces defining the contract
├── service/          # Interface and implementation separated
├── factory/          # Factory for creating implementations
└── controller/       # (If needed) Controller connecting to UI
```

## Configuration

You can modify the following settings through environment variables in your `.env` file:

- `CLAUDE_MODEL`: The Claude model to use (default: claude-3-5-haiku-latest)
  - You can configure any available Claude model as needed including claude-3-7-sonnet-latest.
  - Note that costs vary significantly depending on the selected model. More capable models like Opus and Sonnet have higher token costs than Haiku.
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `PRD_PATH`: Custom directory for storing generated PRD files (default: ~/prds)
- `CACHE_PATH`: Custom directory for caching API responses (default: ~/.prd_generator/cache)

## Troubleshooting

### API Key Issues

If you encounter API key errors, check that:
- Your API key is correctly set in the `.env` file
- Your API key has not expired or been revoked

### Path Configuration Issues

If your custom paths are not being recognized:
- Ensure you're using the correct environment variable names: `PRD_PATH` and `CACHE_PATH`
- Verify the paths display correctly at startup when the tool shows:
  ```
  Using PRD output path: C:\Temp\prds
  Using cache directory: C:\Temp\cache
  ```

### Missing Dependencies

If you get a "module not found" error, install the missing dependency:
```
pip install <missing_module>
```

### Path Permissions

If you encounter permission errors when saving files:
- Ensure the user has write permissions to the specified PRD_PATH and CACHE_PATH
- If using the default path in your home directory, ensure your user account has appropriate permissions
- On Linux/macOS, check directory permissions with `ls -la ~/prds`

## Development

### Project Structure

- `generate-prd.py`: Main application script
- `.env`: Environment variables (API keys)
- `cache/`: Cached API responses
- `prds/`: Output directory for generated PRDs

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built using the Anthropic Claude API
- Inspired by Clean Architecture principles for modular software design
