# Dynamic Phased PRD Generator

A Python tool that automatically generates comprehensive, implementation-ready Product Requirement Documents (PRDs) with dynamic phases. Designed specifically for AI-assisted development workflows.

## Features

- **AI-Powered Document Generation**: Uses Claude 3 to create detailed, contextual PRD content
- **Dynamic Phase Extraction**: Automatically identifies implementation phases from project descriptions
- **Multi-File Output**: Creates structured files for each implementation phase
- **Intelligent Caching**: Saves API responses to reduce token usage and improve performance
- **Checkpoint System**: Preserves progress during generation
- **Clean Architecture Module Structure Support**: Generates documents that follow the Clean Architecture Module Structure
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
   CLAUDE_MODEL=claude-3-7-sonnet-latest
   ANTHROPIC_API_KEY=your-api-key-here
   # PRD_PATH=C:/Temp/prds  # Optional: Path to store PRD files
   # CACHE_DIR=C:/Temp/cache  # Optional: Path to store API cache
   ```

4. Replace `your-api-key-here` with your actual Anthropic API key.
   
   **Note about Windows paths**: When specifying file paths in the .env file, always use forward slashes `/` even on Windows (e.g., `C:/Temp/prds`). The tool will also handle paths with single backslashes (`C:\Temp\prds`) on Windows, but forward slashes are recommended for best compatibility.

## Usage

Run the generator:
```
python generate-prd.py
```

### Interactive Prompts

The script will ask you for:
1. Project name
2. Detailed project goal/description

### Output Files

The generator creates a directory with the following files:

- **instructions.md**: Guidelines for AI-assisted development
- **project_prompt.md**: Project architecture blueprint with Clean Architecture Module Structure details
- **phase_XX.md**: Implementation plans for each development phase

### Directory Structure

Files are saved to the following location:
```
<PRD_PATH>/[project_name]_prd/
```

Where `<PRD_PATH>` is:
- The path specified in your `.env` file's `PRD_PATH` variable, if set (e.g., C:/Temp/prds)
- Your user home directory + "/prds" if not specified (e.g., C:/Users/username/prds on Windows or /home/username/prds on Linux)

### Statistics Output

Upon completion, the generator displays detailed statistics:

```
📊 Generation Statistics
• Output directory: C:\Temp\prds\test_prd
• Cache directory: C:\Temp\cache
• Model: claude-3-7-sonnet-latest
• API calls: 2
• Cache hits: 4
• Total cache entries: 6
• Tokens saved: 1,645 input, 2,466 output
• Total tokens saved: 4,111
```

## Features in Detail

### Intelligent Caching

The tool automatically caches responses to save time and reduce token usage. Statistics including tokens saved and API calls made are displayed on completion. The cache is stored in a `.prd_generator/cache` folder in your home directory by default, or in a custom location specified by the `CACHE_DIR` environment variable (e.g., C:/Temp/cache).

### Phase Extraction

The tool analyzes your project description to identify implementation phases. For best results, format phases in your description using:
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

- `CLAUDE_MODEL`: The Claude model to use (default: claude-3-7-sonnet-latest)
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `PRD_PATH`: Custom directory for storing generated PRD files (default: ~/prds)
- `CACHE_DIR`: Custom directory for caching API responses (default: ~/.prd_generator/cache)

## Troubleshooting

### API Key Issues

If you encounter API key errors, check that:
- Your API key is correctly set in the `.env` file
- Your API key has not expired or been revoked

### Path Configuration Issues

If your custom paths are not being recognized:
- Ensure you're using the correct environment variable names: `PRD_PATH` and `CACHE_DIR`
- Use forward slashes `/` in paths for best compatibility
- For Windows paths, you can use either `C:/Temp/prds` or `C:\Temp\prds` format
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
- Ensure the user has write permissions to the specified PRD_PATH and CACHE_DIR
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