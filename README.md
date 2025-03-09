# Dynamic Phased PRD Generator

A Python tool that automatically generates comprehensive, implementation-ready Product Requirement Documents (PRDs) with dynamic phases. Designed specifically for AI-assisted development workflows.

## Features

- **AI-Powered Document Generation**: Uses Claude 3 to create detailed, contextual PRD content
- **Dynamic Phase Extraction**: Automatically identifies implementation phases from project descriptions
- **Multi-File Output**: Creates structured files for each implementation phase
- **Intelligent Caching**: Saves API responses to reduce token usage and improve performance
- **Checkpoint System**: Preserves progress during generation
- **Clean Module Structure Support**: Generates documents that follow the Cosmos Pattern

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
   CLAUDE_MODEL=claude-3-opus-20240229
   ANTHROPIC_API_KEY=your-api-key-here
   ```

4. Replace `your-api-key-here` with your actual Anthropic API key.

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
- **project_prompt.md**: Project architecture blueprint with Clean Module Structure details
- **phase_XX.md**: Implementation plans for each development phase

### Directory Structure

Files are saved to the following location:
```
/prds/[project_name]_prd/
```

## Features in Detail

### Intelligent Caching

The tool automatically caches responses to save time and reduce token usage. Statistics including tokens saved and API calls made are displayed on completion.

### Phase Extraction

The tool analyzes your project description to identify implementation phases. For best results, format phases in your description using:
```
## Implementation Phases

1. **Foundation Phase**: Description...
2. **Core Functionality Phase**: Description...
```

### Clean Module Structure Support

Generated documents include the Clean Module Structure pattern:
```
<module_name>/
├── public/           # Public interfaces defining the contract
├── service/          # Interface and implementation separated
├── factory/          # Factory for creating implementations
└── controller/       # (If needed) Controller connecting to UI
```

## Configuration

You can modify the following settings:

### Environment Variables

- `CLAUDE_MODEL`: The Claude model to use (default: claude-3-opus-20240229)
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `PRD_OUTPUT_PATH`: Custom output directory (default: ./prds)

## Troubleshooting

### API Key Issues

If you encounter API key errors, check that:
- Your API key is correctly set in the `.env` file
- Your API key has not expired or been revoked

### Missing Dependencies

If you get a "module not found" error, install the missing dependency:
```
pip install <missing_module>
```

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
- Inspired by the Cosmos Pattern for clean architecture 