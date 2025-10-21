# Decart Python SDK

A Python SDK for Decart's models.

## Installation

### Using UV

```bash
uv pip install decart-sdk
```

### Using pip

```bash
pip install decart-sdk
```

## Documentation

For complete documentation, guides, and examples, visit:
**https://docs.platform.decart.ai/sdks/python**

## Quick Start

### Process Files

```python
import asyncio
from decart_sdk import create_decart_client, models
import os

async def main():
    client = create_decart_client(
        api_key=os.getenv("DECART_API_KEY")
    )

    # Generate a video from text
    result = await client.process({
        "model": models.video("lucy-pro-t2v"),
        "prompt": "A cat walking in a lego world",
    })

    # Save the result
    with open("output.mp4", "wb") as f:
        f.write(result)

asyncio.run(main())
```

### Video Transformation

```python
# Transform a video file
with open("input.mp4", "rb") as video_file:
    result = await client.process({
        "model": models.video("lucy-pro-v2v"),
        "prompt": "Anime style with vibrant colors",
        "data": video_file,
        "enhance_prompt": True,
    })

# Save the result
with open("output.mp4", "wb") as f:
    f.write(result)
```

## Development

### Setup with UV

```bash
# Clone the repository
git clone https://github.com/decartai/decart-python
cd decart-python

# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (including dev dependencies)
uv sync --all-extras

# Run tests
uv run pytest

# Run linting
uv run ruff check decart_sdk/

# Format code
uv run black decart_sdk/ tests/ examples/

# Type check
uv run mypy decart_sdk/
```

### Common Commands

```bash
# Install dependencies
uv sync --all-extras

# Run tests with coverage
uv run pytest --cov=decart_sdk --cov-report=html

# Run examples
uv run python examples/process_video.py
uv run python examples/realtime_synthetic.py

# Update dependencies
uv lock --upgrade
```

## License

MIT
