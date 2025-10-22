# Decart Python SDK

A Python SDK for Decart's models.

## Installation

### Using UV

```bash
uv add decart
```

### Using pip

```bash
pip install decart
```

## Documentation

For complete documentation, guides, and examples, visit:
**https://docs.platform.decart.ai/sdks/python**

## Quick Start

### Process Files

```python
import asyncio
import os
from decart import DecartClient, models

async def main():
    async with DecartClient(api_key=os.getenv("DECART_API_KEY")) as client:
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
async with DecartClient(api_key=os.getenv("DECART_API_KEY")) as client:
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
uv run ruff check decart/ tests/ examples/

# Format code
uv run black decart/ tests/ examples/

# Type check
uv run mypy decart/
```

### Common Commands

```bash
# Install dependencies
uv sync --all-extras

# Run tests with coverage
uv run pytest --cov=decart --cov-report=html

# Run examples
uv run python examples/process_video.py
uv run python examples/realtime_synthetic.py

# Update dependencies
uv lock --upgrade
```

### Publishing a New Version

The package is automatically published to PyPI when you create a GitHub release:

```bash
# 1. Update version in pyproject.toml
version = "0.0.4"

# 2. Commit and push
git add pyproject.toml
git commit -m "chore: bump version to 0.0.4"
git push

# 3. Create a GitHub release
gh release create v0.0.4 \
  --title "v0.0.4 - Feature summary" \
  --notes "## What's Changed
- Feature 1
- Bug fix 2"
```

The GitHub Actions workflow will automatically build, test, and publish to PyPI.

## License

MIT
