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

### Async Processing (Queue API)

For video generation jobs, use the queue API to submit jobs and poll for results:

```python
async with DecartClient(api_key=os.getenv("DECART_API_KEY")) as client:
    # Submit and poll automatically
    result = await client.queue.submit_and_poll({
        "model": models.video("lucy-pro-t2v"),
        "prompt": "A cat playing piano",
        "on_status_change": lambda job: print(f"Status: {job.status}"),
    })

    if result.status == "completed":
        with open("output.mp4", "wb") as f:
            f.write(result.data)
    else:
        print(f"Job failed: {result.error}")
```

Or manage the polling manually:

```python
async with DecartClient(api_key=os.getenv("DECART_API_KEY")) as client:
    # Submit the job
    job = await client.queue.submit({
        "model": models.video("lucy-pro-t2v"),
        "prompt": "A cat playing piano",
    })
    print(f"Job ID: {job.job_id}")

    # Poll for status
    status = await client.queue.status(job.job_id)
    print(f"Status: {status.status}")

    # Get result when completed
    if status.status == "completed":
        data = await client.queue.result(job.job_id)
        with open("output.mp4", "wb") as f:
            f.write(data)
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

The package is automatically published to PyPI when you create a GitHub release.

#### Automated Release

Use the release script to automate the entire process:

```bash
python release.py
```

The script will:

1. Display the current version
2. Prompt for the new version
3. Update `pyproject.toml`
4. Commit and push changes
5. Create a GitHub release with release notes

The GitHub Actions workflow will automatically build, test, and publish to PyPI.

## License

MIT
