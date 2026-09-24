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

### Image Editing (Process API)

```python
import asyncio
import os
from decart import DecartClient, models

async def main():
    async with DecartClient(api_key=os.getenv("DECART_API_KEY")) as client:
        # Edit an image
        result = await client.process({
            "model": models.image("lucy-image-2"),
            "prompt": "Apply a painterly oil-on-canvas look while preserving the composition",
            "data": open("input.png", "rb"),
        })

        with open("output.png", "wb") as f:
            f.write(result)

asyncio.run(main())
```

### Video Editing (Queue API)

For video editing jobs, use the queue API to submit jobs and poll for results:

```python
async with DecartClient(api_key=os.getenv("DECART_API_KEY")) as client:
    # Submit and poll automatically
    result = await client.queue.submit_and_poll({
        "model": models.video("lucy-clip"),
        "prompt": "Restyle this footage with anime shading and vibrant neon highlights",
        "data": open("input.mp4", "rb"),
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
        "model": models.video("lucy-clip"),
        "prompt": "Add cinematic teal-and-orange grading and gentle film grain",
        "data": open("input.mp4", "rb"),
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

### Client tokens

Create short-lived client tokens on your backend and hand the signed `token` to your frontend.
Its claims (`service_tier`, allowed models and origins, expiry, ...) are signed into the JWT, so
your backend can verify and read them **offline** instead of round-tripping to the platform.
Verification needs the `verify` extra (`pip install "decart[verify]"`, adds PyJWT + cryptography):

```python
from decart import DecartClient, TokenVerifyError, verify_client_token

async with DecartClient(api_key=os.getenv("DECART_API_KEY")) as client:
    token = await client.tokens.create(expires_in=300, metadata={"service_tier": 0})

    verified = await client.tokens.verify(token.token)  # or: await verify_client_token(token.token)
    verified.service_tier  # 0
    verified.pool          # "free" for tier 0, else "paid"
    verified.user_id, verified.organization_id, verified.api_key_id, verified.expires_at
```

`verify` checks the Ed25519 signature against the platform JWKS (`https://platform.decart.ai/api/auth/jwks`,
fetched once and cached), plus `exp`, `iss` and `aud`. It raises `TokenVerifyError` on a tampered,
expired or foreign token. It is offline JWKS verification, unrelated to the gateway's online
`POST /v1/verify`. To inspect a token *without* verifying it, `client.tokens.decode(token)` /
`decode_client_token(token)` returns the same fields, untrusted. The SDK is async-only; from sync
code use `asyncio.run(verify_client_token(token))`.

### Realtime fast mode

Realtime sessions accept an optional `speed` on `RealtimeConnectOptions`, alongside `resolution`.
Fast mode (`speed="fast"`) serves the session from a higher-compute tier for lower latency and
higher throughput; output quality is unchanged. It is currently available for `lucy-2.5` /
`lucy-latest` and `lucy-vton-3.5` / `lucy-vton-latest`, in the US region only, and is billed at
2x the standard realtime rate for those models. Other models ignore the option (the SDK emits a
warning). Omit it (the default) for standard mode.

```python
from decart import DecartClient, models
from decart.realtime import RealtimeClient, RealtimeConnectOptions

client = DecartClient(api_key=os.getenv("DECART_API_KEY"))
realtime = await RealtimeClient.connect(
    base_url=client.realtime_base_url,
    api_key=client.api_key,
    local_track=local_track,
    options=RealtimeConnectOptions(
        model=models.realtime("lucy-2.5"),
        on_remote_stream=on_remote_stream,
        speed="fast",  # omit for standard mode
    ),
)
```

Each model definition lists the speed tiers it advertises via `ModelDefinition.supported_speeds`
(for example `models.realtime("lucy-2.5").supported_speeds == ("fast",)`). See the
[realtime docs](https://docs.platform.decart.ai/sdks/python) for the full realtime API.

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

### Test UI

The SDK includes an interactive test UI built with Gradio for quickly testing all SDK features without writing code.

```bash
# Install Gradio
pip install gradio

# Run the test UI
python test_ui.py
```

Then open http://localhost:7860 in your browser.

The UI provides tabs for:
- **Image Editing** - Image-to-image edits
- **Video Editing** - Video-to-video edits
- **Video Restyle** - Restyle videos using text prompts or reference images
- **Tokens** - Create short-lived client tokens

Enter your API key at the top of the interface to start testing.

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
