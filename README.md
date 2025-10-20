# Decart Python SDK

A Python SDK for Decart's models.

## Installation

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

## License

MIT
