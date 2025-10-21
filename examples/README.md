# Decart SDK Examples

This directory contains example scripts demonstrating how to use the Decart Python SDK.

## Setup

1. Install the SDK:

```bash
pip install decart
```

2. Set your API key:

```bash
export DECART_API_KEY="your-api-key-here"
```

## Examples

### Process API

- **`process_video.py`** - Generate and transform videos
- **`process_image.py`** - Generate and transform images
- **`process_url.py`** - Transform videos from URLs

### Realtime API

First, install the realtime dependencies:

```bash
pip install decart[realtime]
```

- **`realtime_synthetic.py`** - Test realtime API with synthetic colored frames
- **`realtime_file.py`** - Process a video file in realtime

### Running Examples

```bash
# Generate and transform videos
python examples/process_video.py

# Generate and transform images
python examples/process_image.py

# Transform video from URL
python examples/process_url.py

# Realtime API with synthetic video (saves to output_realtime_synthetic.mp4)
python examples/realtime_synthetic.py

# Realtime API with video file (saves to output_realtime_<filename>.mp4)
python examples/realtime_file.py input.mp4
```

## Next Steps

Check out the [documentation](https://docs.platform.decart.ai/sdks/python) for more examples and detailed guides.
