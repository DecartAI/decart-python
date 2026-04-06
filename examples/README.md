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

- **`process_video.py`** - Edit a local video with `lucy-clip`
- **`process_image.py`** - Edit the bundled example image with `lucy-image-2`
- **`process_url.py`** - Transform videos from URLs
- **`queue_image_example.py`** - Turn the bundled example image into motion with `lucy-motion`

### Realtime API

First, install the realtime dependencies:

```bash
pip install decart[realtime]
```

- **`realtime_synthetic.py`** - Test realtime API with synthetic colored frames
- **`realtime_file.py`** - Process a video file in realtime

### Running Examples

`process_image.py` and `queue_image_example.py` use the bundled `examples/files/image.png` asset.
`process_video.py` expects you to place a local video at `examples/assets/example_video.mp4` first.

```bash
# Edit a local video (requires examples/assets/example_video.mp4)
python examples/process_video.py

# Edit the bundled example image
python examples/process_image.py

# Turn the bundled example image into motion
python examples/queue_image_example.py

# Transform video from URL
python examples/process_url.py

# Realtime API with synthetic video (saves to output_realtime_synthetic.mp4)
python examples/realtime_synthetic.py

# Realtime API with video file (saves to output_realtime_<filename>.mp4)
python examples/realtime_file.py input.mp4
```

## Next Steps

Check out the [documentation](https://docs.platform.decart.ai/sdks/python) for more examples and detailed guides.
