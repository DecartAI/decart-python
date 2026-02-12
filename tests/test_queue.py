"""
Tests for the queue API.
Note: queue API only supports video models.
Image models must use the process API.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from decart import DecartClient, models, DecartSDKError


@pytest.mark.asyncio
async def test_queue_submit_text_to_video() -> None:
    """Test text-to-video submission with queue API."""
    client = DecartClient(api_key="test-key")

    with patch("decart.queue.client.submit_job") as mock_submit:
        mock_submit.return_value = MagicMock(job_id="job-123", status="pending")

        job = await client.queue.submit(
            {
                "model": models.video("lucy-pro-t2v"),
                "prompt": "A cat walking in a park",
                "seed": 42,
            }
        )

        assert job.job_id == "job-123"
        assert job.status == "pending"
        mock_submit.assert_called_once()


@pytest.mark.asyncio
async def test_queue_submit_video_to_video() -> None:
    """Test video-to-video submission with queue API."""
    client = DecartClient(api_key="test-key")

    with patch("decart.queue.client.submit_job") as mock_submit:
        mock_submit.return_value = MagicMock(job_id="job-456", status="pending")

        job = await client.queue.submit(
            {
                "model": models.video("lucy-pro-v2v"),
                "prompt": "Anime style",
                "data": b"fake video data",
                "enhance_prompt": True,
            }
        )

        assert job.job_id == "job-456"
        assert job.status == "pending"


@pytest.mark.asyncio
async def test_queue_rejects_image_models() -> None:
    """Test that queue API rejects image models with helpful error message."""
    client = DecartClient(api_key="test-key")

    with pytest.raises(DecartSDKError) as exc_info:
        await client.queue.submit(
            {
                "model": models.image("lucy-pro-t2i"),
                "prompt": "A beautiful sunset",
            }
        )

    assert "not supported by queue" in str(exc_info.value)
    assert "process" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_queue_missing_model() -> None:
    """Test that missing model raises an error."""
    client = DecartClient(api_key="test-key")

    with pytest.raises(DecartSDKError):
        await client.queue.submit(
            {
                "prompt": "A cat walking",
            }
        )


@pytest.mark.asyncio
async def test_queue_status() -> None:
    """Test getting job status."""
    client = DecartClient(api_key="test-key")

    with patch("decart.queue.client.get_job_status") as mock_status:
        mock_status.return_value = MagicMock(job_id="job-123", status="processing")

        status = await client.queue.status("job-123")

        assert status.job_id == "job-123"
        assert status.status == "processing"
        mock_status.assert_called_once()


@pytest.mark.asyncio
async def test_queue_result() -> None:
    """Test getting job result."""
    client = DecartClient(api_key="test-key")

    with patch("decart.queue.client.get_job_content") as mock_content:
        mock_content.return_value = b"fake video content"

        result = await client.queue.result("job-123")

        assert result == b"fake video content"
        mock_content.assert_called_once()


@pytest.mark.asyncio
async def test_queue_submit_and_poll_completed() -> None:
    """Test submit_and_poll returns completed result."""
    client = DecartClient(api_key="test-key")

    with (
        patch("decart.queue.client.submit_job") as mock_submit,
        patch("decart.queue.client.get_job_status") as mock_status,
        patch("decart.queue.client.get_job_content") as mock_content,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_submit.return_value = MagicMock(job_id="job-123", status="pending")
        mock_status.return_value = MagicMock(job_id="job-123", status="completed")
        mock_content.return_value = b"fake video data"

        result = await client.queue.submit_and_poll(
            {
                "model": models.video("lucy-pro-t2v"),
                "prompt": "A serene lake",
            }
        )

        assert result.status == "completed"
        assert result.data == b"fake video data"


@pytest.mark.asyncio
async def test_queue_submit_and_poll_failed() -> None:
    """Test submit_and_poll returns failed result."""
    client = DecartClient(api_key="test-key")

    with (
        patch("decart.queue.client.submit_job") as mock_submit,
        patch("decart.queue.client.get_job_status") as mock_status,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_submit.return_value = MagicMock(job_id="job-123", status="pending")
        mock_status.return_value = MagicMock(job_id="job-123", status="failed")

        result = await client.queue.submit_and_poll(
            {
                "model": models.video("lucy-pro-t2v"),
                "prompt": "A serene lake",
            }
        )

        assert result.status == "failed"
        assert result.error == "Job failed"


@pytest.mark.asyncio
async def test_queue_submit_and_poll_with_callback() -> None:
    """Test submit_and_poll calls on_status_change callback."""
    client = DecartClient(api_key="test-key")
    status_changes: list[str] = []

    def on_status_change(job):
        status_changes.append(job.status)

    with (
        patch("decart.queue.client.submit_job") as mock_submit,
        patch("decart.queue.client.get_job_status") as mock_status,
        patch("decart.queue.client.get_job_content") as mock_content,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_submit.return_value = MagicMock(job_id="job-123", status="pending")
        mock_status.side_effect = [
            MagicMock(job_id="job-123", status="processing"),
            MagicMock(job_id="job-123", status="completed"),
        ]
        mock_content.return_value = b"fake video data"

        await client.queue.submit_and_poll(
            {
                "model": models.video("lucy-pro-t2v"),
                "prompt": "A serene lake",
                "on_status_change": on_status_change,
            }
        )

        assert "pending" in status_changes
        assert "processing" in status_changes
        assert "completed" in status_changes


@pytest.mark.asyncio
async def test_queue_submit_missing_required_field() -> None:
    """Test that missing required fields raise an error."""
    client = DecartClient(api_key="test-key")

    with pytest.raises(DecartSDKError):
        await client.queue.submit(
            {
                "model": models.video("lucy-pro-v2v"),
                # Missing 'prompt' and 'data' which are required for v2v
            }
        )


@pytest.mark.asyncio
async def test_queue_submit_max_prompt_length() -> None:
    """Test that prompt length validation works."""
    client = DecartClient(api_key="test-key")
    prompt = "a" * 1001

    with pytest.raises(DecartSDKError) as exception:
        await client.queue.submit(
            {
                "model": models.video("lucy-pro-t2v"),
                "prompt": prompt,
            }
        )

    assert "Invalid inputs for lucy-pro-t2v" in str(exception)


@pytest.mark.asyncio
async def test_queue_includes_user_agent_header() -> None:
    """Test that User-Agent header is included in queue requests."""
    client = DecartClient(api_key="test-key")

    with patch("aiohttp.ClientSession") as mock_session_cls:
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json = AsyncMock(return_value={"job_id": "job-123", "status": "pending"})

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_session_cls.return_value = mock_session

        await client.queue.submit(
            {
                "model": models.video("lucy-pro-t2v"),
                "prompt": "Test prompt",
            }
        )

        mock_session.post.assert_called_once()
        call_kwargs = mock_session.post.call_args[1]
        headers = call_kwargs.get("headers", {})

        assert "User-Agent" in headers
        assert headers["User-Agent"].startswith("decart-python-sdk/")


# Tests for lucy-restyle-v2v with reference_image


@pytest.mark.asyncio
async def test_queue_restyle_with_prompt() -> None:
    """Test lucy-restyle-v2v submission with text prompt."""
    client = DecartClient(api_key="test-key")

    with patch("decart.queue.client.submit_job") as mock_submit:
        mock_submit.return_value = MagicMock(job_id="job-789", status="pending")

        job = await client.queue.submit(
            {
                "model": models.video("lucy-restyle-v2v"),
                "prompt": "Make it look like anime",
                "data": b"fake video data",
                "enhance_prompt": True,
            }
        )

        assert job.job_id == "job-789"
        assert job.status == "pending"
        mock_submit.assert_called_once()


@pytest.mark.asyncio
async def test_queue_restyle_with_reference_image() -> None:
    """Test lucy-restyle-v2v submission with reference image."""
    client = DecartClient(api_key="test-key")

    with patch("decart.queue.client.submit_job") as mock_submit:
        mock_submit.return_value = MagicMock(job_id="job-890", status="pending")

        job = await client.queue.submit(
            {
                "model": models.video("lucy-restyle-v2v"),
                "reference_image": b"fake image data",
                "data": b"fake video data",
            }
        )

        assert job.job_id == "job-890"
        assert job.status == "pending"
        mock_submit.assert_called_once()


@pytest.mark.asyncio
async def test_queue_restyle_rejects_both_prompt_and_reference_image() -> None:
    """Test that lucy-restyle-v2v rejects both prompt and reference_image."""
    client = DecartClient(api_key="test-key")

    with pytest.raises(DecartSDKError) as exc_info:
        await client.queue.submit(
            {
                "model": models.video("lucy-restyle-v2v"),
                "prompt": "Make it anime",
                "reference_image": b"fake image data",
                "data": b"fake video data",
            }
        )

    assert "either 'prompt' or 'reference_image'" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_queue_restyle_rejects_neither_prompt_nor_reference_image() -> None:
    """Test that lucy-restyle-v2v rejects when neither prompt nor reference_image provided."""
    client = DecartClient(api_key="test-key")

    with pytest.raises(DecartSDKError) as exc_info:
        await client.queue.submit(
            {
                "model": models.video("lucy-restyle-v2v"),
                "data": b"fake video data",
            }
        )

    assert "either 'prompt' or 'reference_image'" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_queue_restyle_rejects_enhance_prompt_with_reference_image() -> None:
    """Test that enhance_prompt is only valid with text prompt, not reference_image."""
    client = DecartClient(api_key="test-key")

    with pytest.raises(DecartSDKError) as exc_info:
        await client.queue.submit(
            {
                "model": models.video("lucy-restyle-v2v"),
                "reference_image": b"fake image data",
                "data": b"fake video data",
                "enhance_prompt": True,
            }
        )

    assert "enhance_prompt" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_queue_rejects_file_exceeding_20mb() -> None:
    """Test that files exceeding 20MB are rejected before upload."""
    client = DecartClient(api_key="test-key")

    large_data = b"x" * (21 * 1024 * 1024)

    with pytest.raises(DecartSDKError, match="exceeds the maximum allowed size of 20MB"):
        await client.queue.submit(
            {
                "model": models.video("lucy-pro-v2v"),
                "prompt": "test",
                "data": large_data,
            }
        )
