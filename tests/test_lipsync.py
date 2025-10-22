import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch
from decart.lipsync import RealtimeLipsyncClient
from decart.lipsync.messages import (
    LipsyncConfigMessage,
    LipsyncConfigAckMessage,
    LipsyncAudioInputMessage,
    LipsyncVideoInputMessage,
    LipsyncInterruptAudioMessage,
    LipsyncSyncedOutputMessage,
    LipsyncErrorMessage,
    LipsyncServerMessageAdapter,
)


class TestLipsyncMessages:
    """Test Lipsync message serialization and deserialization"""

    def test_config_message_serialization(self):
        """Test LipsyncConfigMessage serialization"""
        msg = LipsyncConfigMessage(video_fps=25, audio_sample_rate=16000)
        data = msg.model_dump()
        assert data["type"] == "config"
        assert data["video_fps"] == 25
        assert data["audio_sample_rate"] == 16000

    def test_config_ack_message_deserialization(self):
        """Test LipsyncConfigAckMessage deserialization"""
        data = '{"type": "config_ack"}'
        msg = LipsyncServerMessageAdapter.validate_json(data)
        assert isinstance(msg, LipsyncConfigAckMessage)
        assert msg.type == "config_ack"

    def test_audio_input_message_with_bytes(self):
        """Test LipsyncAudioInputMessage with binary data"""
        audio_data = b"test audio data"
        msg = LipsyncAudioInputMessage(audio_data=audio_data)
        assert msg.type == "audio_input"
        assert msg.audio_data == audio_data

        # Test JSON serialization with base64 encoding
        json_str = msg.model_dump_json()
        assert "audio_input" in json_str

    def test_video_input_message_with_bytes(self):
        """Test LipsyncVideoInputMessage with binary data"""
        video_frame = b"test video frame"
        msg = LipsyncVideoInputMessage(video_frame=video_frame)
        assert msg.type == "video_input"
        assert msg.video_frame == video_frame

    def test_interrupt_audio_message(self):
        """Test LipsyncInterruptAudioMessage"""
        msg = LipsyncInterruptAudioMessage()
        assert msg.type == "interrupt_audio"

    def test_synced_output_message_deserialization(self):
        """Test LipsyncSyncedOutputMessage deserialization"""
        import base64

        video_data = base64.b64encode(b"video").decode()
        audio_data = base64.b64encode(b"audio").decode()
        data = f'{{"type": "synced_result", "video_frame": "{video_data}", "audio_frame": "{audio_data}"}}'

        msg = LipsyncServerMessageAdapter.validate_json(data)
        assert isinstance(msg, LipsyncSyncedOutputMessage)
        assert msg.type == "synced_result"
        assert msg.video_frame == b"video"
        assert msg.audio_frame == b"audio"

    def test_error_message_deserialization(self):
        """Test LipsyncErrorMessage deserialization"""
        data = '{"type": "error", "message": "Test error"}'
        msg = LipsyncServerMessageAdapter.validate_json(data)
        assert isinstance(msg, LipsyncErrorMessage)
        assert msg.type == "error"
        assert msg.message == "Test error"

    def test_message_discriminator(self):
        """Test that message discriminator works correctly"""
        messages = [
            ('{"type": "config_ack"}', LipsyncConfigAckMessage),
            ('{"type": "error", "message": "test"}', LipsyncErrorMessage),
        ]

        for json_data, expected_type in messages:
            msg = LipsyncServerMessageAdapter.validate_json(json_data)
            assert isinstance(msg, expected_type)


class TestRealtimeLipsyncClient:
    """Test RealtimeLipsyncClient functionality"""

    @pytest.fixture
    def client(self):
        """Create a test client"""
        return RealtimeLipsyncClient(
            api_key="test-api-key",
            base_url="https://api.decart.ai",
            audio_sample_rate=16000,
            video_fps=25,
            sync_latency=0.1,
        )

    def test_client_initialization(self, client):
        """Test client initialization"""
        assert client._api_key == "test-api-key"
        assert client._audio_sample_rate == 16000
        assert client._video_fps == 25
        assert client._sync_latency == 0.1
        assert "wss://api.decart.ai" in client._url
        assert "/router/lipsync/ws" in client._url

    def test_url_conversion(self):
        """Test URL conversion from HTTP to WebSocket"""
        # Test HTTPS to WSS
        client1 = RealtimeLipsyncClient("key", "https://api.example.com")
        assert client1._url == "wss://api.example.com/router/lipsync/ws"

        # Test HTTP to WS
        client2 = RealtimeLipsyncClient("key", "http://localhost:8000")
        assert client2._url == "ws://localhost:8000/router/lipsync/ws"

    @pytest.mark.asyncio
    async def test_connect_success(self, client):
        """Test successful connection to lipsync server"""
        mock_websocket = AsyncMock()

        # Mock the configuration acknowledgment
        mock_websocket.recv = AsyncMock(return_value='{"type": "config_ack"}')
        mock_websocket.send = AsyncMock()

        mock_connect = AsyncMock(return_value=mock_websocket)
        with patch("decart.lipsync.client.websockets.connect", mock_connect):
            await client.connect()

            # Verify WebSocket connection was made
            assert client._websocket == mock_websocket

            # Verify configuration was sent
            mock_websocket.send.assert_called_once()
            sent_data = json.loads(mock_websocket.send.call_args[0][0])
            assert sent_data["type"] == "config"
            assert sent_data["video_fps"] == 25
            assert sent_data["audio_sample_rate"] == 16000

            # Verify response handling task was created
            assert client._response_handling_task is not None

            # Clean up
            client._response_handling_task.cancel()
            await client.disconnect()

    @pytest.mark.asyncio
    async def test_connect_config_not_acknowledged(self, client):
        """Test connection failure when config is not acknowledged"""
        mock_websocket = AsyncMock()

        # Mock an error response instead of config_ack
        mock_websocket.recv = AsyncMock(
            return_value='{"type": "error", "message": "Invalid config"}'
        )
        mock_websocket.send = AsyncMock()

        mock_connect = AsyncMock(return_value=mock_websocket)
        with patch("decart.lipsync.client.websockets.connect", mock_connect):
            with pytest.raises(Exception, match="Configuration not acknowledged"):
                await client.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self, client):
        """Test client disconnection"""
        mock_websocket = AsyncMock()
        client._websocket = mock_websocket

        await client.disconnect()

        mock_websocket.close.assert_called_once()
        assert client._websocket is None

    @pytest.mark.asyncio
    async def test_send_audio(self, client):
        """Test sending audio data"""
        mock_websocket = AsyncMock()
        client._websocket = mock_websocket

        audio_data = b"test audio"
        await client.send_audio(audio_data)

        mock_websocket.send.assert_called_once()
        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data["type"] == "audio_input"

    @pytest.mark.asyncio
    async def test_send_video_frame_bytes(self, client):
        """Test sending video frame bytes"""
        mock_websocket = AsyncMock()
        client._websocket = mock_websocket

        video_frame = b"test video"
        await client.send_video_frame_bytes(video_frame)

        mock_websocket.send.assert_called_once()
        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data["type"] == "video_input"

    @pytest.mark.asyncio
    async def test_send_video_frame(self, client):
        """Test sending video frame from numpy array"""
        import numpy as np

        mock_websocket = AsyncMock()
        client._websocket = mock_websocket

        # Create a test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Mock cv2.imencode to return success
        with patch("decart.lipsync.client.cv2.imencode") as mock_encode:
            mock_encode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))

            await client.send_video_frame(test_image)

            # Verify encoding was called
            mock_encode.assert_called_once_with(".jpeg", test_image)

            # Verify message was sent
            mock_websocket.send.assert_called_once()
            sent_data = json.loads(mock_websocket.send.call_args[0][0])
            assert sent_data["type"] == "video_input"

    @pytest.mark.asyncio
    async def test_encode_video_frame_failure(self, client):
        """Test video encoding failure handling"""
        import numpy as np

        test_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Mock cv2.imencode to return failure
        with patch("decart.lipsync.client.cv2.imencode") as mock_encode:
            mock_encode.return_value = (False, None)

            with pytest.raises(Exception, match="Failed to encode video frame as JPEG"):
                await client._encode_video_frame(test_image)

    @pytest.mark.asyncio
    async def test_interrupt_audio(self, client):
        """Test sending audio interrupt"""
        mock_websocket = AsyncMock()
        client._websocket = mock_websocket

        await client.interrupt_audio()

        mock_websocket.send.assert_called_once()
        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data["type"] == "interrupt_audio"

    @pytest.mark.asyncio
    async def test_handle_server_responses_synced_output(self, client):
        """Test handling synced output from server"""
        import base64

        # Prepare test data
        video_data = base64.b64encode(b"video").decode()
        audio_data = base64.b64encode(b"audio").decode()

        mock_websocket = AsyncMock()
        mock_websocket.recv = AsyncMock(
            side_effect=[
                f'{{"type": "synced_result", "video_frame": "{video_data}", "audio_frame": "{audio_data}"}}',
                asyncio.CancelledError(),  # Stop the loop
            ]
        )

        client._websocket = mock_websocket

        # Run the handler briefly
        task = asyncio.create_task(client._handle_server_responses())
        await asyncio.sleep(0.1)
        task.cancel()

        # Check that message was added to queue
        assert not client._out_queue.empty()
        msg = await client._out_queue.get()
        assert isinstance(msg, LipsyncSyncedOutputMessage)
        assert msg.video_frame == b"video"
        assert msg.audio_frame == b"audio"

    @pytest.mark.asyncio
    async def test_handle_server_responses_error(self, client):
        """Test handling error message from server"""
        mock_websocket = AsyncMock()
        mock_websocket.recv = AsyncMock(return_value='{"type": "error", "message": "Server error"}')

        client._websocket = mock_websocket

        # Should raise exception on error message
        with pytest.raises(Exception, match="Server error"):
            await client._handle_server_responses()

    @pytest.mark.asyncio
    async def test_get_synced_output_with_timing(self, client):
        """Test getting synced output with proper timing"""
        import numpy as np

        # Create mock video frame (small valid JPEG)
        video_data = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00\x00?\x00T\xbf\xff\xd9"
        audio_data = b"audio"

        msg = LipsyncSyncedOutputMessage(video_frame=video_data, audio_frame=audio_data)

        # Add message to queue
        await client._out_queue.put(msg)

        # Mock cv2.imdecode to return a simple array
        with patch("decart.lipsync.client.cv2.imdecode") as mock_decode:
            mock_decode.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

            # Get output
            video, audio = await client.get_synced_output()

            # Verify decoding was attempted
            mock_decode.assert_called_once()
            assert audio == audio_data
            assert isinstance(video, np.ndarray)

    @pytest.mark.asyncio
    async def test_full_connection_flow(self):
        """Test complete connection and communication flow"""
        client = RealtimeLipsyncClient(
            api_key="test-key",
            base_url="https://api.test.com",
            audio_sample_rate=16000,
            video_fps=30,
        )

        mock_websocket = AsyncMock()
        responses = [
            '{"type": "config_ack"}',  # Initial config ack
        ]
        mock_websocket.recv = AsyncMock(side_effect=responses)
        mock_websocket.send = AsyncMock()

        mock_connect = AsyncMock(return_value=mock_websocket)
        with patch("decart.lipsync.client.websockets.connect", mock_connect):
            # Connect
            await client.connect()

            # Send some data
            await client.send_audio(b"audio1")
            await client.send_video_frame_bytes(b"video1")
            await client.interrupt_audio()

            # Verify all messages were sent
            assert mock_websocket.send.call_count == 4  # config + 3 messages

            # Disconnect
            client._response_handling_task.cancel()
            await client.disconnect()
            mock_websocket.close.assert_called_once()


class TestLipsyncIntegration:
    """Integration tests for lipsync with the main client"""

    def test_lipsync_client_import(self):
        """Test that LipsyncClient can be imported"""
        from decart.lipsync import RealtimeLipsyncClient

        assert RealtimeLipsyncClient is not None

    def test_lipsync_client_in_package(self):
        """Test that lipsync is properly integrated in the package"""
        import decart.lipsync

        assert hasattr(decart.lipsync, "RealtimeLipsyncClient")
