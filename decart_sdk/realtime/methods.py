from .webrtc_manager import WebRTCManager
from .messages import PromptMessage, SwitchCameraMessage
from ..errors import create_invalid_input_error


class RealtimeMethods:
    def __init__(self, webrtc_manager: WebRTCManager):
        self._manager = webrtc_manager
    
    async def set_prompt(self, prompt: str, enrich: bool = True) -> None:
        if not prompt or not prompt.strip():
            raise create_invalid_input_error("Prompt cannot be empty")
        
        await self._manager.send_message(PromptMessage(
            type="prompt",
            prompt=prompt
        ))
    
    async def set_mirror(self, enabled: bool) -> None:
        rotate_y = 2 if enabled else 0
        await self._manager.send_message(SwitchCameraMessage(
            type="switch_camera",
            rotateY=rotate_y
        ))
