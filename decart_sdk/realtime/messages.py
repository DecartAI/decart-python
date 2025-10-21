from dataclasses import dataclass, asdict
from typing import Literal, Union, Any, Optional
import json


@dataclass
class OfferMessage:
    type: Literal["offer"]
    sdp: str
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerMessage:
    type: Literal["answer"]
    sdp: str
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IceCandidatePayload:
    candidate: str
    sdpMLineIndex: int
    sdpMid: str
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IceCandidateMessage:
    type: Literal["ice-candidate"]
    candidate: IceCandidatePayload
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "candidate": self.candidate.to_dict()
        }


@dataclass
class PromptMessage:
    type: Literal["prompt"]
    prompt: str
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SwitchCameraMessage:
    type: Literal["switch_camera"]
    rotateY: int
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SessionIdMessage:
    type: Literal["session_id"]
    session_id: str
    server_port: int
    server_ip: str


IncomingMessage = Union[OfferMessage, AnswerMessage, IceCandidateMessage, SessionIdMessage]
OutgoingMessage = Union[OfferMessage, AnswerMessage, IceCandidateMessage, PromptMessage, SwitchCameraMessage]


def parse_incoming_message(data: dict[str, Any]) -> Optional[IncomingMessage]:
    msg_type = data.get("type")
    
    if msg_type == "offer":
        return OfferMessage(type="offer", sdp=data["sdp"])
    elif msg_type == "answer":
        return AnswerMessage(type="answer", sdp=data["sdp"])
    elif msg_type == "ice-candidate":
        candidate_data = data["candidate"]
        return IceCandidateMessage(
            type="ice-candidate",
            candidate=IceCandidatePayload(
                candidate=candidate_data["candidate"],
                sdpMLineIndex=candidate_data["sdpMLineIndex"],
                sdpMid=candidate_data["sdpMid"]
            )
        )
    elif msg_type == "session_id":
        return SessionIdMessage(
            type="session_id",
            session_id=data["session_id"],
            server_port=data["server_port"],
            server_ip=data["server_ip"]
        )
    
    return None


def message_to_json(message: OutgoingMessage) -> str:
    return json.dumps(message.to_dict())
