from pydantic import BaseModel

class ChatBotParam(BaseModel):
    query: str
    session_id: str