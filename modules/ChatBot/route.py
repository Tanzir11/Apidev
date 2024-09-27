from fastapi import APIRouter
from .dto import ChatBotParam
from .services import user_query

QuestionAnswer = APIRouter(prefix="/ChatBot", tags=["LlmEngine"])

@QuestionAnswer.post("/chat_resp")
def Chat_bot(body: ChatBotParam):
    return user_query.chat_response(body.query, body.session_id)