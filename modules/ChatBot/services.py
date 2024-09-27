from .utils import Qna_query

class UserQuery:
    @staticmethod
    def chat_response(question, session_id):
        return Qna_query(question, session_id)
    
user_query = UserQuery()