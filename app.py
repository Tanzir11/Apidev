from modules.Scrapper import EnbeddingGenerator
from modules.ChatBot import QuestionAnswer
from fastapi import FastAPI

app = FastAPI()

app.include_router(EnbeddingGenerator)
app.include_router(QuestionAnswer)