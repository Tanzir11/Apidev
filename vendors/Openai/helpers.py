from langchain_openai import AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
import os

from configs import (AZURE_OPENAI_API_KEY, 
                    AZURE_OPENAI_ENDPOINT,
                    openai_api_version)


os.environ["AZURE_OPENAI_ENDPOINT"] = str(AZURE_OPENAI_ENDPOINT)
os.environ["AZURE_OPENAI_API_KEY"] = str(AZURE_OPENAI_API_KEY)

embeddings = AzureOpenAIEmbeddings(azure_deployment="text-embedding-ada-002")
model = AzureChatOpenAI(openai_api_version=openai_api_version,azure_deployment="gpt4o")