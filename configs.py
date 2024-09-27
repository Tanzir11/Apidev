import os
from dotenv import load_dotenv
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
openai_api_version = os.getenv("openai_api_version")
# Mongo Credential
CONNECTION_STRING = os.getenv("cosmos_db_cluster_url")
NAMESPACE = os.getenv("NAMESPACE")
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")