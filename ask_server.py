
import logging
import os

from genai import Credentials
# Import the ibm-generative-ai library and local server extension
from genai.extensions.localserver import CustomModel, LocalLLMServer
from genai.extensions.localserver.local_api_server import ApiAuthMiddleware, FastAPI, APIRouter
from genai.model import Model
from genai.schemas import GenerateParams, GenerateResult, TokenizeResult, TokenParams
import torch
from dotenv import load_dotenv
from model import FlanT5Model





load_dotenv()

# Get the credentials / connection details for the local server

api_key = os.getenv("GENAI_KEY", None)
api_endpoint = os.getenv("GENAI_API", None)
print("api_key")
print(api_key)
creds = Credentials(api_key, api_endpoint)

creds.api_endpoint = "http://127.0.0.1:8000"

# Instantiate parameters for text generation
params = GenerateParams(decoding_method="sample", max_new_tokens=1000)

# Instantiate a model proxy object to send your requests
flan_ul2 = Model(FlanT5Model.model_id, params=params, credentials=creds)

prompts = ["Hello! How are you?", "How's the weather?"]
for response in flan_ul2.generate_as_completed(prompts):
    print(f"Prompt: {response.input_text}\nResponse: {response.generated_text}")