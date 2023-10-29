import logging
import os
# Import the ibm-generative-ai library and local server extension
from genai.extensions.localserver import CustomModel, LocalLLMServer
from genai.extensions.localserver.local_api_server import ApiAuthMiddleware, FastAPI, APIRouter
from genai.model import Model
from genai.schemas import GenerateParams, GenerateResult, TokenizeResult, TokenParams
import torch
from dotenv import load_dotenv
from model import FlanT5Model

# Instansiate the Local Server with your model

load_dotenv()

def set_api_key(app, api_key):
    app.user_middleware.clear()
    app.add_middleware(ApiAuthMiddleware, api_key=api_key, insecure=False)




server = LocalLLMServer(models=[FlanT5Model], port=8000)

api_key = os.getenv("GENAI_KEY")

set_api_key(app=server.app, api_key=api_key)

app = server.app

print(server.get_credentials().api_key)
print(server.get_credentials().api_endpoint)
print(server.get_credentials().DEFAULT_API)
server.start_server()



'''
# Start the server and execute your code
with server.run_locally():
    print(" > Server is started")
    # Get the credentials / connection details for the local server
    creds = server.get_credentials()
    creds.api_endpoint = "http://127.0.0.1:8080"

    # Instantiate parameters for text generation
    params = GenerateParams(decoding_method="sample", max_new_tokens=1000)

    # Instantiate a model proxy object to send your requests
    flan_ul2 = Model(FlanT5Model.model_id, params=params, credentials=creds)

    prompts = ["Hello! How are you?", "How's the weather?"]
    for response in flan_ul2.generate_as_completed(prompts):
        print(f"Prompt: {response.input_text}\nResponse: {response.generated_text}")


print(" > Server stopped, goodbye!")

'''