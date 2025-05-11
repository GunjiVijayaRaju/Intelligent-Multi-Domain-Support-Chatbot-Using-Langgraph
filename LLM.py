from utils import *
from dotenv import load_dotenv
import os

# Load variables from .env file into environment
load_dotenv()

# Access the variables
model_name = os.getenv("MODEL_NAME")
groq_api_key = os.getenv("GROQ_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key,model_name=model_name)
