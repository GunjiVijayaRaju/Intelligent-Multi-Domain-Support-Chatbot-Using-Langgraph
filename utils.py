from pydantic import BaseModel, Field
from typing_extensions import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, List
import operator
from langchain_groq import ChatGroq
from langgraph.types import interrupt, Command
from langgraph.constants import Send

from langchain_community.embeddings import FakeEmbeddings
import sqlite3
from datetime import datetime
import faiss
import numpy as np
from LLM import llm
import google.generativeai as genai
from google.generativeai import embedding