import os
import jwt
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

app = FastAPI(title="AppleScan Chat Service")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("The SECRET_KEY environment variable is required.")

ALGORITHM = os.getenv("ALGORITHM", "HS256")
AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://localhost:8001/login")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=AUTH_SERVICE_URL)

def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])  # type: ignore
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return email
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Tu es un expert agronome et spécialiste en phytopathologie des pommiers. "
               "Tes réponses doivent être concises, professionnelles et directement actionnables. "
               "Si l'utilisateur demande un traitement, propose systématiquement des solutions biologiques ainsi que des solutions chimiques."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

chain = prompt | llm

store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

agent_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

class DiseaseTrigger(BaseModel):
    disease_name: str

class ChatMessage(BaseModel):
    message: str

@app.post("/chat/auto-suggest")
async def auto_suggest(trigger: DiseaseTrigger, user_email: str = Depends(get_current_user)):
    """Automatically triggered by the frontend after a disease prediction."""
    
    if user_email in store:
        store[user_email].clear()

    disease = trigger.disease_name.lower()
    if "healthy" in disease or "saine" in disease:
        question = "Le modèle IA vient d'indiquer que la feuille de pommier est saine. Donne un conseil préventif court pour maintenir l'arbre en bonne santé."
    else:
        question = f"Le modèle IA vient de détecter la maladie '{trigger.disease_name}' sur mon pommier. Que dois-je faire immédiatement pour la traiter ?"
    response = agent_with_memory.invoke(
        {"question": question},
        config={"configurable": {"session_id": user_email}}
    )
    
    return {"reply": response.content}

@app.post("/chat/ask")
async def chat_ask(chat: ChatMessage, user_email: str = Depends(get_current_user)):
    """Route for free text questions from the user (utilizes chat history)."""
    
    response = agent_with_memory.invoke(
        {"question": chat.message},
        config={"configurable": {"session_id": user_email}}
    )
    
    return {"reply": response.content}