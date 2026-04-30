from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
import jwt
import os
from fastapi.middleware.cors import CORSMiddleware

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

app = FastAPI(title="AppleScan Chat Service")

# Configuration CORS
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# SÉCURITÉ JWT
# ─────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("La variable d'environnement SECRET_KEY est obligatoire.")

def verify_token(authorization: str = os.getenv("AUTHORIZATION", "")):
    # Pour simplifier l'exemple, on récupère le token via une dépendance Header customisée
    # En production, on utiliserait fastapi.security.OAuth2PasswordBearer
    pass # (Version simplifiée ci-dessous pour l'intégration)

from fastapi.security import OAuth2PasswordBearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="http://localhost:8001/login")

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Token invalide")
        return email
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expiré")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Token invalide")

# ─────────────────────────────────────────────
# CONFIGURATION LANGCHAIN & OPENAI
# ─────────────────────────────────────────────
# Initialisation du modèle (nécessite la variable d'environnement OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

# Le prompt système pour donner son rôle à l'IA
prompt = ChatPromptTemplate.from_messages([
    ("system", "Tu es un expert agronome et spécialiste en phytopathologie des pommiers. "
               "Tes réponses doivent être concises, professionnelles et directement actionnables. "
               "Si l'utilisateur demande un traitement, propose des solutions biologiques et chimiques."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

chain = prompt | llm

# Dictionnaire pour stocker l'historique en mémoire (clé = email de l'utilisateur)
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Orchestrateur avec mémoire
agent_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# ─────────────────────────────────────────────
# ROUTES DE L'API
# ─────────────────────────────────────────────
class DiseaseTrigger(BaseModel):
    disease_name: str

class ChatMessage(BaseModel):
    message: str

@app.post("/chat/auto-suggest")
async def auto_suggest(trigger: DiseaseTrigger, user_email: str = Depends(get_current_user)):
    """Déclenché automatiquement par le frontend après la prédiction de la maladie"""
    
    # On vide l'historique précédent si c'est une nouvelle analyse
    if user_email in store:
        store[user_email].clear()

    if "healthy" in trigger.disease_name.lower() or "saine" in trigger.disease_name.lower():
        question = "Le modèle vient d'indiquer que la feuille est saine. Donne un court conseil préventif pour maintenir le pommier en bonne santé."
    else:
        question = f"Le modèle d'IA vient de détecter la maladie '{trigger.disease_name}' sur mon pommier. Que dois-je faire immédiatement pour la traiter ?"

    response = agent_with_memory.invoke(
        {"question": question},
        config={"configurable": {"session_id": user_email}}
    )
    
    return {"reply": response.content}

@app.post("/chat/ask")
async def chat_ask(chat: ChatMessage, user_email: str = Depends(get_current_user)):
    """Route pour les questions libres de l'utilisateur (utilise l'historique)"""
    
    response = agent_with_memory.invoke(
        {"question": chat.message},
        config={"configurable": {"session_id": user_email}}
    )
    
    return {"reply": response.content}