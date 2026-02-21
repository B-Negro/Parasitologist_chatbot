import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pymongo import MongoClient
from datetime import datetime,timezone
from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
mongo_url = os.getenv("MONGODB_URL")

#set up MongoDB connection
client = MongoClient(mongo_url)
db = client['chatbot']
collection = db['user']



app = FastAPI()

class ChatRequest(BaseModel):
    user_id: str
    question: str   
#to ensure all system components can communicate with the API, we need to set up CORS (Cross-Origin Resource Sharing) middleware. This allows our frontend (which may be served from a different origin) to make requests to our FastAPI backend without running into cross-origin issues.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development only)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Role: Act as a Senior Research Parasitologist and Clinical Diagnostician with 30 years of experience in tropical medicine, veterinary helminthology, and zoonotic disease ecology. Your goal is to provide highly technical, evidence-based, and clinically accurate information.
    
    Knowledge Domains:
    - Medical Parasitology: Human pathogens, life cycles (DHS/IHS), pathogenesis, and immunology.
    - Diagnostics: Morphology, Giemsa staining, PCR, ELISA, and AI-driven microscopy.
    - Pharmacology: Antiparasitic mechanisms (e.g., Benzimidazoles, Praziquantel, Artemisinin) and drug resistance.
    - One Health: Vector-borne diseases (Malaria, Leishmaniasis) and zoonoses.

    Operational Rules:
    1. Terminology: Always use binomial nomenclature (e.g., Schistosoma mansoni) and specify the taxonomic group (Protozoa, Trematode, Cestode, Nematode, or Arthropod).
    2. Structure: When discussing a parasite, follow this format: Taxonomy → Host/Vector → Life Cycle Summary → Clinical Presentation → Diagnostic Gold Standard.
    3. Precision: Distinguish clearly between definitive, intermediate, and paratenic hosts.
    4. Disclaimer: Always include a brief medical disclaimer that your output is for educational/research purposes only.
    """),
    ("placeholder", "{history}"),
    ("user", "{question}")
]) 
#Initializing the "Brain"
llm= ChatGroq(api_key=groq_api_key, model="openai/gpt-oss-20b")
chain = prompt | llm


#userid = "user122"  # Replace with actual user ID

def get_history(user_id):
    #fetch the conversation history for the user from MongoDB
    chats= collection.find({"user_id": user_id}).sort("timestamp", 1)
    history = []

    for chat in chats:
        history.append({"role": chat["role"], "content": chat["message"]}) 
    return history


@app.get("/") # Define a simple route to test the API
def home():
    return {"message": "Welcome to the Parasitology Chatbot API!"}

@app.post("/chat") # Define a route to handle chat interactions
def chat(request: ChatRequest):
        # Get the conversation history for the user
        history = get_history(request.user_id)
    
        # Generate a response using the chain
        response = chain.invoke({"history": history, "question": request.question})
    
        # Save the user's question and the assistant's response to MongoDB
        collection.insert_one({
            "user_id": request.user_id,
            "role": "user",
            "message": request.question,
            "timestamp": datetime.now(timezone.utc)
        })
        collection.insert_one({
            "user_id": request.user_id,
            "role": "assistant",
            "message": response.content,
            "timestamp": datetime.now(timezone.utc)
        })
    
        return {"response": response.content}
