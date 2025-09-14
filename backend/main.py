from fastapi import FastAPI
from pymongo import MongoClient

app = FastAPI()

# Correct URI with encoded password
MONGO_URI = "mongodb+srv://Puri:Puri%40123@cluster0.ozg5y7v.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(MONGO_URI)
db = client["puri02"]

@app.get("/")
def root():
    return {"message": "Backend is running!"}

@app.get("/test-db")
def test_db():
    return {"collections": db.list_collection_names()}
