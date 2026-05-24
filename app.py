from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load model + vectorizer
model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message":"Toxic Comment API is running 🚀"}

@app.post("/predict")
def predict(data: TextInput):
    text = [data.text]
    vector = vectorizer.transform(text)
    prediction = model.predict(vector)[0]

    return {
        "input_text": data.text,
        "prediction": str(prediction)
    }
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)