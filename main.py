from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "ALPHA Fake News Detection API is live!"}

# Add your API endpoints here
