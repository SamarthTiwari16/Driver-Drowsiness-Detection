from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .websocket import websocket_endpoint

app = FastAPI(title="Driver Drowsiness Detection API")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Drowsiness Detection API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Register WebSocket route
app.add_api_websocket_route("/ws", websocket_endpoint)
