import json
import os
from fastapi import WebSocket, WebSocketDisconnect
from .inference import DrowsinessDetector
from .utils import decode_base64_image

# Initialize Detector (Singleton-like behavior for the module)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cnnCat2.h5")
CASCADE_DIR = os.path.join(BASE_DIR, "haar_cascade_files")

detector = DrowsinessDetector(MODEL_PATH, CASCADE_DIR)

async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to WebSocket")
    
    # Internal state for this specific connection's drowsiness score
    session_score = 0
    
    try:
        while True:
            # Receive base64 frame from client
            data = await websocket.receive_text()
            
            # Decode image
            frame = decode_base64_image(data)
            
            if frame is None:
                await websocket.send_json({"error": "Invalid frame data"})
                continue
            
            # Run inference
            result = detector.detect(frame, session_score)
            
            # Update local session score
            session_score = result["score"]
            
            # Return prediction results
            await websocket.send_json({
                "is_drowsy": result["is_drowsy"],
                "score": result["score"],
                "status": result["status"],
                "confidence": round(result["confidence"], 2)
            })
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()
