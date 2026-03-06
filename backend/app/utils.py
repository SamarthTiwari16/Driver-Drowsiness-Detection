import cv2
import numpy as np
import base64

def decode_base64_image(base64_string: str):
    """
    Decodes a base64 string into an OpenCV image (numpy array).
    Expected format: "data:image/jpeg;base64,..."
    """
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        img_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None
