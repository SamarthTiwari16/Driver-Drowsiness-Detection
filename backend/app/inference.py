import cv2
import os
import numpy as np
from tf_keras.models import load_model

class DrowsinessDetector:
    def __init__(self, model_path: str, cascade_dir: str):
        # Load the pre-trained CNN model
        self.model = load_model(model_path)
        
        # Load Haar Cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_frontalface_alt.xml'))
        self.left_eye_cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_lefteye_2splits.xml'))
        self.right_eye_cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_righteye_2splits.xml'))
        
        # Thresholds and labels
        self.labels = ['Closed', 'Open']
        self.score_threshold = 15

    def predict_eye_state(self, eye_roi):
        """
        Preprocesses eye ROI and predicts state using the CNN model.
        Includes Histogram Equalization for better lighting handling.
        Returns: (state_idx, confidence)
        """
        try:
            gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
            
            # Apply Histogram Equalization to help with lighting variations
            equ_eye = cv2.equalizeHist(gray_eye)
            
            resized_eye = cv2.resize(equ_eye, (24, 24))
            normalized_eye = resized_eye / 255.0
            reshaped_eye = normalized_eye.reshape(24, 24, -1)
            expanded_eye = np.expand_dims(reshaped_eye, axis=0)
            
            prediction = self.model.predict(expanded_eye, verbose=0)
            probabilities = prediction[0]
            
            state_idx = np.argmax(probabilities)
            confidence = float(probabilities[state_idx])
            
            # If the model is not confident about the "Closed" state (idx 0), 
            # we default to "Open" (idx 1) to reduce false positives.
            if state_idx == 0 and confidence < 0.7:
                state_idx = 1
                confidence = float(probabilities[1])
                
            return state_idx, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return 1, 0.0 # Default to Open on error

    def detect(self, frame, current_score):
        """
        Processes a single frame and updates the drowsiness score.
        Returns: (is_drowsy, updated_score, debug_info)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect Face
        faces = self.face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        
        l_state, r_state = 1, 1 # Default to Open
        l_conf, r_conf = 0.0, 0.0

        if len(faces) > 0:
            # Take the largest face
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Detect eyes within the face ROI
            # Using specific parameters for eyes inside a face ROI for better precision
            left_eyes = self.left_eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
            right_eyes = self.right_eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

            if len(right_eyes) > 0:
                ex, ey, ew, eh = right_eyes[0]
                r_state, r_conf = self.predict_eye_state(roi_color[ey:ey+eh, ex:ex+ew])

            if len(left_eyes) > 0:
                ex, ey, ew, eh = left_eyes[0]
                l_state, l_conf = self.predict_eye_state(roi_color[ey:ey+eh, ex:ex+ew])
        else:
            # No face detected - optionally handle this state (e.g., stay 'Open' or keep previous score)
            pass

        # Logic: If both eyes are closed, increase score
        if l_state == 0 and r_state == 0:
            updated_score = current_score + 1
            status = "Closed"
        else:
            # Decay faster (2x speed) when eyes are open to prevent "sticking" feeling
            updated_score = max(0, current_score - 2)
            status = "Open"

        is_drowsy = updated_score > self.score_threshold
        
        return {
            "is_drowsy": is_drowsy,
            "score": updated_score,
            "status": status,
            "confidence": (l_conf + r_conf) / 2 if (l_conf or r_conf) else 0.0
        }
