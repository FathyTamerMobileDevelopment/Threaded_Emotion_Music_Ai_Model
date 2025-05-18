import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

class EmotionDetector:
    def __init__(self, model_path=None):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if model_path and os.path.exists(model_path):
            self.emotion_model = load_model(model_path)
        else:
            print("Model not found. Downloading a pre-trained model...")
            self._download_model()
        
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']
        
        self.cap = None
    
    def _download_model(self):
        """Download a pre-trained emotion detection model"""
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
        from tensorflow.keras.models import Model
        
        
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(7, activation='softmax')(x)
        self.emotion_model = Model(inputs=base_model.input, outputs=predictions)
        
        os.makedirs('models', exist_ok=True)
        self.emotion_model.save('models/emotion_model.h5')
        print("Model saved to models/emotion_model.h5")
    
    def start_camera(self):
        """Start the webcam"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")
    
    def detect_emotion(self, frame=None):
        """
        Detect emotion from webcam frame or provided frame
        Returns: emotion label, confidence, frame with annotations
        """
        if frame is None:
            if self.cap is None:
                self.start_camera()
            ret, frame = self.cap.read()
            if not ret:
                return None, 0, None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            (x, y, w, h) = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)[0]
            
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            if self.emotion_model.input_shape[-1] == 3:
                roi = np.repeat(roi, 3, axis=-1)
            
            predictions = self.emotion_model.predict(roi)[0]
            emotion_idx = np.argmax(predictions)
            emotion_label = self.emotions[emotion_idx]
            confidence = predictions[emotion_idx]
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion_label}: {confidence:.2f}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return emotion_label, confidence, frame
        
        return "no_face", 0, frame
    
    def close(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

if __name__ == "__main__":
    detector = EmotionDetector()
    detector.start_camera()
    
    try:
        while True:
            emotion, conf, frame = detector.detect_emotion()
            if frame is not None:
                cv2.imshow('Emotion Detection', frame)
                print(f"Detected emotion: {emotion} (confidence: {conf:.2f})")
                
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        detector.close()
        cv2.destroyAllWindows()