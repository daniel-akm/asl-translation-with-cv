import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

class RealTimePredictor:
    def __init__(self, model_path='models/sign_classifier.h5', 
                 labels_path='models/label_encoder.npy'):
        self.model = load_model(model_path)
        self.labels = np.load(labels_path, allow_pickle=True)
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        
        # Buffer to store recent frames
        self.frame_buffer = deque(maxlen=30)
        self.sequence_length = 30
    
    def extract_landmarks(self, frame):
        """Extract hand landmarks from a single frame"""
        results = self.hands.process(frame)
        
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y])
        
        if len(landmarks) == 0:
            landmarks = [0] * 42
        
        return landmarks
    
    def predict(self):
        """Run real-time prediction"""
        cap = cv2.VideoCapture(0)
        
        print("Real-time Sign Language Recognition")
        print("Press 'q' to quit")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract landmarks and add to buffer
            landmarks = self.extract_landmarks(rgb_frame)
            self.frame_buffer.append(landmarks)
            
            # Draw hand landmarks
            results = self.hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS)
            
            # Make prediction when buffer is full
            if len(self.frame_buffer) == self.sequence_length:
                sequence = np.array([list(self.frame_buffer)])
                prediction = self.model.predict(sequence, verbose=0)
                predicted_class = np.argmax(prediction[0])
                confidence = prediction[0][predicted_class]
                
                label = self.labels[predicted_class]
                
                # Display prediction
                if confidence > 0.7:  # Only show if confident
                    cv2.putText(frame, f"Sign: {label}", (10, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 100),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Collecting frames...", (10, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imshow('Sign Language Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    predictor = RealTimePredictor()
    predictor.predict()