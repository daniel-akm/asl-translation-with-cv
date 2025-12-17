import cv2
import mediapipe as mp
import numpy as np
import os
from glob import glob

class FeatureExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
    
    def extract_landmarks(self, video_frames):
        """
        Extract hand landmarks from video frames
        Returns: array of shape (num_frames, 42) for 1 hand or (num_frames, 84) for 2 hands
        21 landmarks × 2 coordinates (x, y) per hand
        """
        sequence_features = []
        
        for frame in video_frames:
            results = self.hands.process(frame)
            
            frame_features = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract x, y coordinates for all 21 landmarks
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y])
                    frame_features.extend(landmarks)
            
            # If no hands detected, use zeros
            if len(frame_features) == 0:
                frame_features = [0] * 42  # 21 landmarks × 2 coords
            
            sequence_features.append(frame_features)
        
        return np.array(sequence_features)
    
    def process_dataset(self, data_dir='data/raw_videos'):
        """Process all collected videos and extract features"""
        video_files = glob(os.path.join(data_dir, '*.npy'))
        
        X = []  # Features
        y = []  # Labels
        
        print(f"Processing {len(video_files)} videos...")
        
        for video_file in video_files:
            # Get label from filename (e.g., 'hello_0.npy' -> 'hello')
            label = os.path.basename(video_file).split('_')[0]
            
            # Load video frames
            frames = np.load(video_file)
            
            # Extract features
            features = self.extract_landmarks(frames)
            
            X.append(features)
            y.append(label)
        
        # Save processed data
        os.makedirs('data/processed', exist_ok=True)
        np.save('data/processed/X.npy', np.array(X, dtype=object))
        np.save('data/processed/y.npy', np.array(y))
        
        print(f"✓ Processed {len(X)} samples")
        print(f"✓ Feature shape per sample: {X[0].shape}")
        print(f"✓ Saved to data/processed/")
        
        return np.array(X, dtype=object), np.array(y)

# Usage
if __name__ == "__main__":
    extractor = FeatureExtractor()
    X, y = extractor.process_dataset()