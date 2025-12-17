import cv2
import mediapipe as mp
import numpy as np
import os
import time

class SignDataCollector:
    def __init__(self, signs_list, samples_per_sign=30, frames_per_sample=30):
        """
        signs_list: List of sign names (e.g., ['hello', 'thank_you', 'please'])
        samples_per_sign: Number of video samples per sign
        frames_per_sample: Number of frames to record per sample
        """
        self.signs = signs_list
        self.samples_per_sign = samples_per_sign
        self.frames_per_sample = frames_per_sample
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        
        # Create data directory
        os.makedirs('data/raw_videos', exist_ok=True)
    
    def collect_data(self):
        cap = cv2.VideoCapture(0)
        
        for sign in self.signs:
            print(f"\n{'='*50}")
            print(f"Collecting data for sign: {sign.upper()}")
            print(f"{'='*50}")
            
            for sample_num in range(self.samples_per_sign):
                print(f"\nSample {sample_num + 1}/{self.samples_per_sign}")
                print("Press 's' to start recording, 'r' to redo, 'q' to quit")
                
                frames = []
                recording = False
                frame_count = 0
                
                while True:
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(rgb_frame)
                    
                    # Draw landmarks
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                frame, hand_landmarks, 
                                self.mp_hands.HAND_CONNECTIONS)
                    
                    # Display instructions
                    if not recording:
                        cv2.putText(frame, f"Sign: {sign}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, "Press 'S' to start", (10, 70),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "RECORDING", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(frame, f"Frame: {frame_count}/{self.frames_per_sample}",
                                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                        # Record frame
                        frames.append(rgb_frame)
                        frame_count += 1
                        
                        if frame_count >= self.frames_per_sample:
                            # Save the sequence
                            filename = f"data/raw_videos/{sign}_{sample_num}.npy"
                            np.save(filename, np.array(frames))
                            print(f"✓ Saved: {filename}")
                            break
                    
                    cv2.imshow('Data Collection', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('s') and not recording:
                        recording = True
                        frame_count = 0
                        frames = []
                        print("Recording started...")
                    elif key == ord('r'):
                        recording = False
                        frame_count = 0
                        frames = []
                        print("Redo this sample")
                        continue
                    elif key == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    
                    if recording and frame_count >= self.frames_per_sample:
                        break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Data collection complete!")

# Usage
if __name__ == "__main__":
    # Start with 5 simple signs
    my_signs = ['hello', 'thank_you', 'please', 'yes', 'no']
    
    collector = SignDataCollector(
        signs_list=my_signs,
        samples_per_sign=30,  # 30 examples per sign
        frames_per_sample=30   # 30 frames per example (1 second at 30fps)
    )
    
    collector.collect_data()