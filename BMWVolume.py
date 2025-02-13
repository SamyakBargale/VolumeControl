import cv2
import mediapipe as mp
import numpy as np
import subprocess
from collections import deque
from math import atan2, degrees, dist

class GestureVolumeController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            max_num_hands=1
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Gesture detection parameters
        self.angle_buffer = deque(maxlen=10)
        self.angle_threshold = 15
        self.min_finger_distance = 50  # Minimum distance between fingers to detect gesture
        self.cumulative_rotation = 0
        self.volume_change_threshold = 30  # Reduced rotation needed for change
        self.volume_step = 10  # Increased volume step for faster adjustment

    def get_volume(self):
        output = subprocess.run(
            ["osascript", "-e", "output volume of (get volume settings)"],
            capture_output=True, text=True
        )
        return int(output.stdout.strip())

    def set_volume(self, vol):
        vol = max(0, min(100, vol))
        subprocess.run(["osascript", "-e", f"set volume output volume {vol}"])

    def calculate_finger_angle(self, index_pos, middle_pos):
        dx, dy = middle_pos[0] - index_pos[0], middle_pos[1] - index_pos[1]
        return degrees(atan2(dy, dx))

    def detect_gesture(self, hand_landmarks, img_shape):
        h, w = img_shape[:2]
        
        # Get finger positions
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        
        # Convert to pixel coordinates
        index_pos = (int(index_tip.x * w), int(index_tip.y * h))
        middle_pos = (int(middle_tip.x * w), int(middle_tip.y * h))
        
        # Check if fingers are extended
        if dist(index_pos, middle_pos) < self.min_finger_distance:
            return None
        
        # Calculate angle
        current_angle = self.calculate_finger_angle(index_pos, middle_pos)
        
        # Process rotation
        if len(self.angle_buffer) > 0:
            angle_diff = current_angle - self.angle_buffer[-1]
            
            # Normalize angle difference
            if angle_diff > 180:
                angle_diff -= 360
            elif angle_diff < -180:
                angle_diff += 360
                
            if abs(angle_diff) < self.angle_threshold:
                self.cumulative_rotation += angle_diff
                
                if abs(self.cumulative_rotation) >= self.volume_change_threshold:
                    volume_change = self.volume_step if self.cumulative_rotation > 0 else -self.volume_step
                    current_volume = self.get_volume()
                    self.set_volume(current_volume + volume_change)
                    self.cumulative_rotation = 0  # Reset immediately

        self.angle_buffer.append(current_angle)

    def run(self):
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break
                
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    self.detect_gesture(hand_landmarks, img.shape)
            
            cv2.imshow("Gesture Volume Control", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = GestureVolumeController()
    controller.run()