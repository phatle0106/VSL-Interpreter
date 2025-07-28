import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import time

model_path = r'C:\Users\phatl\OneDrive\Documents\Study Material\AI projects\pt_models\hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variable to store the latest detection results
latest_result = None

def print_results(result, output_image: mp.Image, timestamp_ms: int):
    """Callback function to receive detection results asynchronously"""
    global latest_result
    latest_result = result
    if result.hand_landmarks:
        print(f'Detected {len(result.hand_landmarks)} hand(s) at timestamp {timestamp_ms}ms')
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            handedness = result.handedness[i][0].category_name if result.handedness else "Unknown"
            confidence = result.handedness[i][0].score if result.handedness else 0
            print(f'Hand {i+1}: {handedness} (confidence: {confidence:.2f})')

def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw hand landmarks and connections on the image"""
    if not detection_result or not detection_result.hand_landmarks:
        return rgb_image
    
    annotated_image = np.copy(rgb_image)
    
    # Define connections between landmarks (hand skeleton)
    HAND_CONNECTIONS = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm
        (5, 9), (9, 13), (13, 17)
    ]
    
    for hand_landmarks in detection_result.hand_landmarks:
        # Convert normalized coordinates to pixel coordinates
        height, width, _ = annotated_image.shape
        landmarks_px = []
        
        for landmark in hand_landmarks:
            x_px = int(landmark.x * width)
            y_px = int(landmark.y * height)
            landmarks_px.append((x_px, y_px))
        
        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(landmarks_px) and end_idx < len(landmarks_px):
                start_point = landmarks_px[start_idx]
                end_point = landmarks_px[end_idx]
                cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks
        for i, (x, y) in enumerate(landmarks_px):
            # Different colors for different landmark types
            if i == 0:  # Wrist
                color = (255, 0, 0)  # Red
                radius = 8
            elif i in [4, 8, 12, 16, 20]:  # Fingertips
                color = (0, 0, 255)  # Blue
                radius = 6
            else:  # Other joints
                color = (0, 255, 0)  # Green
                radius = 4
                
            cv2.circle(annotated_image, (x, y), radius, color, -1)
            cv2.putText(annotated_image, str(i), (x-10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return annotated_image

def main():
    """Main function to run the hand landmark detection"""
    global latest_result
    
    # Configure hand landmarker options
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=2,  # Detect up to 2 hands
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=print_results
    )

    print("Initializing Hand Landmarker...")
    try:
        # Create the hand landmarker
        with HandLandmarker.create_from_options(options) as landmarker:
            print("Hand Landmarker initialized successfully!")
            print("Starting webcam capture...")
            print("Controls:")
            print("- Press 'q' or 'ESC' to quit")
            print("- Press 'h' to toggle help overlay")
            print("- Put your hands in front of the camera to see landmarks")
            
            # Use OpenCV's VideoCapture to start capturing from the webcam
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("Error: Could not open webcam")
                return
            
            # Set camera resolution for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            frame_count = 0
            fps_start_time = time.time()
            show_help = True
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                frame_count += 1
                # Calculate timestamp in milliseconds
                timestamp_ms = int(time.time() * 1000)

                # Convert the frame from BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert the frame received from OpenCV to a MediaPipe's Image object
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # Perform hand landmark detection asynchronously
                landmarker.detect_async(mp_image, timestamp_ms)
                
                # Draw landmarks on the frame if available
                if latest_result:
                    annotated_frame = draw_landmarks_on_image(rgb_frame, latest_result)
                    # Convert back to BGR for OpenCV display
                    display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                    
                    # Add text overlay with detection info
                    if latest_result.hand_landmarks:
                        for i, hand_landmarks in enumerate(latest_result.hand_landmarks):
                            handedness = latest_result.handedness[i][0].category_name if latest_result.handedness else "Unknown"
                            confidence = latest_result.handedness[i][0].score if latest_result.handedness else 0
                            text = f"Hand {i+1}: {handedness} ({confidence:.2f})"
                            cv2.putText(display_frame, text, (10, 30 + i*30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Show landmark count
                        total_landmarks = sum(len(hand) for hand in latest_result.hand_landmarks)
                        cv2.putText(display_frame, f"Total landmarks: {total_landmarks}", 
                                   (10, display_frame.shape[0] - 80), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                else:
                    display_frame = frame
                    # Show "no hands detected" message
                    cv2.putText(display_frame, "No hands detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Calculate and display FPS
                if frame_count % 30 == 0:  # Update FPS every 30 frames
                    fps = 30 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                
                if frame_count >= 30:
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                               (display_frame.shape[1] - 120, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Show help overlay
                if show_help:
                    help_text = [
                        "Hand Landmark Detection",
                        "Press 'h' to toggle this help",
                        "Press 'q' or ESC to quit",
                        "Landmark colors:",
                        "  Red: Wrist (0)",
                        "  Blue: Fingertips (4,8,12,16,20)",
                        "  Green: Other joints"
                    ]
                    
                    for i, text in enumerate(help_text):
                        y_pos = display_frame.shape[0] - 160 + i * 20
                        cv2.putText(display_frame, text, (10, y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Display the frame
                cv2.imshow('Hand Landmark Detection - MediaPipe', display_frame)
                
                # Check for exit conditions
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # Press Esc or 'q' to exit
                    break
                elif key == ord('h'):  # Toggle help
                    show_help = not show_help

            cap.release()
            cv2.destroyAllWindows()
            print("Program terminated successfully!")

    except Exception as e:
        print(f"Error initializing Hand Landmarker: {e}")
        print(f"Please check that the model file exists and is valid: {model_path}")
        print("\nTroubleshooting:")
        print("1. Ensure MediaPipe is installed: pip install mediapipe")
        print("2. Check if the model file path is correct")
        print("3. Verify your webcam is working and not used by another application")

if __name__ == "__main__":
    main()
