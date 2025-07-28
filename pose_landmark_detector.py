import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import time

# Model path - update this to your model location
model_path = r'C:\Users\phatl\OneDrive\Documents\Study Material\AI projects\pt_models\pose_landmarker_lite.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variable to store the latest detection results
latest_result = None

def print_results(result, output_image: mp.Image, timestamp_ms: int):
    """Callback function to receive detection results asynchronously"""
    global latest_result
    latest_result = result
    if result.pose_landmarks:
        print(f'Detected {len(result.pose_landmarks)} pose(s) at timestamp {timestamp_ms}ms')
        for i, pose_landmarks in enumerate(result.pose_landmarks):
            print(f'Pose {i+1}: {len(pose_landmarks)} landmarks detected')

def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw pose landmarks and connections on the image"""
    if not detection_result or not detection_result.pose_landmarks:
        return rgb_image
    
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
            for landmark in pose_landmarks
        ])
        
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    
    return annotated_image

def draw_pose_info(image, detection_result):
    """Draw additional pose information on the image"""
    if not detection_result or not detection_result.pose_landmarks:
        return image
    
    annotated_image = np.copy(image)
    height, width, _ = annotated_image.shape
    
    for idx, pose_landmarks in enumerate(detection_result.pose_landmarks):
        # Draw landmark count
        cv2.putText(annotated_image, f"Pose {idx+1}: {len(pose_landmarks)} landmarks", 
                   (10, 30 + idx*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Get key pose points for additional visualization
        if len(pose_landmarks) >= 33:  # Standard pose model has 33 landmarks
            # Head (nose)
            nose = pose_landmarks[0]
            nose_x, nose_y = int(nose.x * width), int(nose.y * height)
            cv2.circle(annotated_image, (nose_x, nose_y), 8, (255, 0, 255), -1)
            
            # Shoulders
            left_shoulder = pose_landmarks[11]
            right_shoulder = pose_landmarks[12]
            left_shoulder_x = int(left_shoulder.x * width)
            left_shoulder_y = int(left_shoulder.y * height)
            right_shoulder_x = int(right_shoulder.x * width)
            right_shoulder_y = int(right_shoulder.y * height)
            
            # Draw shoulder line
            cv2.line(annotated_image, (left_shoulder_x, left_shoulder_y), 
                    (right_shoulder_x, right_shoulder_y), (255, 255, 0), 3)
            
            # Calculate and display shoulder width
            shoulder_width_px = abs(left_shoulder_x - right_shoulder_x)
            cv2.putText(annotated_image, f"Shoulder width: {shoulder_width_px}px", 
                       (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    return annotated_image

def analyze_pose(detection_result):
    """Analyze pose for basic posture information"""
    if not detection_result or not detection_result.pose_landmarks:
        return {}
    
    analysis = {}
    
    for idx, pose_landmarks in enumerate(detection_result.pose_landmarks):
        if len(pose_landmarks) >= 33:
            # Get key landmarks
            nose = pose_landmarks[0]
            left_shoulder = pose_landmarks[11]
            right_shoulder = pose_landmarks[12]
            left_hip = pose_landmarks[23]
            right_hip = pose_landmarks[24]
            
            # Calculate posture metrics
            # Shoulder level (difference in y-coordinates)
            shoulder_level_diff = abs(left_shoulder.y - right_shoulder.y)
            
            # Hip level
            hip_level_diff = abs(left_hip.y - right_hip.y)
            
            # Head tilt (nose position relative to shoulder center)
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            head_tilt = nose.x - shoulder_center_x
            
            analysis[f'pose_{idx}'] = {
                'shoulder_level_diff': shoulder_level_diff,
                'hip_level_diff': hip_level_diff,
                'head_tilt': head_tilt,
                'posture_score': 1.0 - min(1.0, abs(head_tilt) + shoulder_level_diff + hip_level_diff)
            }
    
    return analysis

def main():
    """Main function to run the pose landmark detection"""
    global latest_result
    
    # Configure pose landmarker options
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_poses=2,  # Detect up to 2 poses
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=True,  # Enable segmentation masks
        result_callback=print_results
    )

    print("Initializing Pose Landmarker...")
    try:
        # Create the pose landmarker
        with PoseLandmarker.create_from_options(options) as landmarker:
            print("Pose Landmarker initialized successfully!")
            print("Starting webcam capture...")
            print("Controls:")
            print("- Press 'q' or 'ESC' to quit")
            print("- Press 'h' to toggle help overlay")
            print("- Press 's' to toggle segmentation mask")
            print("- Press 'p' to toggle posture analysis")
            print("- Stand in front of the camera to see pose landmarks")
            
            # Use OpenCV's VideoCapture to start capturing from the webcam
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("Error: Could not open webcam")
                return
            
            # Set camera resolution for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            frame_count = 0
            fps_start_time = time.time()
            show_help = True
            show_segmentation = False
            show_posture_analysis = True
            
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

                # Perform pose landmark detection asynchronously
                landmarker.detect_async(mp_image, timestamp_ms)
                
                # Process results if available
                if latest_result:
                    if show_segmentation and latest_result.segmentation_masks:
                        # Show segmentation mask
                        segmentation_mask = latest_result.segmentation_masks[0].numpy_view()
                        mask_rgb = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2)
                        # Blend with original image
                        alpha = 0.7
                        rgb_frame_float = rgb_frame.astype(np.float32) / 255.0
                        blended = alpha * rgb_frame_float + (1 - alpha) * mask_rgb
                        annotated_frame = (blended * 255).astype(np.uint8)
                    else:
                        # Draw landmarks on the frame
                        annotated_frame = draw_landmarks_on_image(rgb_frame, latest_result)
                    
                    # Add pose information
                    annotated_frame = draw_pose_info(annotated_frame, latest_result)
                    
                    # Convert back to BGR for OpenCV display
                    display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                    
                    # Add posture analysis
                    if show_posture_analysis and latest_result.pose_landmarks:
                        analysis = analyze_pose(latest_result)
                        y_offset = 100
                        for pose_id, metrics in analysis.items():
                            posture_score = metrics['posture_score']
                            color = (0, 255, 0) if posture_score > 0.8 else (0, 255, 255) if posture_score > 0.6 else (0, 0, 255)
                            cv2.putText(display_frame, f"Posture Score: {posture_score:.2f}", 
                                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            y_offset += 25
                else:
                    display_frame = frame
                    # Show "no pose detected" message
                    cv2.putText(display_frame, "No pose detected", (10, 30), 
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
                        "Pose Landmark Detection - MediaPipe",
                        "Controls:",
                        "  'h' - Toggle this help",
                        "  'q'/'ESC' - Quit",
                        "  's' - Toggle segmentation mask",
                        "  'p' - Toggle posture analysis",
                        "",
                        "Pose Landmarks (33 points):",
                        "  Magenta: Head/Face",
                        "  Yellow: Shoulders/Arms", 
                        "  Blue: Torso/Hips",
                        "  Green: Legs/Feet"
                    ]
                    
                    # Draw semi-transparent background for help text
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (10, display_frame.shape[0] - 250), 
                                (400, display_frame.shape[0] - 10), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
                    
                    for i, text in enumerate(help_text):
                        y_pos = display_frame.shape[0] - 240 + i * 18
                        cv2.putText(display_frame, text, (15, y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Add status indicators
                status_y = 60
                if show_segmentation:
                    cv2.putText(display_frame, "Segmentation: ON", (10, status_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    status_y += 20
                
                if show_posture_analysis:
                    cv2.putText(display_frame, "Posture Analysis: ON", (10, status_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # Display the frame
                cv2.imshow('Pose Landmark Detection - MediaPipe', display_frame)
                
                # Check for exit conditions and controls
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # Press Esc or 'q' to exit
                    break
                elif key == ord('h'):  # Toggle help
                    show_help = not show_help
                elif key == ord('s'):  # Toggle segmentation
                    show_segmentation = not show_segmentation
                    print(f"Segmentation mask: {'ON' if show_segmentation else 'OFF'}")
                elif key == ord('p'):  # Toggle posture analysis
                    show_posture_analysis = not show_posture_analysis
                    print(f"Posture analysis: {'ON' if show_posture_analysis else 'OFF'}")

            cap.release()
            cv2.destroyAllWindows()
            print("Program terminated successfully!")

    except Exception as e:
        print(f"Error initializing Pose Landmarker: {e}")
        print(f"Please check that the model file exists and is valid: {model_path}")
        print("\nTroubleshooting:")
        print("1. Ensure MediaPipe is installed: pip install mediapipe")
        print("2. Download the pose landmarker model:")
        print("   wget -O pose_landmarker.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task")
        print("3. Check if the model file path is correct")
        print("4. Verify your webcam is working and not used by another application")

if __name__ == "__main__":
    main()
