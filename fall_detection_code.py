import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient


# API Configuration
API_KEY = "1FfK6B9ypa412xqLynYN"
PROJECT_ID = "fall-detection-mbldh-epfrq"
MODEL_VERSION = "2"
MODEL_ID = f"{PROJECT_ID}/2"

# Initialize Roboflow Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

# Input and output image and videos path
INPUT_VIDEO_PATH = "fall_video.mp4"
OUTPUT_VIDEO_PATH = "fall_video_annotatedv9.mp4"

# Confidence threshold (predictions below this value be ignored)
CONFIDENCE_THRESHOLD = 0.2

# ========================== #
# FUNCTION TO RUN INFERENCE

def inter_frame(frame):
    # Convert frame to RGB for inference
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run inference using the new SDK
    result = CLIENT.infer(frame_rgb, model_id=MODEL_ID)
    
    return result


# MAIN PROCESS
def main():
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

    # Prepare video write with same resolution and fps as input video #
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}...")

        predictions = inter_frame(frame)

        if predictions and "predictions" in predictions:
            for pred in predictions["predictions"]:
                confidence = pred["confidence"]
                
                # Skip low-confidence detections
                if confidence < CONFIDENCE_THRESHOLD:
                    continue
                
                # Extract bounding box center coordinates and size 
                x, y = int(pred["x"]), int(pred["y"])
                w, h = int(pred["width"]), int(pred["height"])
                class_name = pred["class"].lower()
                
                # Display labels as detected (no flipping)
                if class_name == "standing":
                    display = "Standing"
                    color = (0, 255, 0)  # green for stand
                elif class_name == "fall detected":
                    display = "Fall Detected"
                    color = (0, 0, 255)  # red for fall
                else:
                    display = class_name
                    color = (255, 255, 0)  # yellow for any unknown class

                # Drawing bounding box around detected object
                cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), color, 5)

                # Draww label text above the bounding box
                label = f"{display} ({confidence:.2f})"
                cv2.putText(frame, label, (x - w//2, y - h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 5)

        # Write the annotated frame to output video
        out.write(frame)

        # Show frame in a window(press 'q' to  stop early)
        cv2.imshow('Fall Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped by user")
            break
    
    # Release resources  
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete. Annotated video saved to:", OUTPUT_VIDEO_PATH)


if __name__ == "__main__":
    main()

            



# from inference_sdk import InferenceHTTPClient

# CLIENT = InferenceHTTPClient(
#     api_url="https://serverless.roboflow.com",
#     api_key="xUAov0eEs2TBXFQTyNpT"
# )

# result = CLIENT.infer(your_image.jpg, model_id="fall-detection-mbldh/1")