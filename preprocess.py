import os
import json
import cv2

# Load metadata
with open("metadata.json", "r") as file:
    metadata = json.load(file)

# Create dataset folders
for label in ["real", "fake"]:
    os.makedirs(f"dataset/{label}", exist_ok=True)

# Function to extract frames
def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    success, frame_id = True, 0
    while success:
        success, frame = cap.read()
        if success:
            frame_path = f"{output_folder}/{os.path.basename(video_path)}_frame{frame_id}.jpg"
            cv2.imwrite(frame_path, frame)
            frame_id += 1
    cap.release()

# Process each video
for video, info in metadata.items():
    label = info["label"]  # "real" or "fake"
    video_path = f"videos/{video}"
    if os.path.exists(video_path):
        extract_frames(video_path, f"dataset/{label}")
        print(f"✅ Processed: {video}")
    else:
        print(f"❌ File not found: {video}")

print("🎯 All frames extracted successfully!")
