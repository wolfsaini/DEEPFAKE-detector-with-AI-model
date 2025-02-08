import tkinter as tk
from tkinter import filedialog, Label, Button
import torch
import cv2
import numpy as np
from model import DeepFakeDetector
from PIL import Image, ImageTk

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepFakeDetector().to(device)
model.load_state_dict(torch.load("deepfake_model.pth", map_location=device))
model.eval()

# Function to preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))  # Resize to match model input
    image = image / 255.0  # Normalize
    image = np.transpose(image, (2, 0, 1))  # Rearrange channels
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
    return image

# Function to analyze video
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames, fake_count, real_count = 0, 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1

        image = cv2.resize(frame, (128, 128)) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)

        output = torch.sigmoid(model(image)).item()
        if output > 0.5:
            fake_count += 1
        else:
            real_count += 1

    cap.release()
    
    fake_percentage = (fake_count / total_frames) * 100 if total_frames > 0 else 0
    real_percentage = (real_count / total_frames) * 100 if total_frames > 0 else 0

    return fake_percentage, real_percentage

# Function to predict deepfake
def detect_deepfake():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
    if not file_path:
        return

    # Show video preview
    cap = cv2.VideoCapture(file_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (300, 180))
        img_tk = ImageTk.PhotoImage(Image.fromarray(frame))
        image_label.config(image=img_tk)
        image_label.image = img_tk

    # Analyze video
    fake_percentage, real_percentage = analyze_video(file_path)
    result_label.config(text=f"Fake: {fake_percentage:.2f}% | Real: {real_percentage:.2f}%", 
                        fg="red" if fake_percentage > real_percentage else "green")

# Create GUI
root = tk.Tk()
root.title("DeepShield - AI Deepfake Detector")
root.geometry("400x450")

Label(root, text="DeepShield - Deepfake Detector", font=("Arial", 14, "bold")).pack(pady=10)
Button(root, text="Select Video", command=detect_deepfake, font=("Arial", 12)).pack(pady=10)
image_label = Label(root)
image_label.pack()
result_label = Label(root, text="", font=("Arial", 14, "bold"))
result_label.pack(pady=10)

root.mainloop()
