DeepShield - AI-Based Deepfake Video Detector

📌 Overview

DeepShield is an AI-powered deepfake detection application that analyzes video files and classifies them as real or fake. Built using PyTorch and OpenCV, the project extracts frames from videos, processes them through a trained CNN model, and provides an accuracy percentage for deepfake detection.

🚀 Features

✅ AI-powered deepfake detection using a CNN model✅ Video upload & preview functionality✅ Real-time frame analysis using PyTorch✅ Displays accuracy percentage (Real vs. Fake)✅ Interactive and user-friendly Tkinter GUI

🛠️ Tools & Technologies Used

📌 Frameworks & Libraries

PyTorch (Deep Learning Framework)

OpenCV (Video Processing)

Tkinter (GUI Development)

NumPy (Numerical Operations)

PIL (Pillow) (Image Handling)

📌 Hardware & Software Requirements

Operating System: Windows / Linux / macOS

RAM: Minimum 8GB (16GB Recommended)

GPU (Optional): CUDA-enabled GPU for faster processing

Python Version: 3.8+

📂 Project Structure

DeepShield/
│── model.py                  # DeepFakeDetector model architecture
│── deepfake_detector.py       # Core logic for video analysis
│── gui.py                     # Tkinter GUI implementation
│── preprocess.py              # Frame extraction and preprocessing
│── README.md                  # Project Documentation
│── requirements.txt            # Dependencies list
│── deepfake_model.pth          # Trained PyTorch model
│── dataset/                    # Real and Fake video dataset
│── assets/                     # UI assets (if any)

🔧 Installation & Setup

1️⃣ Clone the repository

git clone https://github.com//deepshield.git
cd deepshield

2️⃣ Create a virtual environment (Recommended)

python -m venv env
source env/bin/activate  # For Linux/Mac
env\Scripts\activate     # For Windows

3️⃣ Install dependencies

pip install -r requirements.txt

4️⃣ Run the GUI Application

python gui.py

📊 Methodology & Implementation

1️⃣ Data Collection & Preprocessing

Extract frames from videos using OpenCV

Resize to 128×128 pixels

Normalize pixel values (0-1 range)

Convert to PyTorch tensors

2️⃣ Model Training (CNN-based DeepFakeDetector)

Input: Processed video frames

Architecture: Convolutional Neural Network (CNN)

Loss Function: Binary Cross-Entropy Loss

Optimizer: Adam

Output: Probability of being Fake or Real

3️⃣ Evaluation & Accuracy Calculation

Measure accuracy, precision, recall, and F1-score

Convert probabilities to percentage-based accuracy

4️⃣ GUI Integration & Deployment

Implement Tkinter-based GUI

Provide video preview & real-time processing

Display real vs. fake percentage


🤖 How It Works

1️⃣ Upload a video file via the GUI2️⃣ The model analyzes frames and classifies them as real or fake3️⃣ Displays real vs. fake percentage in the GUI4️⃣ User gets an instant detection result

📌 Future Improvements

🔹 Implement real-time webcam deepfake detection🔹 Train with a larger, more diverse dataset🔹 Improve accuracy with advanced architectures (e.g., ResNet, Vision Transformers)🔹 Optimize processing for faster detection

🤝 Contributing

Feel free to fork this repository, make changes, and submit pull requests. Contributions are welcome!

📝 License

This project is open-source and available under the MIT License.

📧 For queries or collaborations: Contact 121abhay2saini@gmail.com
