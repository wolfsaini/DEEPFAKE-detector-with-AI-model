import torch
import torch.nn as nn
import torch.optim as optim
from dataset import dataloader  # Import DataLoader from dataset.py
from model import DeepFakeDetector  # Import model

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = DeepFakeDetector().to(device)

# Define loss function & optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary classification loss
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train model
print("🚀 Training started...")
for epoch in range(5):  # Train for 5 epochs
    for images, labels in dataloader:
        images, labels = images.to(device), labels.float().to(device)
        optimizer.zero_grad()
        loss = criterion(model(images).squeeze(), labels)
        loss.backward()
        optimizer.step()
    print(f"✅ Epoch {epoch+1} completed.")

# Save trained model
torch.save(model.state_dict(), "deepfake_model.pth")
print("🎯 Model training complete! Saved as deepfake_model.pth")
