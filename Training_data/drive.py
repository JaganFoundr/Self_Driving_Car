import socketio
import eventlet
import numpy as np
from flask import Flask
import torch
from io import BytesIO
from PIL import Image
import base64
import cv2
import torch.nn as nn
import torch.nn.functional as F

# Initialize the SocketIO server
sio = socketio.Server()

# Flask app for the server
app = Flask(__name__) 
speed_limit = 10

class NvidiaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)  
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3)  # Changed kernel size to 3x3
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)  # Added 5th convolutional layer

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 1 * 18, 1164)  # Flattened dimension (calculated manually)
        self.fc2 = nn.Linear(1164, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.fc5 = nn.Linear(10, 1)  # Output layer
        
    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))

        x = torch.flatten(x, start_dim=1)
        
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = F.elu(self.fc4(x))
        x = self.fc5(x)
        return x

model = NvidiaModel()

def img_preprocess_inference(img):
    # Ensure img is a PIL.Image object
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.crop((0, 60, img.width, 135))  # Crop the region of interest (ROI)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0  # Normalize the image to [0, 1]
    img = np.transpose(img, (2, 0, 1))  # Convert from HWC to CHW format (required by PyTorch)
    img = torch.tensor(img, dtype=torch.float32)  # Convert to tensor
    img = (img - 0.5) / 0.5  # Normalize to [-1, 1] (equivalent to img/255)
    
    return img


# Load the PyTorch model (no change here)
def load_pytorch_model(model_path):
    model = NvidiaModel()  # Initialize the model
    model.load_state_dict(torch.load(model_path))  # Load the model weights
    model.eval()  # Set the model to evaluation mode
    return model

# SocketIO telemetry event
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))  # Decode the image from base64
    image = np.asarray(image)  # Convert PIL image to numpy array
    image = img_preprocess_inference(image)  # Preprocess the image (same as during training)

    image = image.unsqueeze(0)  # Add batch dimension (1, 3, 66, 200)
    steering_angle = model(image).item()  # Get the predicted steering angle

    throttle = 1.0 - speed / speed_limit  # Simple throttle logic
    print(f"Steering Angle: {steering_angle}, Throttle: {throttle}, Speed: {speed}")
    
    send_control(steering_angle, throttle)

# SocketIO connect event
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)  # Send initial control signals

# Function to send control commands back to the simulator
def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })

# Main entry point for the app
if __name__ == '__main__':
    model = load_pytorch_model('/home/jagannath/Desktop/SELF_DRIVING_DATA/Training_data/NVIDIA_model.pth')  # Load the saved model
    app = socketio.Middleware(sio, app)  # Integrate SocketIO with Flask
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)  # Run the server
