import asyncio
import websockets
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import VGG19_Weights
from PIL import Image
import io
import json

class YourModel(nn.Module):
    def __init__(self, num_classes):
        super(YourModel, self).__init__()
        self.vgg19 = models.vgg19(weights=VGG19_Weights.DEFAULT)
        for param in self.vgg19.features.parameters():
            param.requires_grad = False
        self.vgg19.classifier[-1] = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.vgg19(x)
        return x

num_classes = 2  # Assuming binary classification
model = YourModel(num_classes)

model_path = "C:/Users/Dell/OneDrive/Documents/safelens_model.pth"  # Adjust the path as necessary
state_dict = torch.load(model_path, map_location=torch.device('cpu'))

new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace("features", "vgg19.features").replace("classifier", "vgg19.classifier")
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)
model.eval()

class_names = ["shirted", "shirtless"]

def predict(model, img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img)

    with torch.no_grad():
        outputs = model(img_tensor.unsqueeze(0))
        predictions = torch.softmax(outputs, dim=1)

    predictions_np = predictions.numpy()
    predicted_class_index = np.argmax(predictions_np)
    predicted_class = class_names[predicted_class_index]
    confidence = round(100 * predictions_np[0, predicted_class_index], 2)

    return predicted_class, confidence

async def handle_connection(websocket, path):
    async for message in websocket:
        try:
            print("Message received")
            img = Image.open(io.BytesIO(message))  # Read image from bytes
            print(f"Image received: {img.size}")  # Print image size
            predicted_class, confidence = predict(model, img)
            print(f"Predicted class: {predicted_class}, Confidence: {confidence}")  # Print predictions
            result = {"class": predicted_class, "confidence": confidence}
            await websocket.send(json.dumps(result))  # Send the prediction result
            print("Result sent:", result)
        except Exception as e:
            print(f"An error occurred: {e}")
            break

start_server = websockets.serve(handle_connection, "127.0.0.1", 12345)  # Changed port to 12345

print("Server listening on ws://127.0.0.1:12345")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
#python backend.py