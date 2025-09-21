File: requirements.txt
streamlit
torch
torchvision
pillow
numpy
File: models.py
import os
import torch
import torch.nn as nn
from torchvision import models
CLASSES = [
 "no_defect",
 "picking",
 "lamination",
 "capping",
 "sticking",
 "binding",
 "cracking",
 "mottling",
 "chipping"
]
DEFECT_DESCRIPTIONS = {
 "picking": "Picking refers to removing a small amount of material from a tablet's surface
by a punch. This material adheres to the face of the punch, resulting in 'picking.'",
 "lamination": "Lamination refers to separation of layers in a tablet, caused by trapped air
or low binder levels.",
 "capping": "Tablet capping occurs when the top or bottom of a tablet separates from the
main body.",
 "sticking": "Sticking occurs when tablet surface adheres to the punch face or die wall
during compression, causing pitting or surface detachment.",
 "binding": "Binding happens when powder adheres to punch edges or die, preventing smooth
ejection or causing splits.",
 "cracking": "Cracking is when small cracks appear on the top or bottom central surfaces of
tablets.",
 "mottling": "Mottling is uneven color distribution on a tablet surface (dark/light
patches).",
 "chipping": "Chipping is when the edges of a tablet break off during ejection or
handling.",
 "no_defect": "Tablet surface is normal with no visible defects."
}
def build_model(num_classes=len(CLASSES), pretrained=True):
 model = models.resnet18(pretrained=pretrained)
 in_features= model.fc.in_features
 model.fc = nn.Linear(in_features, num_classes)
 return model
def load_model(weights_path=None, device='cpu'):
 model = build_model()
 model.to(device)
 if weights_path and os.path.exists(weights_path):
 model.load_state_dict(torch.load(weights_path, map_location=device))
 model.eval()
 return model
File: feedback_manager.py
import os, json
from datetime import datetime
FEEDBACK_FILE = "data/feedback.json"
def log_feedback(image_path, predicted_label, correct_label):
 entry = {
 "image": image_path,
 "predicted": predicted_label,
 "correct": correct_label,
 "timestamp": datetime.now().isoformat()
 }
 os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
 if not os.path.exists(FEEDBACK_FILE):with open(FEEDBACK_FILE, "w") as f:
 json.dump([entry], f, indent=2)
 else:
 with open(FEEDBACK_FILE, "r") as f:
 data = json.load(f)
 data.append(entry)
 with open(FEEDBACK_FILE, "w") as f:
 json.dump(data, f, indent=2)
 print(f"Logged feedback: {entry}")
File: retrain.py
import json
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from models import build_model, CLASSES
FEEDBACK_FILE = "data/feedback.json"
class FeedbackDataset(Dataset):
 def __init__(self, feedback):
 self.feedback = feedback
 self.transform = transforms.Compose([
 transforms.Resize((224,224)),
 transforms.ToTensor()
 ])
 def __len__(self):
 return len(self.feedback)def __getitem__(self, idx):
 item = self.feedback[idx]
 from PIL import Image
 img = Image.open(item['image']).convert('RGB')
 label_name = item['correct']
 if label_name in CLASSES:
 label = CLASSES.index(label_name)
 else:
 label = CLASSES.index("no_defect")
 return self.transform(img), label
def retrain(epochs=3, batch_size=4, lr=1e-5, device='cpu'):
 if not os.path.exists(FEEDBACK_FILE):
 print("No feedback collected yet.")
 return
 with open(FEEDBACK_FILE, 'r') as f:
 feedback = json.load(f)
 if len(feedback) == 0:
 print("Feedback file empty.")
 return
 dataset = FeedbackDataset(feedback)
 loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
 model = build_model(num_classes=len(CLASSES), pretrained=False)
 model.to(device)
 if os.path.exists('models/adg126_finetuned.pth'):
 model.load_state_dict(torch.load('models/adg126_finetuned.pth', map_location=device))
 elif os.path.exists('models/adg126_initial.pth'):
 model.load_state_dict(torch.load('models/adg126_initial.pth', map_location=device))import torch.optim as optim
 criterion = torch.nn.CrossEntropyLoss()
 optimizer = optim.Adam(model.parameters(), lr=lr)
 model.train()
 for epoch in range(epochs):
 running = 0.0
 for imgs, labels in loader:
 imgs = imgs.to(device)
 labels = labels.to(device)
 optimizer.zero_grad()
 outputs = model(imgs)
 loss = criterion(outputs, labels)loss.backward()
 optimizer.step()running += loss.item()
 avg_loss = running / max(1, len(loader))
 print(f'Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}')
 os.makedirs('models', exist_ok=True)
 torch.save(model.state_dict(), 'models/adg126_finetuned.pth')
 print("Retraining complete. Saved to models/adg126_finetuned.pth")
if __name__ == '__main__':
 retrain()
File: app.py
import os
import streamlit as st
from models import build_model, CLASSES, DEFECT_DESCRIPTIONS
import torch
from PIL import Image
from torchvision import transforms
from feedback_manager import log_feedback
st.set_page_config(page_title="adg126 - Pharma Defect Detector")
UPLOAD_DIR = "static_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
device = "cpu"
model = build_model()
if os.path.exists("models/adg126_finetuned.pth"):
 model.load_state_dict(torch.load("models/adg126_finetuned.pth", map_location=device))
elif os.path.exists("models/adg126_initial.pth"):
 model.load_state_dict(torch.load("models/adg126_initial.pth", map_location=device))
model.eval()
transform = transforms.Compose([
 transforms.Resize((224,224)),
 transforms.ToTensor()
])st.title("adg126 - Self-learning Pharma Defect Detector")
st.write("Supported defects:", ", ".join(CLASSES[1:]))
uploaded_file = st.file_uploader("Upload tablet image", type=["jpg","jpeg","png"])
if uploaded_file is not None:
 save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
 with open(save_path, "wb") as f:
 f.write(uploaded_file.getbuffer())
 st.image(save_path, caption="Uploaded image", use_column_width=True)
 img = Image.open(save_path).convert("RGB")
 x = transform(img).unsqueeze(0)
 with torch.no_grad():
 logits = model(x)
 pred = torch.argmax(logits, dim=1).item()
 label = CLASSES[pred]
 desc = DEFECT_DESCRIPTIONS.get(label, "")
 st.subheader(f"Predicted: {label}")
 st.write(desc)
 st.markdown("---")
 st.subheader("Feedback (help improve the model)")
 correct_label = st.selectbox("Correct label", ["-- select --"] + CLASSES)
 if st.button("Submit Feedback"):
 if correct_label == "-- select --":
 st.warning("Please select a correct label before submitting feedback.")
 else:
 log_feedback(save_path, label, correct_label)
 st.success("Thanks — feedback recorded. Run retrain.py to fine-tune the model with
collected feedback.")# adg126 - Streamlit Self-learning Pharma Defect Detector
## Run Instructions1. Install dependencies:
 ```bash
 pip install -r requirements.txt
 ```
2. (Optional) Train initial model if you have labelled dataset:
 ```bash
 python train.py
 ```
3. Run the Streamlit app:
 ```bash
 streamlit run app.py
 ```
4. Upload images, review predictions, and submit feedback.
 Feedback saved to `data/feedback.json`.
5. Fine-tune the model with feedback:
 ```bash
 python retrain.py
