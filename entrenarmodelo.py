import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from tqdm import tqdm

# -------------------- Config --------------------
DATA_DIR = 'dataset'          # Fight/ y NonFight/
SEQ_LEN = 16
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------- Transform --------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -------------------- Step 1: Extraer features CNN --------------------
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
cnn = nn.Sequential(*list(resnet.children())[:-1]).to(DEVICE)
cnn.eval()

for cls in os.listdir(DATA_DIR):
    cls_dir = os.path.join(DATA_DIR, cls)
    feature_dir = os.path.join(DATA_DIR, cls + '_features')
    os.makedirs(feature_dir, exist_ok=True)

    for video_file in tqdm(os.listdir(cls_dir), desc=f"Extrayendo features {cls}"):
        if not video_file.endswith(('.mp4','.avi','.mov')):
            continue
        video_path = os.path.join(cls_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform(frame).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feat = cnn(frame).squeeze(-1).squeeze(-1).cpu()
            frames.append(feat)
        cap.release()
        np.save(os.path.join(feature_dir, video_file.split('.')[0]+'.npy'), np.stack(frames))

# -------------------- Step 2: Dataset LSTM --------------------
class FeatureVideoDataset(Dataset):
    def __init__(self, root_dir, sequence_length=SEQ_LEN):
        self.samples = []
        self.sequence_length = sequence_length
        for cls_idx, cls in enumerate(os.listdir(root_dir)):
            cls_dir = os.path.join(root_dir, cls)
            if not cls.endswith('_features'):
                continue
            for file in os.listdir(cls_dir):
                if file.endswith('.npy'):
                    self.samples.append((os.path.join(cls_dir, file), cls_idx))
                    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        feats = np.load(path)
        if len(feats) >= self.sequence_length:
            start = np.random.randint(0, len(feats) - self.sequence_length + 1)
            seq = feats[start:start+self.sequence_length]
        else:
            while len(feats) < self.sequence_length:
                feats = np.concatenate([feats, feats], axis=0)
            seq = feats[:self.sequence_length]
        return torch.tensor(seq, dtype=torch.float), label

dataset = FeatureVideoDataset(DATA_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# -------------------- Step 3: LSTM Classifier --------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTMClassifier().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------- Step 4: Entrenamiento --------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for X, y in loop:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs,1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        loop.set_postfix(loss=running_loss/(total/BATCH_SIZE), acc=correct/total*100)

torch.save(model.state_dict(), 'fight_detection_lstm_end2end.pth')
print("Entrenamiento completo y modelo guardado: fight_detection_lstm_end2end.pth")
