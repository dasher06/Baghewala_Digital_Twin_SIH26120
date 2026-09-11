"""
training/train_failure.py

Trains MODEL 4 (Failure/Anomaly Model): a small CNN classifying rendered
dynamometer-card images into one of 5 classes:
    normal / rod_floating / gas_interference / pump_off / other_anomaly

Per specification:
    "Model type: CNN on rendered card images (load-position plot
    rasterized as image)."
    "Loss: Standard classification cross-entropy -- no physics residual
    term (card shape is a discrete-regime classification problem, not a
    continuous physical quantity with a clean differentiable formula)."
    "The physics is primarily used to GENERATE TRAINING DATA here rather
    than forcing a physics loss into the CNN."

This matches Model 4's architecture note in the cross-reference table:
    Mechanism: Physics-simulated training data (Gibbs wave equation)
    Not used: Physics loss term

Architecture: small CNN (3 conv blocks + 2 FC layers), trained on the
64x64 grayscale synthetic card images from
data/synthetic/dynacard_data/images/, using data/synthetic/dynacard_data/labels.csv.

Run:
    cd Baghewala_Digital_Twin
    python training/train_failure.py

Output:
    models/failure/failure_cnn.pt
    models/failure/class_names.json
    models/failure/training_report.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DYNACARD_DIR = os.path.join(BASE_DIR, "data", "synthetic", "dynacard_data")
IMAGES_DIR = os.path.join(DYNACARD_DIR, "images")
LABELS_PATH = os.path.join(DYNACARD_DIR, "labels.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "failure")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = 64
BATCH_SIZE = 32
N_EPOCHS = 18
LR = 1e-3
SEED = 7

torch.manual_seed(SEED)
np.random.seed(SEED)


class DynacardDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["filename"])
        img = Image.open(img_path).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = 1.0 - arr  # invert: card line becomes high-value on low background
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
        label = int(row["class_id"])
        return tensor, label


class FailureCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # 64 -> 32 -> 16 -> 8 after 3 pools
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # logits
        return x


def train():
    df = pd.read_csv(LABELS_PATH)
    class_names = sorted(df["class_name"].unique(), key=lambda c: df[df.class_name == c]["class_id"].iloc[0])
    n_classes = len(class_names)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["class_name"]
    )

    train_ds = DynacardDataset(train_df, IMAGES_DIR)
    test_ds = DynacardDataset(test_df, IMAGES_DIR)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FailureCNN(n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()  # standard cross-entropy, per spec: no physics loss term

    history = []
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_ds)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        test_acc = accuracy_score(all_labels, all_preds)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "test_acc": round(test_acc, 4)})
        print(f"Epoch {epoch:2d}/{N_EPOCHS}  train_loss={train_loss:.4f}  test_acc={test_acc:.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    final_acc = accuracy_score(all_labels, all_preds)
    final_f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds).tolist()

    report = {
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "n_classes": n_classes,
        "class_names": list(class_names),
        "final_test_accuracy": round(float(final_acc), 4),
        "final_test_macro_f1": round(float(final_f1), 4),
        "confusion_matrix": cm,
        "confusion_matrix_note": "rows=true class, cols=predicted class, order matches class_names",
        "epoch_history": history,
        "data_is_synthetic": True,
        "note": (
            "Trained entirely on physics-simulated synthetic dynamometer cards "
            "(see physics/rod_dynamics.py). Real/public card datasets should be "
            "used to validate and reduce sim-to-real gap before field deployment."
        ),
    }

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "failure_cnn.pt"))
    with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
        json.dump(list(class_names), f, indent=2)
    with open(os.path.join(MODEL_DIR, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: v for k, v in report.items() if k != "epoch_history"}, indent=2))
    print(f"\nSaved model to {MODEL_DIR}/failure_cnn.pt")
    return report


if __name__ == "__main__":
    train()
