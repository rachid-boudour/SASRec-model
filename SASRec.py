import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder


# 1. 🔹 Chargement du dataset1
df = pd.read_csv("interactions200K_collab_V2.csv")

# 2. 🔹 Tri par utilisateur et par timestep
df = df.sort_values(by=["user_id", "timestamp"])

# 3. 🔹 Encodage des items
item_encoder = LabelEncoder()
df["service_id_enc"] = item_encoder.fit_transform(df["service_id"])

# Mapping inverse si besoin
idx2item = dict(enumerate(item_encoder.classes_))

# 4. 🔹 Construction des séquences par utilisateur
def make_sequences(df, min_len=2):
    sequences = []
    grouped = df.groupby("user_id")["service_id_enc"].apply(list)
    for items in grouped:
        if len(items) >= min_len:
            for i in range(1, len(items)):
                seq = items[:i]
                target = items[i]
                sequences.append((seq, target))
    return sequences

sequences = make_sequences(df)

# 5. 🔹 Dataset PyTorch
class SequenceDataset(Dataset):
    def __init__(self, sequences, max_len, item_count):
        self.sequences = sequences
        self.max_len = max_len
        self.item_count = item_count

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        padded_seq = [0] * (self.max_len - len(seq)) + seq[-self.max_len:]
        return torch.tensor(padded_seq), torch.tensor(target)

max_seq_len = 10
item_count = len(item_encoder.classes_)
dataset = SequenceDataset(sequences, max_seq_len, item_count)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 6. 🔹 Modèle Transformer simple
class TransformerRecModel(nn.Module):
    def __init__(self, item_count, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(item_count, d_model, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, item_count)

    def forward(self, x):
        emb = self.embedding(x)  # [batch_size, seq_len, d_model]
        emb = emb.permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        out = self.transformer(emb)  # [seq_len, batch_size, d_model]
        out = out[-1]  # dernier token: [batch_size, d_model]
        return self.output(out)

model = TransformerRecModel(item_count)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 7. 🔹 Entraînement
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    total_loss = 0
    correct, total = 0, 0

    for batch in loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        preds = model(x)
        loss = loss_fn(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 🔹 Calcul précision
        _, predicted = torch.max(preds, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

    acc = 100 * correct / total
    print(f"📚 Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f} | Accuracy: {acc:.2f}%")

# 8. 🔹 Fonction de prédiction
def recommend_next(session_items, model, item_encoder, max_len=10):
    model.eval()
    encoded = item_encoder.transform(session_items)
    padded = [0] * (max_len - len(encoded)) + list(encoded[-max_len:])
    X = torch.tensor([padded]).to(device)
    with torch.no_grad():
        scores = model(X)
        top_item_id = torch.argmax(scores, dim=-1).item()
        return item_encoder.inverse_transform([top_item_id])[0]

# 🔍 Exemple de prédiction
example_session = df[df["user_id"] == df["user_id"].iloc[0]]["service_id"].tolist()[:2]
print("▶️ Historique:", example_session)
predicted = recommend_next(example_session, model, item_encoder)
print("🔮 Item recommandé :", predicted)
