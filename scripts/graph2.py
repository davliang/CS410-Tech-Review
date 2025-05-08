import os

import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from gensim.models import Word2Vec
from rich.progress import track

from cs410_han.config import settings
from cs410_han.data import (
    load_data,
    split_data,
    YelpReviewDataset,
    create_data_loader,
    build_vocab,
    create_embedding_matrix,
    get_max_lengths,
)
from cs410_han.model import HAN

os.makedirs("graphs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Only use the test set for a clean confusion matrix
raw = load_data(settings.data_path)
_, _, test_raw = split_data(raw, settings.seed)

train_tok = joblib.load(settings.cache_train_path)
val_tok = joblib.load(settings.cache_val_path)
test_tok = joblib.load(settings.cache_test_path)

# Build vocab which only needs the train set, but val and test are needed for lengths
word_to_idx, _ = build_vocab(train_tok, settings.min_freq)

# Load the Word2Vec and build embedding matrix
w2v_model = Word2Vec.load(str(settings.cache_w2v_model_path))
embedding_matrix = create_embedding_matrix(w2v_model, word_to_idx, settings.embed_dim)

# Figure out max lengths
max_sent_len, max_doc_len = get_max_lengths(train_tok, val_tok, test_tok)

# Make DataLoader for test set
test_dataset = YelpReviewDataset(test_tok, word_to_idx, max_sent_len, max_doc_len)
test_loader = create_data_loader(test_dataset, settings.batch_size, shuffle=False)

# Init HAN and load best weights from running the training script
model = HAN(
    vocab_size=len(word_to_idx),
    embed_dim=settings.embed_dim,
    word_hidden_dim=settings.word_hidden_dim,
    sent_hidden_dim=settings.sent_hidden_dim,
    num_classes=settings.num_classes,
    embedding_matrix=embedding_matrix,
    pad_idx=word_to_idx["<PAD>"],
).to(device)

state = torch.load(settings.model_save_dir / "han_model_best.pt", map_location=device)
model.load_state_dict(state)
model.eval()

# Run inference
all_preds = []
all_labels = []
with torch.no_grad():
    for docs, labels, sent_lens, doc_lens in track(
        test_loader, description="Running test batches"
    ):
        docs = docs.to(device)
        sent_lens = sent_lens.to(device)
        doc_lens = doc_lens.to(device)
        outputs, _, _ = model(docs, sent_lens, doc_lens)
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Compute and plot confusion matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.xlabel("Predicted star rating")
plt.ylabel("True star rating")
ticks = np.arange(settings.num_classes)
plt.xticks(ticks, ticks + 1)
plt.yticks(ticks, ticks + 1)
plt.colorbar()
for i in range(settings.num_classes):
    for j in range(settings.num_classes):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("graphs/confusion_matrix.png", dpi=300)
plt.show()
