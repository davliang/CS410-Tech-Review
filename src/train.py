from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from models import HAN
from dataset import HANDataset
import pandas as pd


def load_yelp(file_path="data/yelp-2015.json"):
    df = pd.read_json(file_path, lines=True)
    df = df[["stars", "text"]]
    return df


def split_data(df, train_frac=0.8, eval_frac=0.1, test_frac=0.1, random_state=0):
    # Ensure the fractions sum to 1.0
    assert abs(train_frac + eval_frac + test_frac - 1.0) < 1e-6, (
        "Fractions must sum to 1.0"
    )

    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_frac), random_state=random_state
    )
    eval_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=random_state
    )

    return train_df, eval_df, test_df


def train_model(
    model,
    train_dataloader,
    eval_dataloader,
    num_epochs=5,
    lr=1e-3,
    device=torch.device("cpu"),
):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_docs, batch_labels in train_dataloader:
            batch_docs = batch_docs.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits, _, _ = model(batch_docs)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_docs.size(0)
        avg_loss = running_loss / len(train_dataloader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {avg_loss:.4f}")

        # Evaluate on validation set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for val_docs, val_labels in eval_dataloader:
                val_docs = val_docs.to(device)
                val_labels = val_labels.to(device)
                logits, _, _ = model(val_docs)
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == val_labels).sum().item()
                total += val_labels.size(0)
        val_acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch + 1}/{num_epochs} - Validation Accuracy: {val_acc:.4f}")


def main():
    print("Loading Yelp dataset...")
    df = load_yelp()
    print(f"Loaded {len(df)} samples from Yelp dataset.")
    print(df.head())

    print("Splitting data into train, eval, and test sets...")
    train_df, eval_df, test_df = split_data(df)

    train_documents = train_df["text"].tolist()
    train_labels = train_df["stars"].tolist()

    eval_documents = eval_df["text"].tolist()
    eval_labels = eval_df["stars"].tolist()

    test_documents = test_df["text"].tolist()
    test_labels = test_df["stars"].tolist()

    print("Creating HAN datasets...")
    train_dataset = HANDataset(train_documents, train_labels)
    eval_dataset = HANDataset(eval_documents, eval_labels)
    test_dataset = HANDataset(test_documents, test_labels)

    print("Creating DataLoaders...")
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    vocab_size = len(train_dataset.vocab)
    embedding_dim = 200  # Arbitrary Word2Vec embedding size

    # Hyperparameters for GRU layers
    word_hidden_dim = 50
    sent_hidden_dim = 50
    num_classes = 5  # 1-5 star ratings

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Initializing HAN model...")
    model = HAN(
        vocab_size,
        embedding_dim,
        word_hidden_dim,
        sent_hidden_dim,
        num_classes,
        device=device,
    )

    print("Training HAN model...")
    train_model(
        model, train_dataloader, eval_dataloader, num_epochs=5, lr=1e-3, device=device
    )


if __name__ == "__main__":
    main()
