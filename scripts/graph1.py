import matplotlib.pyplot as plt

epochs = list(range(1, 18))

train_loss = [
    0.8498,
    0.7728,
    0.7495,
    0.7339,
    0.7220,
    0.7125,
    0.7045,
    0.6978,
    0.6919,
    0.6867,
    0.6817,
    0.6768,
    0.6723,
    0.6681,
    0.6639,
    0.6598,
    0.6559,
]

val_loss = [
    0.8009,
    0.7717,
    0.7441,
    0.7331,
    0.7245,
    0.7172,
    0.7131,
    0.7105,
    0.7127,
    0.7037,
    0.7041,
    0.7007,
    0.7027,
    0.6997,
    0.7007,
    0.7055,
    0.7029,
]

val_acc = [
    64.41,
    65.85,
    67.07,
    67.56,
    67.99,
    68.34,
    68.35,
    68.52,
    68.31,
    68.75,
    68.82,
    68.99,
    68.89,
    69.04,
    68.92,
    68.83,
    68.94,
]

# Loss curves
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, marker="o", label="Train Loss")
plt.plot(epochs, val_loss, marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
plt.show()

# Accuracy curve
plt.figure(figsize=(8, 5))
plt.plot(epochs, val_acc, marker="o", label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy Over Epochs")
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_curve.png", dpi=150)
plt.show()
