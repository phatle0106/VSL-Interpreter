import pandas as pd
import matplotlib.pyplot as plt

# đọc log CSV
df = pd.read_csv("train_log.csv")

# Vẽ Loss
plt.figure()
plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig("loss_curve.png")

# Vẽ Accuracy
plt.figure()
plt.plot(df["epoch"], df["train_acc"], label="Train Acc")
plt.plot(df["epoch"], df["val_acc"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.savefig("accuracy_curve.png")

# Vẽ Learning Rate
plt.figure()
plt.plot(df["epoch"], df["lr"], label="Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("LR")
plt.title("Learning Rate Schedule")
plt.legend()
plt.savefig("lr_curve.png")

print("Saved figures: loss_curve.png, accuracy_curve.png, lr_curve.png")