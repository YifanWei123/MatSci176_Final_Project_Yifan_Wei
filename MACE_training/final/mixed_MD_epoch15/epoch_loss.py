import json
from collections import defaultdict
import matplotlib.pyplot as plt

# =================================
# Change this to your training log
# =================================
logfile = "full_model_run-123_train.txt"
# =================================

train_losses = defaultdict(list)
eval_loss = {}
eval_force_mae = {}

# =================================
# Read training log
# =================================
with open(logfile, "r") as f:
    for line in f:
        line = line.strip()

        # Skip non-JSON lines
        if not line.startswith("{"):
            continue

        try:
            data = json.loads(line)
        except Exception:
            continue

        mode = data.get("mode", None)
        epoch = data.get("epoch", None)

        # -------------------------
        # Training loss (mode="opt")
        # -------------------------
        if mode == "opt" and epoch is not None and "loss" in data:
            train_losses[int(epoch)].append(float(data["loss"]))

        # -------------------------
        # Validation metrics
        # -------------------------
        elif mode == "eval" and epoch is not None:
            epoch = int(epoch)

            if "loss" in data:
                eval_loss[epoch] = float(data["loss"])

            if "mae_f" in data:
                eval_force_mae[epoch] = float(data["mae_f"])


# =================================
# Compute average train loss per epoch
# =================================
epochs_train = sorted(train_losses.keys())
train_loss_avg = [
    sum(train_losses[e]) / len(train_losses[e])
    for e in epochs_train
]

epochs_eval_loss = sorted(eval_loss.keys())
eval_loss_vals = [eval_loss[e] for e in epochs_eval_loss]

epochs_eval_f = sorted(eval_force_mae.keys())
eval_force_mae_vals = [eval_force_mae[e] for e in epochs_eval_f]


# =================================
# Print results
# =================================
print("\n=== Train loss vs epoch ===")
for e, v in zip(epochs_train, train_loss_avg):
    print(f"epoch {e:3d}: train_loss_mean = {v:.6f}")

print("\n=== Eval loss vs epoch ===")
for e, v in zip(epochs_eval_loss, eval_loss_vals):
    print(f"epoch {e:3d}: eval_loss = {v:.6f}")

print("\n=== Force MAE vs epoch ===")
for e, v in zip(epochs_eval_f, eval_force_mae_vals):
    print(f"epoch {e:3d}: force_MAE = {v:.6f}")


# =================================
# Plot training vs validation loss
# =================================
plt.figure(figsize=(6,4))

plt.plot(epochs_train, train_loss_avg, label="Train loss")
plt.plot(epochs_eval_loss, eval_loss_vals, label="Validation loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)

plt.show()
