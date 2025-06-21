import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
csv_path = 'runs/train/shiprs_yolov5s2/results.csv'
df = pd.read_csv(csv_path)

# Clean column names
df.columns = df.columns.str.strip()

# Plotting
plt.figure(figsize=(12, 8))

# 🔹 Training Losses
plt.subplot(2, 2, 1)
plt.plot(df['train/box_loss'], label='Box Loss')
plt.plot(df['train/obj_loss'], label='Object Loss')
plt.plot(df['train/cls_loss'], label='Class Loss')
plt.title('Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 🔹 Precision and Recall
plt.subplot(2, 2, 2)
plt.plot(df['metrics/precision'], label='Precision')
plt.plot(df['metrics/recall'], label='Recall')
plt.title('Precision and Recall')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()

# 🔹 mAP Scores
plt.subplot(2, 2, 3)
plt.plot(df['metrics/mAP_0.5'], label='mAP@0.5')
plt.plot(df['metrics/mAP_0.5:0.95'], label='mAP@0.5:0.95')
plt.title('mAP Scores')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()

plt.tight_layout()

# Save as PNG
output_path = 'runs/train/shiprs_yolov5s2/results.png'
plt.savefig(output_path)
plt.show()

print(f"✅ Saved results graph as {output_path}")
