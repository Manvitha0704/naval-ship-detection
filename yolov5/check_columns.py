import pandas as pd

df = pd.read_csv('runs/train/shiprs_yolov5s2/results.csv')
df.columns = df.columns.str.strip()

print("\n📋 Actual Columns in Your CSV:\n")
for col in df.columns:
    print(repr(col))
