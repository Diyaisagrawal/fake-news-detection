import pandas as pd

# Load datasets
fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

# Add labels
fake_df["label"] = 0   # Fake
true_df["label"] = 1   # Real

# Combine datasets
df = pd.concat([fake_df, true_df], axis=0)

# Shuffle data
df = df.sample(frac=1).reset_index(drop=True)

# Keep only needed columns
df = df[["text", "label"]]

print(df.head())
print(df["label"].value_counts())

