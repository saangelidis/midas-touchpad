from pathlib import Path
import pandas as pd
from PIL import Image

data = Path(__file__).parent / "data"

# Find files
labels_path = list(data.rglob("hasy-data-labels.csv"))[0]
symbols_path = list(data.rglob("symbols.csv"))[0]

# Load metadata
labels_df = pd.read_csv(labels_path)
symbols_df = pd.read_csv(symbols_path)

print("symbols.csv columns:")
print(symbols_df.columns.tolist())

# Find equiv
equiv_row = symbols_df[symbols_df["latex"] == r"\equiv"].iloc[0]
equiv_id = equiv_row["symbol_id"]

print("Equiv symbol ID:", equiv_id)

# Find first handwritten equiv image
sample = labels_df[labels_df["symbol_id"] == equiv_id].iloc[0]
image_path = labels_path.parent / sample["path"]

print("Image:", image_path)

image = Image.open(image_path)

print("Image size:", image.size)

image.show()