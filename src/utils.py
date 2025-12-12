import re
import pandas as pd

def clean_text(text: str) -> str:
    """
    Basic preprocessing function: lowercase, remove special characters.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

def load_dataset(csv_path: str, text_col: str, label_col: str):
    """
    Reads a CSV and returns text + labels.
    """
    df = pd.read_csv(csv_path)
    df[text_col] = df[text_col].astype(str).apply(clean_text)
    return df[text_col].tolist(), df[label_col].tolist()