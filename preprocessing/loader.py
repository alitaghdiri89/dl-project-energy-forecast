import pandas as pd

# -----------------------------
# Data Loader
# -----------------------------

def load_group(mode: str, key: str, pickle_block: str, pickle_house: str):
    if mode == "block":
        path = pickle_block
        id_col = "block_number"
    elif mode == "house":
        path = pickle_house
        id_col = "LCLid"
    else:
        raise ValueError("MODE must be 'block' or 'house'")

    df = pd.read_pickle(path)

    if key not in df[id_col].values:
        raise ValueError(f"{id_col} '{key}' not found in DataFrame.")

    return df[df[id_col] == key].copy()
