
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
SKETCHES_DIR = DATA_DIR / "sketches"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Dataset paths
BONN_DATA_ROOT = RAW_DATA_DIR / "furniture_dataset" / "Bonn_Furniture_Styles_Dataset" / "houzz"
SPLITS_ROOT = RAW_DATA_DIR / "furniture_dataset" / "Bonn_Furniture_Styles_Dataset" / "splits"

# Classes
SELECTED_CLASSES = ["beds", "chairs", "dressers", "sofas", "tables"]
EXCLUDED_CLASSES = ["lamps"]
CLASS_NAME_MAP = {
    "beds": "bed",
    "chairs": "chair",
    "dressers": "dresser",
    "sofas": "sofa",
    "tables": "table",
}

# Settings
IMAGE_SIZE = (128, 128)
RANDOM_STATE = 42
SAMPLES_PER_CLASS_PROTOTYPE = {
    "train": 350,
    "val": 75,
    "test": 75,
}



def parse_split_file(split_path, split_name, selected_classes=None, exclude_classes=None):
    """
    Parse a Bonn dataset split file and return a DataFrame.

    Parameters
    ----------
    split_path : str or Path
        Path to the split file.
    split_name : str
        Name of the dataset split, e.g. 'train', 'val', or 'test'.
    selected_classes : list, optional
        List of class names to keep, e.g. ['beds', 'chairs'].
    exclude_classes : list, optional
        List of class names to exclude, e.g. ['lamps'].

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: split, path, category, class_name, style
    """
    records = []
    split_path = Path(split_path)

    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            style = parts[0]
            img_path = parts[1]
            category = img_path.split("/")[1]

            if selected_classes and category not in selected_classes:
                continue

            if exclude_classes and category in exclude_classes:
                continue

            records.append({
                "split": split_name,
                "path": BONN_DATA_ROOT.parent / img_path,
                "relative_path": img_path,
                "category": category,
                "class_name": CLASS_NAME_MAP.get(category, category),
                "style": style,
            })

    return pd.DataFrame(records).reset_index(drop=True)




def load_dataset_splits(drop_duplicates=True):
    """
    Load train, validation, and test split files into one DataFrame.

    Parameters
    ----------
    drop_duplicates : bool, optional
        If True, duplicate image paths are removed.

    Returns
    -------
    pd.DataFrame
        Combined metadata DataFrame for train, validation, and test splits.
    """
    split_files = {
        "train": SPLITS_ROOT / "train_split.txt",
        "val": SPLITS_ROOT / "val_split.txt",
        "test": SPLITS_ROOT / "test_split.txt",
    }

    dataframes = []

    for split_name, split_path in split_files.items():
        if not split_path.exists():
            split_path = split_path.with_suffix("")

        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        df_split = parse_split_file(
            split_path=split_path,
            split_name=split_name,
            selected_classes=SELECTED_CLASSES,
            exclude_classes=EXCLUDED_CLASSES,
        )

        dataframes.append(df_split)

    df = pd.concat(dataframes, ignore_index=True)

    if drop_duplicates:
        df = remove_duplicate_paths(df)

    return df



def load_image(path, size=IMAGE_SIZE, normalize=False):
    """
    Load and resize an image from disk.

    Parameters
    ----------
    path : str or Path
        Path to the image file.
    size : tuple, optional
        Target size as (width, height).
    normalize : bool, optional
        If True, convert pixel values from [0, 255] to [0, 1].

    Returns
    -------
    np.ndarray or None
        RGB image array, or None if loading fails.
    """
    path = Path(path)

    img = cv2.imread(str(path))

    if img is None:
        return None

    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if normalize:
        img = img.astype("float32") / 255.0

    return img



def remove_duplicate_paths(df):
    """
    Remove duplicated image paths from the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset metadata DataFrame containing a 'relative_path' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicate image paths removed.
    """
    duplicates_count = df["relative_path"].duplicated().sum()

    if duplicates_count > 0:
        print(f"Found {duplicates_count} duplicated image path(s). Removing duplicates...")

    df_clean = df.drop_duplicates(subset="relative_path").reset_index(drop=True)

    return df_clean