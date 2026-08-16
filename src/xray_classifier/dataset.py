from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def label_from_path(path: str | Path) -> int:
    text = str(path).lower()
    if "positive" in text:
        return 1
    if "negative" in text:
        return 0
    raise ValueError(f"Cannot find MURA label in path: {path}")


def resolve_mura_path(raw_path: str | Path, data_dir: str | Path) -> Path:
    raw_path = Path(str(raw_path))
    data_dir = Path(data_dir)

    if raw_path.is_absolute():
        return raw_path
    if str(raw_path).startswith("MURA-v1.1"):
        return data_dir.parent / raw_path
    return data_dir / raw_path


def load_mura_split(data_dir: str | Path, split: str) -> pd.DataFrame:
    data_dir = Path(data_dir)
    csv_path = data_dir / f"{split}_image_paths.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path, header=None, names=["path"])
        df["path"] = df["path"].apply(lambda path: str(resolve_mura_path(path, data_dir)))
    else:
        image_files = []
        for extension in ["*.png", "*.jpg", "*.jpeg"]:
            image_files.extend((data_dir / split).rglob(extension))
        df = pd.DataFrame({"path": [str(path) for path in image_files]})

    df["label"] = df["path"].apply(label_from_path)
    return df.reset_index(drop=True)


class MuraDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        label = int(row["label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label
