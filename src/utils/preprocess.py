import re
import argparse
import html
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from loguru import logger
from sklearn.model_selection import train_test_split

from src.utils.logger import setup_logger


class TextPreprocessor:

    def __init__(self, cfg: dict):
        self.min_len = cfg["text"]["min_word_count"]
        self.max_len = cfg["text"]["max_word_count"]
        self.pos_thresh = cfg["rating"]["positive_threshold"]
        self.neg_thresh = cfg["rating"]["negative_threshold"]

    def clean_text(self, text: str) -> str:
        """Full text cleaning pipeline."""
        if not isinstance(text, str):
            return ""

        text = html.unescape(text)                      
        text = re.sub(r'<[^>]+>', ' ', text)            
        text = re.sub(r'\bbr\b', ' ', text)             
        text = re.sub(r"http\S+|www\S+", "", text)      
        text = re.sub(r"[^\w\s.,!?'-]", " ", text)      
        text = re.sub(r"\s+", " ", text).strip()        
        text = re.sub(r'\bbr\b', ' ', text)
        return text

    def rating_to_label(self, rating: float) -> str:
        if rating >= self.pos_thresh:
            return "positive"
        elif rating <= self.neg_thresh:
            return "negative"
        else:
            return "neutral"

    def is_valid(self, text: str) -> bool:
        words = text.split()
        return self.min_len <= len(words) <= self.max_len

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Processing {len(df)} reviews...")

        df = df.copy()
        df["clean_text"] = df["text"].apply(self.clean_text)
        df = df[df["clean_text"].apply(self.is_valid)].reset_index(drop=True)
        df["label"] = df["rating"].apply(self.rating_to_label)
        df["label_id"] = df["label"].map({"negative": 0, "neutral": 1, "positive": 2})
        df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))

        logger.info(f"After filtering: {len(df)} reviews")
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")
        return df


class ImagePreprocessor:
  
    def __init__(self, cfg: dict):
        self.image_size = cfg["data"]["mvtec"]["image_size"]
        self.categories = cfg["data"]["mvtec"]["categories"]

    def resize_and_save(self, src_path: Path, dst_path: Path) -> bool:
        try:
            img = Image.open(src_path).convert("RGB")
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst_path)
            return True
        except Exception as e:
            logger.warning(f"Could not process {src_path}: {e}")
            return False

    def build_metadata(self, raw_mvtec_dir: Path, processed_dir: Path) -> pd.DataFrame:
        records = []

        for category in self.categories:
            cat_dir = raw_mvtec_dir / category
            if not cat_dir.exists():
                logger.warning(f"Category dir not found: {cat_dir} — skipping")
                continue

            for split in ["train", "test"]:
                split_dir = cat_dir / split
                if not split_dir.exists():
                    continue

                for defect_type_dir in split_dir.iterdir():
                    if not defect_type_dir.is_dir():
                        continue

                    defect_type = defect_type_dir.name
                    is_defect = 0 if defect_type == "good" else 1

                    for img_path in defect_type_dir.glob("*.png"):
                        dst_path = (
                            processed_dir / "mvtec" / category /
                            split / defect_type / img_path.name
                        )
                        success = self.resize_and_save(img_path, dst_path)
                        if success:
                            records.append({
                                "image_path": str(dst_path),
                                "category": category,
                                "split": split,
                                "defect_type": defect_type,
                                "label": is_defect,
                            })

        df = pd.DataFrame(records)
        if len(df) > 0:
            out_path = processed_dir / "mvtec_metadata.csv"
            df.to_csv(out_path, index=False)
            logger.success(f"MVTec metadata saved → {out_path}")
            logger.info(f"Total images: {len(df)} | Defects: {df['label'].sum()}")
        else:
            logger.warning("No MVTec images found. Did you download the dataset?")

        return df


def split_dataset(df: pd.DataFrame, cfg: dict, label_col: str = "label") -> dict:
    seed = cfg["data"]["amazon_reviews"]["random_seed"]
    test_size = cfg["data"]["amazon_reviews"]["test_split"]
    val_size = cfg["data"]["amazon_reviews"]["val_split"]

    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[label_col]
    )
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio, random_state=seed, stratify=train_val[label_col]
    )

    logger.info(
        f"Split sizes — Train: {len(train)} | Val: {len(val)} | Test: {len(test)}"
    )
    return {"train": train, "val": val, "test": test}

def main(config_path: str) -> None:
    setup_logger()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    categories = cfg["data"]["amazon_reviews"].get("categories", ["Electronics"])
    raw_dfs = []
    
    for cat in categories:
        file_name = cat.lower().replace("_and_", "&").replace("_", "") + "_reviews.parquet"
        cat_path = raw_dir / "amazon_reviews" / file_name
       
        if not cat_path.exists():
            file_name = cat.lower() + "_reviews.parquet"
            cat_path = raw_dir / "amazon_reviews" / file_name
            
        if cat_path.exists():
            logger.info(f"Loading raw category dataset: {cat} ({file_name})...")
            df_cat = pd.read_parquet(cat_path)
            raw_dfs.append(df_cat)
        else:
            logger.warning(f"Category dataset file not found at {cat_path}")
            
    if raw_dfs:
        logger.info("Combining and preprocessing Amazon Reviews text datasets...")
        df_raw = pd.concat(raw_dfs, ignore_index=True)

        rename_map = {}
        if "review_text" in df_raw.columns:
            rename_map["review_text"] = "text"
        if "overall" in df_raw.columns:
            rename_map["overall"] = "rating"
        if rename_map:
            df_raw = df_raw.rename(columns=rename_map)

        keep_cols = [c for c in ["text", "rating", "verified_purchase",
                                 "asin", "reviewerID", "unixReviewTime"]
                     if c in df_raw.columns]
        df_raw = df_raw[keep_cols]

        text_proc = TextPreprocessor(cfg)
        df_clean = text_proc.process_dataframe(df_raw)

        splits = split_dataset(df_clean, cfg)
        for split_name, split_df in splits.items():
            out = processed_dir / f"reviews_{split_name}.parquet"
            split_df.to_parquet(out, index=False)
            logger.success(f"Saved {split_name} split → {out}")

        sample_path = Path(cfg["data"]["samples_dir"]) / "reviews_sample.parquet"
        Path(cfg["data"]["samples_dir"]).mkdir(parents=True, exist_ok=True)
        sample_n = min(cfg["data"]["sample_size"], len(df_clean))
        df_clean.sample(n=sample_n, random_state=42).to_parquet(sample_path, index=False)
        logger.success(f"Sample ({sample_n} rows) → {sample_path}")
    else:
        logger.error("No Amazon reviews dataset files found in raw/amazon_reviews directory. Run download.py first.")

    mvtec_dir = raw_dir / "mvtec"
    if mvtec_dir.exists() and any(mvtec_dir.iterdir()):
        logger.info("Processing MVTec images...")
        img_proc = ImagePreprocessor(cfg)
        img_proc.build_metadata(mvtec_dir, processed_dir)
    else:
        logger.warning(
            "MVTec dataset not found. Follow the download instructions printed by download_data.py."
        )

    logger.success("=== Preprocessing complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/data_config.yaml",
        help="Path to data config YAML"
    )
    args = parser.parse_args()
    main(args.config)