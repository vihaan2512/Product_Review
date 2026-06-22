import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
from loguru import logger


class ReviewDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        text_col: str = "clean_text",
        label_col: str = "label_id",
    ):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_col = text_col
        self.label_col = label_col

        for col in [text_col, label_col]:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found. Available: {list(self.data.columns)}")

        logger.info(f"Dataset created — {len(self.data)} samples | max_length={max_length}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        text = str(self.data.loc[idx, self.text_col])
        label = int(self.data.loc[idx, self.label_col])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",    
            truncation=True,         
            return_tensors="pt",     
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),      
            "attention_mask": encoding["attention_mask"].squeeze(0),   
            "label":          torch.tensor(label, dtype=torch.long),   
        }


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    batch_size: int = 32,
    max_length: int = 256,
    num_workers: int = 0,
) -> tuple:
    
    train_dataset = ReviewDataset(train_df, tokenizer, max_length)
    val_dataset   = ReviewDataset(val_df,   tokenizer, max_length)
    test_dataset  = ReviewDataset(test_df,  tokenizer, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          
        num_workers=num_workers,
        pin_memory=True,      
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,         
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        f"DataLoaders ready — "
        f"Train: {len(train_dataset)} | "
        f"Val: {len(val_dataset)} | "
        f"Test: {len(test_dataset)} | "
        f"Batch size: {batch_size}"
    )
    return train_loader, val_loader, test_loader