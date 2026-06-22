import torch
import torch.nn as nn
from transformers import AutoModel
from loguru import logger


class SentimentClassifier(nn.Module):

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_classes: int = 3,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        self.bert = AutoModel.from_pretrained(model_name)

        self.dropout = nn.Dropout(dropout_rate)

        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

        logger.info(
            f"SentimentClassifier loaded — "
            f"model: {model_name} | "
            f"classes: {num_classes} | "
            f"dropout: {dropout_rate}"
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_output = outputs.last_hidden_state[:, 0, :]

        cls_output = self.dropout(cls_output)

        logits = self.classifier(cls_output)

        return logits

    def get_param_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def load_model(checkpoint_path: str, device: torch.device) -> SentimentClassifier:
    model = SentimentClassifier()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info(f"Model loaded from {checkpoint_path}")
    return model