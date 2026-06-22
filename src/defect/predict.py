import io
import base64
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.models as tvm
from loguru import logger

from src.defect.gradcam import GradCAMDefectVisualizer
from src.utils.logger import setup_logger
from src.utils.metrics import evaluate_defect


LABEL_MAP = {0: "normal", 1: "defective"}


class DefectPredictor:
    
    def __init__(
        self,
        model_path: str = "models/defect_resnet_best.pt",
        confidence_threshold: float = 0.5,
        device: torch.device = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Fine-tuned model not found at {model_path}.\n"
                "Run: python src/defect/train_resnet.py first."
            )

        logger.info(f"Loading fine-tuned ResNet from {model_path}...")

        self.model = tvm.resnet50(weights=None)
        self.model.fc = nn.Linear(2048, 2)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        self.gradcam = GradCAMDefectVisualizer(model_path=model_path)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225],
            ),
        ])

        logger.success(f"DefectPredictor ready on {self.device}")

    def _load_image(self, image: Union[str, Path, Image.Image]) -> tuple:
    
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")

        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        return pil_image, tensor

    def extract_features(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        
        _, tensor = self._load_image(image)
        with torch.no_grad():
            x = self.model.conv1(tensor)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            features = torch.flatten(x, 1)   # (1, 2048)
        return features.squeeze(0).cpu().numpy()

    def compare_with_reference(
        self,
        reference_image: Union[str, Path, Image.Image],
        test_image:       Union[str, Path, Image.Image],
        threshold: float  = 0.82,
        generate_heatmap: bool = True,
    ) -> dict:
        
        ref_feats  = self.extract_features(reference_image)
        test_feats = self.extract_features(test_image)

        similarity = float(
            np.dot(ref_feats, test_feats) /
            (np.linalg.norm(ref_feats) * np.linalg.norm(test_feats) + 1e-8)
        )

        is_defective = similarity < threshold
        label        = "defective" if is_defective else "normal"

        distance_from_threshold = abs(similarity - threshold)
        confidence = min(distance_from_threshold / threshold, 1.0)

        result = {
            "label":      label,
            "confidence": round(confidence, 4),
            "similarity": round(similarity, 4),
            "uncertain":  distance_from_threshold < 0.05,
            "method":     "reference_comparison",
            "scores": {
                "normal":    round(similarity, 4),
                "defective": round(1.0 - similarity, 4),
            },
        }

        if generate_heatmap:
            try:
                pil_test, _ = self._load_image(test_image)
                heatmap_result = self.gradcam.generate_heatmap(pil_test, target_class=1)
                overlay = heatmap_result["overlay"]
                buf = io.BytesIO()
                overlay.save(buf, format="PNG")
                result["overlay_b64"] = base64.b64encode(buf.getvalue()).decode("utf-8")
                result["overlay_pil"] = overlay
            except Exception as e:
                logger.warning(f"GradCAM failed for reference comparison: {e}")

        logger.info(
            f"Reference comparison: similarity={similarity:.3f} "
            f"(threshold={threshold}) → {label.upper()} "
            f"(confidence={confidence:.1%})"
        )
        return result

    def predict(
        self,
        image: Union[str, Path, Image.Image],
        generate_heatmap: bool = True,
        save_path: str = None,
        true_label: str = None,
    ) -> dict:
       
        pil_image, tensor = self._load_image(image)

        # ── Classification ────────────────────────────────────
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze()

        pred_id    = probs.argmax().item()
        confidence = probs[pred_id].item()
        label      = LABEL_MAP[pred_id]

        result = {
            "label":      label,
            "confidence": round(confidence, 4),
            "uncertain":  confidence < self.confidence_threshold,
            "scores": {
                "normal":    round(probs[0].item(), 4),
                "defective": round(probs[1].item(), 4),
            },
        }

        # ── GradCAM visualization ─────────────────────────────
        if generate_heatmap:
            target_class   = 1
            heatmap_result = self.gradcam.generate_heatmap(
                pil_image, target_class=target_class
            )
            overlay = heatmap_result["overlay"]

            buffer = io.BytesIO()
            overlay.save(buffer, format="PNG")
            result["overlay_b64"] = base64.b64encode(
                buffer.getvalue()
            ).decode("utf-8")
            result["overlay_pil"] = overlay

            if save_path:
                self.gradcam.save_visualization(
                    pil_image,
                    save_path,
                    true_label=true_label or label,
                    target_class=target_class,
                )

        return result

    def predict_batch(
        self,
        images: list,
        generate_heatmap: bool = False,
    ) -> list[dict]:
        
        return [
            self.predict(img, generate_heatmap=generate_heatmap)
            for img in images
        ]

    def evaluate(self, metadata_csv: str) -> dict:
       
        import pandas as pd
        from sklearn.metrics import classification_report

        df      = pd.read_csv(metadata_csv)
        test_df = df[df["split"] == "test"].reset_index(drop=True)

        logger.info(f"Evaluating on {len(test_df)} test images...")

        y_true, y_scores = [], []
        for _, row in test_df.iterrows():
            result = self.predict(row["image_path"], generate_heatmap=False)
            y_true.append(int(row["label"]))
            y_scores.append(result["scores"]["defective"])

        y_pred = [1 if s >= 0.5 else 0 for s in y_scores]

        metrics = evaluate_defect(y_true, y_scores)

        logger.info(
            f"\n{classification_report(y_true, y_pred, target_names=['normal','defective'])}"
        )
        logger.info(
            f"Results — Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"AUROC: {metrics['auroc']:.4f}"
        )
        return metrics

    def generate_sample_heatmaps(
        self,
        metadata_csv: str,
        output_dir: str = "outputs/gradcam_samples",
        n_per_defect: int = 2,
    ) -> None:
        
        import pandas as pd
        df = pd.read_csv(metadata_csv)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for category in df["category"].unique():
            normal_samples = df[
                (df["category"] == category) &
                (df["defect_type"] == "good") &
                (df["split"] == "test")
            ].head(1)

            for _, row in normal_samples.iterrows():
                save_path = (
                    f"{output_dir}/{category}_normal_"
                    f"{Path(row['image_path']).stem}.png"
                )
                self.predict(
                    row["image_path"],
                    generate_heatmap=True,
                    save_path=save_path,
                    true_label="normal",
                )
                logger.info(f"Saved: {save_path}")

        defect_df = df[
            (df["label"] == 1) & (df["split"] == "test")
        ]
        for _, group in defect_df.groupby(["category", "defect_type"]):
            for _, row in group.head(n_per_defect).iterrows():
                save_path = (
                    f"{output_dir}/{row['category']}_{row['defect_type']}_"
                    f"{Path(row['image_path']).stem}.png"
                )
                self.predict(
                    row["image_path"],
                    generate_heatmap=True,
                    save_path=save_path,
                    true_label="defective",
                )
                logger.info(f"Saved: {save_path}")

        logger.success(f"All GradCAM samples saved -> {output_dir}/")


# ── Run directly for full evaluation ─────────────────────────
if __name__ == "__main__":
    setup_logger()

    meta_path = Path("data/processed/mvtec_metadata.csv")
    if not meta_path.exists():
        logger.error("Run preprocess.py first.")
        exit(1)

    predictor = DefectPredictor()

    print("\n" + "="*55)
    print("Full Dataset Evaluation — Fine-tuned ResNet-50")
    print("="*55)
    metrics = predictor.evaluate(str(meta_path))
    print(f"\nPrecision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"AUROC:     {metrics['auroc']:.4f}")

    print("\nGenerating GradCAM heatmap samples...")
    predictor.generate_sample_heatmaps(
        str(meta_path),
        output_dir="outputs/gradcam_samples",
        n_per_defect=2,
    )
    print("Done. Check outputs/gradcam_samples/")