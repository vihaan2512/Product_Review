from pathlib import Path
from typing import Union

import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as tvm
import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from loguru import logger

from src.utils.logger import setup_logger


class GradCAMDefectVisualizer:

    def __init__(self, model_path: str = None):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Linear(2048, 2)

        if model_path and Path(model_path).exists():
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            logger.info(f"GradCAM loaded fine-tuned weights from {model_path}")
        else:
            logger.warning(
                "No fine-tuned weights found — using ImageNet pretrained.\n"
                "Run train_resnet.py first for accurate heatmaps."
            )

        self.model.to(self.device)
        self.model.eval()

        self.gradients   = None
        self.activations = None
        self._register_hooks()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225],
            ),
        ])

        logger.success(f"GradCAM visualizer ready on {self.device}")

    def _register_hooks(self):

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer = self.model.layer4[-1]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def _preprocess(self, image: Union[str, Path, Image.Image]) -> tuple:

        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")

        pil_resized = pil_image.resize((224, 224))
        tensor      = self.transform(pil_image).unsqueeze(0).to(self.device)
        return pil_resized, tensor

    def _compute_gradcam(
        self,
        tensor: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
       
        self.model.zero_grad()

        output = self.model(tensor)

        output[0, target_class].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()   # (7, 7)

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        cam = cv2.resize(cam, (224, 224))

        return cam

    def generate_heatmap(
        self,
        image: Union[str, Path, Image.Image],
        target_class: int = 1,
        alpha: float = 0.5,
    ) -> dict:
      
        pil_resized, tensor = self._preprocess(image)

        with torch.no_grad():
            logits     = self.model(tensor.clone())
            probs      = torch.softmax(logits, dim=1).squeeze()
            pred_class = probs.argmax().item()
            confidence = probs[pred_class].item()

        tensor_grad = self._preprocess(image)[1]  
        cam = self._compute_gradcam(tensor_grad, target_class)

        heatmap_rgb     = cm.jet(cam)[:, :, :3]                  
        heatmap_rgb     = (heatmap_rgb * 255).astype(np.uint8)
        heatmap_pil     = Image.fromarray(heatmap_rgb)

        original_arr    = np.array(pil_resized, dtype=np.float32)
        heatmap_arr     = heatmap_rgb.astype(np.float32)
        overlay_arr     = (1 - alpha) * original_arr + alpha * heatmap_arr
        overlay_arr     = np.clip(overlay_arr, 0, 255).astype(np.uint8)
        overlay_pil     = Image.fromarray(overlay_arr)

        return {
            "original":   pil_resized,
            "heatmap":    heatmap_pil,
            "overlay":    overlay_pil,
            "cam":        cam,
            "pred_label": "defective" if pred_class == 1 else "normal",
            "confidence": round(confidence, 4),
        }

    def save_visualization(
        self,
        image: Union[str, Path, Image.Image],
        save_path: str,
        true_label: str = None,
        target_class: int = 1,
    ) -> None:
    
        result = self.generate_heatmap(image, target_class=target_class)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor("white")

        axes[0].imshow(result["original"])
        axes[0].set_title("Original Image", fontsize=13, fontweight="bold")
        axes[0].axis("off")
        if true_label:
            color = "green" if true_label == "normal" else "red"
            axes[0].set_xlabel(
                f"True: {true_label.upper()}",
                fontsize=11, color=color, fontweight="bold"
            )

        axes[1].imshow(result["heatmap"])
        axes[1].set_title("GradCAM Heatmap", fontsize=13, fontweight="bold")
        axes[1].axis("off")
        axes[1].set_xlabel(
            "Red = defect region | Blue = background",
            fontsize=10, color="gray"
        )

        pred  = result["pred_label"]
        conf  = result["confidence"]
        color = "red" if pred == "defective" else "green"
        axes[2].imshow(result["overlay"])
        axes[2].set_title("GradCAM Overlay", fontsize=13, fontweight="bold")
        axes[2].axis("off")
        axes[2].set_xlabel(
            f"Predicted: {pred.upper()} ({conf:.1%})",
            fontsize=11, color=color, fontweight="bold"
        )

        plt.suptitle(
            "Defect Detection — GradCAM Visualization (ResNet-50)",
            fontsize=14, fontweight="bold", y=1.02
        )
        plt.tight_layout()
        plt.savefig(
            save_path, dpi=150, bbox_inches="tight",
            facecolor="white", edgecolor="none"
        )
        plt.close()
        logger.success(f"Visualization saved -> {save_path}")

    def batch_visualize(
        self,
        image_paths: list,
        save_dir: str,
        true_labels: list = None,
    ) -> None:

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, img_path in enumerate(image_paths):
            true_label = true_labels[i] if true_labels else None
            save_path  = save_dir / f"gradcam_{Path(str(img_path)).stem}.png"
            self.save_visualization(img_path, str(save_path), true_label=true_label)
            logger.info(f"[{i+1}/{len(image_paths)}] Saved: {save_path.name}")


if __name__ == "__main__":
    setup_logger()

    model_path = "models/defect_resnet_best.pt"
    mvtec_dir  = Path("data/raw/mvtec/bottle")

    if not mvtec_dir.exists():
        logger.error("MVTec bottle data not found at data/raw/mvtec/bottle/")
        exit(1)

    visualizer = GradCAMDefectVisualizer(model_path=model_path)
    Path("outputs").mkdir(exist_ok=True)

    test_cases = []

    good_dir = mvtec_dir / "test/good"
    if good_dir.exists():
        good_imgs = list(good_dir.glob("*.png"))
        if good_imgs:
            test_cases.append((good_imgs[0], "normal"))

    test_dir = mvtec_dir / "test"
    if test_dir.exists():
        defect_dirs = [d for d in test_dir.iterdir() if d.is_dir() and d.name != "good"]
        for d in defect_dirs:
            defect_imgs = list(d.glob("*.png"))
            if defect_imgs:
                test_cases.append((defect_imgs[0], "defective"))
                if len(test_cases) >= 4:
                    break

    print("\n" + "="*55)
    print("GradCAM Visualization — Fine-tuned ResNet-50")
    print("="*55)

    for img_path, true_label in test_cases:
        save_path = f"outputs/gradcam_{img_path.parent.name}_{img_path.stem}.png"
        visualizer.save_visualization(img_path, save_path, true_label=true_label)
        result = visualizer.generate_heatmap(img_path)
        print(
            f"\n{img_path.parent.name}/{img_path.name}"
            f" -> {result['pred_label'].upper()} ({result['confidence']:.1%})"
            f" | Saved: {save_path}"
        )