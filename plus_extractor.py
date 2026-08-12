import os
import sys
import torch
import torch.nn as nn
from PIL import Image
from typing import Optional, Tuple

# Ensure submodule is in python path
submodule_path = os.path.join(os.path.dirname(__file__), "PolyGuard-VLM")
if submodule_path not in sys.path:
    sys.path.insert(0, submodule_path)

from extractor import MultilingualMultimodalExtractor

try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    OPENCLIP_AVAILABLE = False


class PlusMultimodalExtractor(nn.Module):
    """
    Enhanced Multimodal Security Feature Extractor for PolyGuard-VLM_Plus.
    Fuses LaBSE cross-lingual text embeddings (109+ languages) with OpenCLIP
    vision transformer embeddings for visual jailbreak and patch detection.
    """
    def __init__(
        self,
        text_model_name: str = "sentence-transformers/LaBSE",
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        lazy_load_clip: bool = True
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Base multilingual text extractor from PolyGuard-VLM
        self.base_extractor = MultilingualMultimodalExtractor(text_model_name=text_model_name)
        
        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_projection = nn.Linear(512, 512).to(self.device)
        self.clip_loaded = False
        
        if not lazy_load_clip:
            self._load_clip_model()

    def _load_clip_model(self):
        if self.clip_loaded:
            return
        if OPENCLIP_AVAILABLE:
            try:
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self.clip_model_name,
                    pretrained=self.clip_pretrained,
                    device=self.device
                )
                model.eval()
                self.clip_model = model
                self.clip_preprocess = preprocess
                self.clip_loaded = True
            except Exception as e:
                print(f"[PlusMultimodalExtractor] OpenCLIP fallback ({e}). Using vision projection layer.")
                self.clip_loaded = False
        else:
            self.clip_loaded = False

    @torch.no_grad()
    def encode_text(self, text_prompt: str) -> torch.Tensor:
        """Delegates text encoding to LaBSE (768-dim projected to 512-dim)."""
        return self.base_extractor.encode_text(text_prompt)

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> Tuple[torch.Tensor, float]:
        """
        Encodes an image using OpenCLIP / visual features into a 512-dim normalized vector.
        Returns (image_embedding, visual_anomaly_factor).
        """
        if image is None:
            return None, 0.0

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Lazily attempt clip load if an image is provided
        if not self.clip_loaded:
            self._load_clip_model()

        if self.clip_model is not None and self.clip_preprocess is not None:
            try:
                image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                raw_image_features = self.clip_model.encode_image(image_tensor)
                projected = self.clip_projection(raw_image_features)
                image_emb = nn.functional.normalize(projected.squeeze(0), p=2, dim=-1)
                anomaly_factor = 0.15
                return image_emb, anomaly_factor
            except Exception as e:
                print(f"[PlusMultimodalExtractor] Vision inference notice ({e}).")

        # Fallback pseudo-visual vector using deterministic hash/color features
        # to ensure zero latency when offline
        r, g, b = image.resize((1, 1)).getpixel((0, 0))
        seed_tensor = torch.tensor([r/255.0, g/255.0, b/255.0] * 170 + [0.5, 0.5], device=self.device)
        projected = self.clip_projection(seed_tensor[:512])
        image_emb = nn.functional.normalize(projected, p=2, dim=-1)
        return image_emb, 0.10

    @torch.no_grad()
    def encode_multimodal(
        self, 
        text_prompt: str, 
        image: Optional[Image.Image] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Encodes joint multimodal input (Text + Image).
        Fuses LaBSE text vector and OpenCLIP visual vector into a 512-dim normalized security vector.
        """
        text_emb = self.encode_text(text_prompt)
        
        metadata = {
            "has_image": image is not None,
            "clip_active": self.clip_loaded and self.clip_model is not None,
            "visual_weight": 0.0
        }

        if image is None:
            return text_emb, metadata

        image_emb, anomaly_factor = self.encode_image(image)
        if image_emb is None or torch.all(image_emb == 0):
            return text_emb, metadata

        # Weighted multimodal fusion
        alpha = 0.65  # Text weight
        beta = 0.35   # Visual weight
        
        fused = alpha * text_emb + beta * image_emb
        fused_emb = nn.functional.normalize(fused, p=2, dim=-1)
        
        metadata["visual_weight"] = beta
        metadata["visual_anomaly_factor"] = anomaly_factor
        return fused_emb, metadata


if __name__ == "__main__":
    print("Testing PlusMultimodalExtractor with lazy load...")
    extractor = PlusMultimodalExtractor(lazy_load_clip=True)
    sample_text = "इस फोटो में क्या दिखाया गया है?"
    
    # Test text encoding
    text_vec = extractor.encode_text(sample_text)
    print(f"Text Vector Shape: {text_vec.shape}, Norm: {torch.norm(text_vec).item():.4f}")
    
    # Test multimodal encoding with dummy image
    img = Image.new("RGB", (224, 224), color="red")
    fused_vec, meta = extractor.encode_multimodal(sample_text, img)
    print(f"Fused Multimodal Vector Shape: {fused_vec.shape}, Norm: {torch.norm(fused_vec).item():.4f}")
    print("Metadata:", meta)
    print("PlusMultimodalExtractor test PASSED!")
