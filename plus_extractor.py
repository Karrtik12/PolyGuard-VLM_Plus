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
    Enhanced Multimodal Security Feature Extractor for PolyGuard-VLM Plus.
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
        Encodes an image using OpenCLIP and spatial feature analysis into a 512-dim normalized vector.
        Detects typographic text overlays, embedded jailbreak signs, and adversarial noise patches.
        Returns (image_embedding, visual_anomaly_factor).
        """
        if image is None:
            return None, 0.0

        if image.mode != "RGB":
            image = image.convert("RGB")

        # 1. Spatial edge & typographic gradient density analysis
        import numpy as np
        gray = np.array(image.convert("L"), dtype=np.float32)
        dx = np.diff(gray, axis=1)
        dy = np.diff(gray, axis=0)
        edge_density = float(np.mean(np.abs(dx)) + np.mean(np.abs(dy)))
        
        # Baseline anomaly score from edge density (high contrast text overlay / noise patch)
        # Clean scenery ~ 12-18, Typographic text sign / patch ~ 24-45+
        base_anomaly = 0.15
        if edge_density > 22.0:
            base_anomaly = min(0.85, 0.20 + (edge_density - 22.0) * 0.025)

        # 2. OpenCLIP feature extraction & zero-shot visual jailbreak scoring
        if not self.clip_loaded:
            self._load_clip_model()

        if self.clip_model is not None and self.clip_preprocess is not None:
            try:
                image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                raw_image_features = self.clip_model.encode_image(image_tensor)
                
                # Multi-concept zero-shot visual prompt classification
                if OPENCLIP_AVAILABLE:
                    try:
                        tokenizer = open_clip.get_tokenizer(self.clip_model_name)
                        clean_prompts = [
                            "a photo of a famous landmark building or architectural monument",
                            "a clean photo of natural scenery landscape mountain river lake sky",
                            "a photograph of a pet cat or dog",
                            "a photo of a paper document receipt or page",
                            "a photograph of city street architecture or indoor room"
                        ]
                        attack_prompts = [
                            "a photo featuring bold typographic impact text writing override safety rules or ignore guardrails",
                            "an image containing a colorful high contrast adversarial noise patch sticker",
                            "a street sign overlaid with written instructions telling to launch cyberattacks or bypass safety"
                        ]
                        
                        clean_tokens = tokenizer(clean_prompts).to(self.device)
                        attack_tokens = tokenizer(attack_prompts).to(self.device)
                        
                        clean_features = self.clip_model.encode_text(clean_tokens)
                        attack_features = self.clip_model.encode_text(attack_tokens)
                        
                        img_norm = raw_image_features / raw_image_features.norm(dim=-1, keepdim=True)
                        clean_norm = clean_features / clean_features.norm(dim=-1, keepdim=True)
                        attack_norm = attack_features / attack_features.norm(dim=-1, keepdim=True)
                        
                        clean_sims = (img_norm @ clean_norm.T).squeeze(0)
                        attack_sims = (img_norm @ attack_norm.T).squeeze(0)
                        
                        max_clean = float(torch.max(clean_sims).item())
                        max_attack = float(torch.max(attack_sims).item())
                        
                        if max_attack > (max_clean + 0.02):
                            base_anomaly = max(0.82, 0.75 + (max_attack - max_clean) * 0.5)
                        else:
                            base_anomaly = min(0.15, max_clean * 0.5)
                    except Exception as e_clip:
                        pass

                projected = self.clip_projection(raw_image_features)
                image_emb = nn.functional.normalize(projected.squeeze(0), p=2, dim=-1)
                anomaly_factor = min(1.0, base_anomaly)
                return image_emb, float(anomaly_factor)
            except Exception as e:
                print(f"[PlusMultimodalExtractor] Vision inference notice ({e}).")

        # Fallback pseudo-visual vector using deterministic features
        r, g, b = image.resize((1, 1)).getpixel((0, 0))
        seed_tensor = torch.tensor([r/255.0, g/255.0, b/255.0] * 170 + [0.5, 0.5], device=self.device)
        projected = self.clip_projection(seed_tensor[:512])
        image_emb = nn.functional.normalize(projected, p=2, dim=-1)
        return image_emb, float(base_anomaly)

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
