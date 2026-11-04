"""Load Qwen3.5-4B with 4-bit nf4 weights on the GPU.

Quantizing on load memory-maps the full bf16 checkpoint (~9 GB) first, which can exhaust Windows'
virtual memory when other programs (e.g. Docker's VM) are running. So the 4-bit weights are saved once
to QUANTIZED_MODEL_DIR (~3 GB) and loaded from there afterwards.

Usage: python -m quillmark_bot.model   (one-time: quantize and save the local copy)
"""
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from quillmark_bot.config import MODEL_ID, QUANTIZED_MODEL_DIR


def _quantize(model_id: str):
    """4-bit generates ~3x faster than 8-bit on the RTX 5050 (14.9 vs 4.6 tok/s) and uses 1.5 GiB less VRAM."""
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        ),
        device_map="auto",
        dtype=torch.bfloat16,
    )
    return model, processor


def load_model(model_id: str = MODEL_ID):
    """The 4-bit model and its processor, from the local copy when it exists."""
    if model_id == MODEL_ID and (QUANTIZED_MODEL_DIR / "config.json").exists():
        processor = AutoProcessor.from_pretrained(QUANTIZED_MODEL_DIR)
        # The saved config carries its own quantization settings; passing quantization_config here
        # (even None) would make transformers ignore them
        model = AutoModelForImageTextToText.from_pretrained(QUANTIZED_MODEL_DIR, device_map="auto",
                                                            dtype=torch.bfloat16)
        return model, processor
    return _quantize(model_id)


def save_quantized() -> None:
    model, processor = _quantize(MODEL_ID)
    model.save_pretrained(QUANTIZED_MODEL_DIR)
    processor.save_pretrained(QUANTIZED_MODEL_DIR)
    print(f"Saved the 4-bit model to {QUANTIZED_MODEL_DIR}")


if __name__ == "__main__":
    save_quantized()
