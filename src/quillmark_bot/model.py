import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from quillmark_bot.config import MODEL_ID


def load_model(model_id: str = MODEL_ID):
    """Load the model with 4-bit nf4 (bitsandbytes) weights on the GPU, plus its processor.

    4-bit generates ~3x faster than 8-bit on the RTX 5050 (14.9 vs 4.6 tok/s) and uses 1.5 GiB less VRAM.
    """
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
