import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from quillmark_bot.config import MODEL_ID


def load_model(model_id: str = MODEL_ID):
    """Load the model with 8-bit (bitsandbytes) weights on the GPU, plus its processor."""
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        dtype=torch.bfloat16,
    )
    return model, processor
