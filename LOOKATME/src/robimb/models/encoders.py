
"""Encoders — user models: atipiqal/BOB, atipiqal/roBERTino (stub)."""
from transformers import AutoModel, AutoTokenizer
def load_backbone(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModel.from_pretrained(name)
    return tok, mdl
