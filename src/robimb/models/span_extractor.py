"""Span-based property extraction model.

This model learns to find the relevant text span for extracting properties,
similar to Question Answering models like BERT for SQuAD.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class PropertySpanExtractor(nn.Module):
    """Span extraction model for property values.

    Given a context (product description) and a property query,
    predicts start and end positions of the answer span.
    """

    def __init__(
        self,
        backbone_name: str = "dbmdz/bert-base-italian-xxl-cased",
        num_properties: int = 20,
        dropout: float = 0.1,
        hf_token: Optional[str] = None,
    ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(backbone_name, token=hf_token)
        self.backbone = AutoModel.from_pretrained(backbone_name, token=hf_token)

        hidden_size = self.backbone.config.hidden_size

        # Property embeddings (learnable query representations)
        self.property_embeddings = nn.Embedding(num_properties, hidden_size)

        # Span prediction heads
        self.qa_outputs = nn.Linear(hidden_size, 2)  # start and end logits
        self.dropout = nn.Dropout(dropout)

        self.num_properties = num_properties

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        property_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            property_ids: [batch_size] - which property to extract
            token_type_ids: [batch_size, seq_len] - optional
            start_positions: [batch_size] - ground truth start (for training)
            end_positions: [batch_size] - ground truth end (for training)

        Returns:
            Dict with:
                - start_logits: [batch_size, seq_len]
                - end_logits: [batch_size, seq_len]
                - loss: scalar (if start/end_positions provided)
        """
        # Encode context
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden]
        sequence_output = self.dropout(sequence_output)

        # Get property embeddings
        prop_embeds = self.property_embeddings(property_ids)  # [batch_size, hidden]

        # Condition on property by adding to each token
        prop_embeds_expanded = prop_embeds.unsqueeze(1)  # [batch_size, 1, hidden]
        conditioned = sequence_output + prop_embeds_expanded  # broadcast

        # Predict start and end logits
        logits = self.qa_outputs(conditioned)  # [batch_size, seq_len, 2]
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # [batch_size, seq_len]
        end_logits = end_logits.squeeze(-1)  # [batch_size, seq_len]

        # Mask padding tokens
        start_logits = start_logits.masked_fill(~attention_mask.bool(), -1e9)
        end_logits = end_logits.masked_fill(~attention_mask.bool(), -1e9)

        output = {
            "start_logits": start_logits,
            "end_logits": end_logits,
        }

        # Calculate loss if labels provided
        if start_positions is not None and end_positions is not None:
            # Clamp positions to valid range
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            output["loss"] = total_loss

        return output

    def predict_span(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        property_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> List[Tuple[int, int]]:
        """Predict answer spans.

        Returns:
            List of (start_token_idx, end_token_idx) for each example in batch
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            property_ids=property_ids,
            token_type_ids=token_type_ids,
        )

        start_logits = outputs["start_logits"]  # [batch_size, seq_len]
        end_logits = outputs["end_logits"]  # [batch_size, seq_len]

        batch_size = start_logits.size(0)
        spans = []

        for i in range(batch_size):
            start_idx = start_logits[i].argmax().item()
            end_idx = end_logits[i].argmax().item()

            # Ensure end >= start
            if end_idx < start_idx:
                end_idx = start_idx

            spans.append((start_idx, end_idx))

        return spans


def convert_token_span_to_char_span(
    text: str,
    token_span: Tuple[int, int],
    tokenizer,
    input_ids: torch.Tensor,
) -> Tuple[int, int]:
    """Convert token-level span to character-level span.

    Args:
        text: Original text
        token_span: (start_token_idx, end_token_idx)
        tokenizer: The tokenizer used
        input_ids: Token IDs tensor [seq_len]

    Returns:
        (start_char, end_char) in original text
    """
    start_token, end_token = token_span

    # Get offset mapping if available
    if hasattr(tokenizer, 'convert_tokens_to_string'):
        # Reconstruct text from tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())

        # Find character positions
        # This is approximate - for exact positions, use tokenizer with return_offsets_mapping=True
        text_before = tokenizer.convert_tokens_to_string(tokens[:start_token])
        text_span = tokenizer.convert_tokens_to_string(tokens[start_token:end_token+1])

        start_char = len(text_before)
        end_char = start_char + len(text_span)

        return (start_char, end_char)

    # Fallback
    return (0, len(text))


class PropertyExtractorPipeline:
    """End-to-end pipeline for property extraction."""

    def __init__(
        self,
        model: PropertySpanExtractor,
        tokenizer,
        property_id_map: Dict[str, int],
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.property_id_map = property_id_map
        self.device = device

        # Reverse map
        self.id_to_property = {v: k for k, v in property_id_map.items()}

    def extract(
        self,
        text: str,
        property_ids: List[str],
    ) -> Dict[str, Dict[str, any]]:
        """Extract properties from text.

        Args:
            text: Product description
            property_ids: List of property IDs to extract (e.g., ["marchio", "materiale"])

        Returns:
            Dict mapping property_id to extraction result:
            {
                "marchio": {
                    "raw_text": "Grohe",
                    "span": (47, 52),
                    "confidence": 0.95
                },
                ...
            }
        """
        self.model.eval()
        results = {}

        with torch.no_grad():
            for prop_id in property_ids:
                if prop_id not in self.property_id_map:
                    continue

                prop_idx = self.property_id_map[prop_id]

                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_offsets_mapping=True,  # For char-level span
                )

                offset_mapping = inputs.pop("offset_mapping")[0]
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Predict
                property_tensor = torch.tensor([prop_idx], device=self.device)
                outputs = self.model.forward(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    property_ids=property_tensor,
                )

                start_logits = outputs["start_logits"][0]  # [seq_len]
                end_logits = outputs["end_logits"][0]

                # Get best span
                start_idx = start_logits.argmax().item()
                end_idx = end_logits[start_idx:].argmax().item() + start_idx

                # Calculate confidence (normalized logit scores)
                start_conf = torch.softmax(start_logits, dim=0)[start_idx].item()
                end_conf = torch.softmax(end_logits, dim=0)[end_idx].item()
                confidence = (start_conf + end_conf) / 2

                # Convert to character span
                if start_idx < len(offset_mapping) and end_idx < len(offset_mapping):
                    char_start = offset_mapping[start_idx][0].item()
                    char_end = offset_mapping[end_idx][1].item()
                    raw_text = text[char_start:char_end]
                else:
                    char_start = 0
                    char_end = 0
                    raw_text = ""

                results[prop_id] = {
                    "raw_text": raw_text,
                    "span": (char_start, char_end),
                    "confidence": confidence,
                    "token_span": (start_idx, end_idx),
                }

        return results


__all__ = [
    "PropertySpanExtractor",
    "PropertyExtractorPipeline",
    "convert_token_span_to_char_span",
]
