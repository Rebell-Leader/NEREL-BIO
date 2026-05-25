"""
BioNNE-R relation extraction model.

Architecture:
    HuggingFace transformer backbone
        → hidden states at <H:TYPE> and <T:TYPE> start-marker positions
        → concat([h_repr, t_repr, nesting_flag])          (optional flag)
        → dropout → Linear → num_classes

The start-marker representation follows Zhong & Chen (2021) "An Improved
Baseline for Sentence-level Relation Extraction" (AACL-IJCNLP 2022).
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class BioNNEModel(nn.Module):
    """Typed entity marker relation extraction model.

    Args:
        model_name:       HuggingFace model id or local path.
        num_classes:      Total number of relation classes (incl. no_relation).
        use_nesting_flag: If True, concatenate binary nesting feature to
                          the entity representation before the classifier.
        dropout:          Dropout probability applied before the linear head.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        use_nesting_flag: bool = True,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.use_nesting_flag = use_nesting_flag

        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        hidden_size: int = config.hidden_size

        # [h_repr ; t_repr] + optional scalar nesting flag
        head_input_dim = hidden_size * 2 + (1 if use_nesting_flag else 0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(head_input_dim, num_classes)

    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,       # [B, L]
        attention_mask: torch.Tensor,  # [B, L]
        h_pos: torch.Tensor,           # [B]  – token index of <H:TYPE>
        t_pos: torch.Tensor,           # [B]  – token index of <T:TYPE>
        nesting_flag: torch.Tensor | None = None,  # [B]  – 0.0 or 1.0
    ) -> torch.Tensor:
        """Return logits of shape [B, num_classes]."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq_output = outputs.last_hidden_state  # [B, L, H]

        batch_idx = torch.arange(seq_output.size(0), device=seq_output.device)
        h_repr = seq_output[batch_idx, h_pos]   # [B, H]
        t_repr = seq_output[batch_idx, t_pos]   # [B, H]

        if self.use_nesting_flag and nesting_flag is not None:
            combined = torch.cat(
                [h_repr, t_repr, nesting_flag.unsqueeze(-1)], dim=-1
            )  # [B, 2H+1]
        else:
            combined = torch.cat([h_repr, t_repr], dim=-1)  # [B, 2H]

        logits = self.classifier(self.dropout(combined))
        return logits

    # ------------------------------------------------------------------

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        """Extend the embedding matrix after adding special tokens."""
        self.encoder.resize_token_embeddings(new_num_tokens)
