from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F


def symmetric_multi_positive_infonce(
    query: torch.Tensor,
    gallery: torch.Tensor,
    query_code: torch.Tensor,
    gallery_code: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE with one or more positives per query."""
    if query.shape[0] == 0 or gallery.shape[0] == 0:
        return query.new_tensor(0.0)

    logits = query @ gallery.T / float(temperature)
    positives = query_code.view(-1, 1).eq(gallery_code.view(1, -1))
    losses: List[torch.Tensor] = []

    row_has_positive = positives.any(dim=1)
    if bool(row_has_positive.any()):
        target = positives[row_has_positive].float()
        target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0)
        losses.append(-(target * F.log_softmax(logits[row_has_positive], dim=1)).sum(dim=1).mean())

    col_has_positive = positives.any(dim=0)
    if bool(col_has_positive.any()):
        target_t = positives[:, col_has_positive].T.float()
        target_t = target_t / target_t.sum(dim=1, keepdim=True).clamp_min(1.0)
        logits_t = logits[:, col_has_positive].T
        losses.append(-(target_t * F.log_softmax(logits_t, dim=1)).sum(dim=1).mean())

    return torch.stack(losses).mean() if losses else query.new_tensor(0.0)


def cgp_align_objective(
    z_compound: torch.Tensor,
    z_compound_profile: torch.Tensor,
    compound_code: torch.Tensor,
    compound_profile_code: torch.Tensor,
    z_orf: torch.Tensor,
    z_orf_profile: torch.Tensor,
    orf_code: torch.Tensor,
    orf_profile_code: torch.Tensor,
    z_crispr: torch.Tensor,
    z_crispr_profile: torch.Tensor,
    crispr_code: torch.Tensor,
    crispr_profile_code: torch.Tensor,
    temperature: float = 0.07,
) -> Dict[str, torch.Tensor]:
    """Return branch losses and the total CGP-Align contrastive objective."""
    losses = {
        "compound_profile": symmetric_multi_positive_infonce(
            z_compound, z_compound_profile, compound_code, compound_profile_code, temperature
        ),
        "orf_profile": symmetric_multi_positive_infonce(z_orf, z_orf_profile, orf_code, orf_profile_code, temperature),
        "crispr_profile": symmetric_multi_positive_infonce(
            z_crispr, z_crispr_profile, crispr_code, crispr_profile_code, temperature
        ),
    }
    losses["total"] = losses["compound_profile"] + losses["orf_profile"] + losses["crispr_profile"]
    return losses
