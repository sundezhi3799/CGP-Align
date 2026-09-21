from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from cgp_common import (
    CHECKPOINT_DIR,
    DATA_DIR,
    LOG_DIR,
    PROTEIN_EMB_DIR,
    CGPAlignModel,
    CGPBundle,
    append_jsonl,
    bidirectional_infonce,
    ensure_output_dirs,
    long_tensor,
    metrics_from_ranks,
    retrieval_ranks_single_positive,
    set_seed,
    tensor_from_numpy,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CGP-Align-Base on intrinsic entity train split only.")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--protein_embedding_dir", type=Path, default=PROTEIN_EMB_DIR)
    parser.add_argument("--intrinsic_split_path", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--log_dir", type=Path, default=LOG_DIR)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--steps_per_epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda_compound_profile", type=float, default=1.0)
    parser.add_argument("--lambda_protein_profile", type=float, default=1.0)
    parser.add_argument("--selection_metric", default="val_base_alignment_score")
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--val_max_queries", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def read_split(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def encode_profile_rows(model: CGPAlignModel, bundle: CGPBundle, rows: np.ndarray, device: torch.device) -> torch.Tensor:
    return model.encode_profile(
        tensor_from_numpy(bundle.profile_features_for_rows(rows), device),
        long_tensor(bundle.source_ids_for_rows(rows), device),
    )


def quick_val_metrics(
    model: CGPAlignModel,
    bundle: CGPBundle,
    split: Dict[str, Any],
    device: torch.device,
    rng: np.random.Generator,
    max_queries: int,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    model.eval()
    with torch.no_grad():
        c_val = np.asarray(split["compound"]["val"], dtype=np.int64)
        if len(c_val) > max_queries:
            c_val = rng.choice(c_val, size=max_queries, replace=False)
        if len(c_val) >= 2:
            cp_rows = bundle.profile_rows_for_compounds(c_val)
            z_c = model.encode_compound(tensor_from_numpy(bundle.structure_features[c_val], device)).detach().cpu().numpy()
            z_cp = encode_profile_rows(model, bundle, cp_rows, device).detach().cpu().numpy()
            ranks, gaps = retrieval_ranks_single_positive(z_c, z_cp, np.arange(len(c_val), dtype=np.int64))
            metrics["val_compound_to_profile_Recall@10"] = metrics_from_ranks(ranks, gaps)["Recall@10"]

        g_val = np.asarray(split["gene"]["val"], dtype=np.int64)
        if len(g_val) > max_queries:
            g_val = rng.choice(g_val, size=max_queries, replace=False)
        if len(g_val) >= 2:
            gp_rows = bundle.profile_rows_for_genes(g_val)
            z_p = model.encode_protein(tensor_from_numpy(bundle.protein_embeddings[g_val], device)).detach().cpu().numpy()
            z_gp = encode_profile_rows(model, bundle, gp_rows, device).detach().cpu().numpy()
            ranks, gaps = retrieval_ranks_single_positive(z_p, z_gp, np.arange(len(g_val), dtype=np.int64))
            metrics["val_protein_to_profile_Recall@10"] = metrics_from_ranks(ranks, gaps)["Recall@10"]

    vals = [metrics.get("val_compound_to_profile_Recall@10"), metrics.get("val_protein_to_profile_Recall@10")]
    vals = [float(v) for v in vals if v is not None]
    if vals:
        metrics["val_base_alignment_score"] = float(np.mean(vals))
    return metrics


def save_checkpoint(path: Path, model: CGPAlignModel, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**payload, "model_state_dict": model.state_dict()}, path)


def main() -> None:
    args = parse_args()
    ensure_output_dirs(args.checkpoint_dir.parent)
    set_seed(args.seed)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    bundle = CGPBundle(args.data_dir, args.protein_embedding_dir)
    split = read_split(args.intrinsic_split_path)
    train_compounds = np.asarray(split["compound"]["train"], dtype=np.int64)
    train_genes = np.asarray(split["gene"]["train"], dtype=np.int64)
    if args.smoke_test:
        train_compounds = train_compounds[: min(len(train_compounds), 2048)]
        train_genes = train_genes[: min(len(train_genes), 2048)]
        args.epochs = min(args.epochs, 2)
        args.val_max_queries = min(args.val_max_queries, 256)

    model = CGPAlignModel(
        bundle.structure_dim,
        bundle.protein_dim,
        bundle.profile_dim,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        feature_dim_by_source=bundle.feature_dim_by_source,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ckpt_dir = args.checkpoint_dir / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"{args.run_name}_train_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    config = vars(args).copy()
    config.update(
        {
            "mode": "cgp_base_intrinsic_entity_split",
            "structure_dim": bundle.structure_dim,
            "protein_dim": bundle.protein_dim,
            "profile_dim": bundle.profile_dim,
            "feature_dim_by_source": bundle.feature_dim_by_source,
            "intrinsic_split_summary": split.get("summary"),
            "uses_target_or_link_labels_for_training": False,
        }
    )
    write_json(ckpt_dir / "config.json", config)

    rng = np.random.default_rng(args.seed)
    steps_per_epoch = args.steps_per_epoch or int(np.ceil(max(len(train_compounds), len(train_genes)) / max(1, args.batch_size)))
    best_score = -float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        component_values: Dict[str, List[float]] = {"compound_profile": [], "protein_profile": [], "total": []}
        for _ in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            loss = next(model.parameters()).new_tensor(0.0)
            compound_loss = next(model.parameters()).new_tensor(0.0)
            protein_loss = next(model.parameters()).new_tensor(0.0)
            if args.lambda_compound_profile > 0 and len(train_compounds) >= 2:
                c_sel = rng.choice(train_compounds, size=min(args.batch_size, len(train_compounds)), replace=False)
                cp_rows = bundle.profile_rows_for_compounds(c_sel)
                z_c = model.encode_compound(tensor_from_numpy(bundle.structure_features[c_sel], device))
                z_cp = encode_profile_rows(model, bundle, cp_rows, device)
                compound_loss = bidirectional_infonce(z_c, z_cp, args.temperature)
                loss = loss + args.lambda_compound_profile * compound_loss
            if args.lambda_protein_profile > 0 and len(train_genes) >= 2:
                g_sel = rng.choice(train_genes, size=min(args.batch_size, len(train_genes)), replace=False)
                gp_rows = bundle.profile_rows_for_genes(g_sel)
                z_p = model.encode_protein(tensor_from_numpy(bundle.protein_embeddings[g_sel], device))
                z_gp = encode_profile_rows(model, bundle, gp_rows, device)
                protein_loss = bidirectional_infonce(z_p, z_gp, args.temperature)
                loss = loss + args.lambda_protein_profile * protein_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            component_values["compound_profile"].append(float(compound_loss.detach().cpu().item()))
            component_values["protein_profile"].append(float(protein_loss.detach().cpu().item()))
            component_values["total"].append(float(loss.detach().cpu().item()))

        val = quick_val_metrics(model, bundle, split, device, rng, args.val_max_queries) if epoch % args.val_every == 0 else {}
        record = {
            "epoch": epoch,
            "steps": steps_per_epoch,
            "train_compound_profile": float(np.mean(component_values["compound_profile"])),
            "train_protein_profile": float(np.mean(component_values["protein_profile"])),
            "train_total": float(np.mean(component_values["total"])),
            **val,
        }
        append_jsonl(log_path, record)
        score = val.get(args.selection_metric)
        if score is not None and float(score) > best_score:
            best_score = float(score)
            save_checkpoint(ckpt_dir / "best_model.pt", model, {"epoch": epoch, "best_score": best_score, "selection_metric": args.selection_metric, "config": config})
        print(json.dumps(record))

    if not (ckpt_dir / "best_model.pt").exists():
        save_checkpoint(ckpt_dir / "best_model.pt", model, {"epoch": args.epochs, "best_score": best_score, "config": config})
    print(f"Saved checkpoint to {ckpt_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
