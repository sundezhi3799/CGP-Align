from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np
import torch

import train_cgp_align_replicate as cgp
from cgp_align.checkpoint_paths import add_data_arguments, configure_inference
from eval_cgp_align_crossmodal_bridge import namespace_from_config
from eval_cgp_align_target_enrichment_replicate import parse_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate intrinsic branch retrieval from CGP-Align checkpoints.")
    p.add_argument("--rows", required=True, help="Comma-separated label:checkpoint entries.")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--negative_ratios", default="100")
    p.add_argument("--num_repeats", type=int, default=5)
    p.add_argument("--max_compounds", type=int, default=2048)
    p.add_argument("--max_genes", type=int, default=2048)
    p.add_argument("--eval_batch_size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--device", default="auto")
    add_data_arguments(p)
    return p.parse_args()


def json_default(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    raise TypeError(f"Unsupported JSON type: {type(x)!r}")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def load_model_and_data(checkpoint: Path, device: torch.device, eval_batch_size: int, data_overrides=None) -> Tuple[cgp.ReplicateCGPAlign, Dict[str, Any], Dict[str, Any], argparse.Namespace]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = namespace_from_config(payload.get("config", {}))
    configure_inference(args, data_overrides)
    args.eval_batch_size = int(eval_batch_size)
    args.eval_max_replicates_per_entity = int(getattr(args, "eval_max_replicates_per_entity", 0))
    args.smoke_test = False
    data = cgp.build_data(args)
    model, _ = cgp.build_model(args, data, device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    data["compound"]["norm"] = payload.get("compound_profile_normalizer", data["compound"]["norm"])
    data["gene"]["norm"] = payload.get("gene_profile_normalizer", data["gene"]["norm"])
    model.eval()
    return model, data, payload, args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = cgp.select_device(str(args.device))
    ratios = cgp.parse_ints(args.negative_ratios)
    rows = parse_rows(args.rows)
    all_metrics: Dict[str, Any] = {}
    flat: List[Dict[str, Any]] = []
    for label, checkpoint in rows:
        model, data, payload, model_args = load_model_and_data(checkpoint, device, args.eval_batch_size, args)
        metrics = cgp.evaluate_all(
            model,
            data,
            model_args,
            device,
            args.split,
            ratios,
            int(args.num_repeats),
            int(args.seed) + 999,
            max_compounds=int(args.max_compounds),
            max_genes=int(args.max_genes),
        )
        top = metrics.get("top10_100", {})
        rec = {
            "row": label,
            "checkpoint": str(checkpoint),
            "epoch": payload.get("epoch"),
            "split": args.split,
            "max_compounds": int(args.max_compounds),
            "max_genes": int(args.max_genes),
            "C2P_Top10_1to100": top.get("compound_to_profile"),
            "P2C_Top10_1to100": top.get("profile_to_compound"),
            "G2P_Top10_1to100": top.get("gene_to_profile"),
            "P2G_Top10_1to100": top.get("profile_to_gene"),
            "Branch_Mean": top.get("mean"),
            "Branch_HMean": top.get("hmean"),
        }
        all_metrics[label] = metrics
        flat.append(rec)
        print(json.dumps(rec, sort_keys=True), flush=True)
    write_json(args.output_dir / "branch_retrieval_metrics.json", all_metrics)
    with (args.output_dir / "branch_retrieval_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({"csv": str(args.output_dir / "branch_retrieval_summary.csv")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
