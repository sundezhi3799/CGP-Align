from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd

from cgp_common import DATA_DIR, write_json


ESM2_ALIASES = {
    "esm2_35m": "esm2_t12_35M_UR50D",
    "esm2_150m": "esm2_t30_150M_UR50D",
    "esm2_650m": "esm2_t33_650M_UR50D",
}
PROTT5_ALIASES = {
    "prott5_xl_half": "Rostlab/prot_t5_xl_half_uniref50-enc",
    "prott5_xl": "Rostlab/prot_t5_xl_uniref50",
}
ANKH_ALIASES = {
    "ankh_base": "ElnaggarLab/ankh-base",
    "ankh_large": "ElnaggarLab/ankh-large",
}
SAPROT_ALIASES = {
    "saprot_650m_af2": "westlake-repl/SaProt_650M_AF2",
    "saprot_650m_unknown": "westlake-repl/SaProt_650M_AF2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute frozen protein embeddings for encoder ablation.")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "One of esm2_35m, esm2_150m, esm2_650m, prott5_xl_half, prott5_xl, "
            "ankh_base, ankh_large, saprot_650m_af2, saprot_650m_unknown, "
            "or a concrete model name supported by the corresponding backend."
        ),
    )
    parser.add_argument("--backend", choices=["auto", "esm2", "prott5", "ankh", "saprot", "esmc", "aa_kmer"], default="auto")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=1022)
    parser.add_argument("--window_stride", type=int, default=768)
    parser.add_argument("--max_windows_per_sequence", type=int, default=16)
    parser.add_argument("--long_strategy", choices=["truncate", "sliding"], default="sliding")
    parser.add_argument(
        "--esm_layers",
        default="",
        help="Comma-separated ESM representation layers. Empty uses the final layer inferred from model name. Multiple layers are concatenated.",
    )
    parser.add_argument(
        "--esm_pooling",
        choices=["mean", "cls", "mean_cls", "mean_max", "mean_cls_max"],
        default="mean",
        help="Residue pooling for ESM backends. Multiple pooling outputs are concatenated.",
    )
    parser.add_argument(
        "--window_aggregation",
        choices=["mean", "length_weighted"],
        default="mean",
        help="How to aggregate sliding-window embeddings for long proteins.",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def resolve_model(model: str) -> str:
    return ESM2_ALIASES.get(model, PROTT5_ALIASES.get(model, ANKH_ALIASES.get(model, SAPROT_ALIASES.get(model, model))))


def infer_backend(model: str, backend: str) -> str:
    if backend != "auto":
        return backend
    concrete = resolve_model(model).lower()
    if model in ESM2_ALIASES or concrete.startswith("esm2_"):
        return "esm2"
    if model in PROTT5_ALIASES or "prot_t5" in concrete or "prott5" in concrete:
        return "prott5"
    if model in ANKH_ALIASES or "ankh" in concrete:
        return "ankh"
    if model in SAPROT_ALIASES or "saprot" in concrete:
        return "saprot"
    if model.startswith("esmc") or "esmc" in concrete:
        return "esmc"
    if model in {"aa_kmer", "amino_acid_kmer"}:
        return "aa_kmer"
    raise ValueError(f"Cannot infer backend for model={model}; pass --backend explicitly.")


def clean_protein_sequence(seq: Any) -> str:
    seq = str(seq or "").strip().upper()
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWYBXZUO]", "X", seq)


def make_windows(
    seq: str,
    max_len: int,
    stride: int,
    strategy: str,
    max_windows: int,
) -> List[str]:
    seq = clean_protein_sequence(seq)
    if not seq:
        return []
    if len(seq) <= max_len:
        return [seq]
    if strategy == "truncate":
        return [seq[:max_len]]
    stride = max(1, int(stride))
    last_start = max(0, len(seq) - max_len)
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    if max_windows > 0 and len(starts) > max_windows:
        keep = np.linspace(0, len(starts) - 1, num=max_windows)
        starts = [starts[int(round(x))] for x in keep]
        starts = sorted(set(starts))
    return [seq[s : s + max_len] for s in starts]


def iter_window_batches(
    sequences: Sequence[str],
    batch_size: int,
    max_seq_len: int,
    stride: int,
    strategy: str,
    max_windows: int,
) -> Iterable[Tuple[List[int], List[str]]]:
    parents: List[int] = []
    windows: List[str] = []
    for i, seq in enumerate(sequences):
        for window in make_windows(seq, max_seq_len, stride, strategy, max_windows):
            parents.append(i)
            windows.append(window)
            if len(windows) >= batch_size:
                yield parents, windows
                parents, windows = [], []
    if windows:
        yield parents, windows


def finalize_embeddings(acc: np.ndarray, counts: np.ndarray) -> np.ndarray:
    valid = counts > 0
    acc[valid] = acc[valid] / counts[valid, None]
    acc[~valid] = 0.0
    return acc.astype(np.float32)


def esm2_repr_layer(model_name: str) -> int:
    match = re.search(r"_t(\d+)_", model_name)
    if not match:
        raise ValueError(f"Cannot infer ESM-2 representation layer from {model_name}")
    return int(match.group(1))


def parse_esm_layers(text: str, default_layer: int) -> List[int]:
    if not str(text or "").strip():
        return [int(default_layer)]
    layers = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not layers:
        return [int(default_layer)]
    return layers


def pool_esm_representations(reps: Any, seq_len: int, pooling: str) -> np.ndarray:
    import torch

    seq_len = int(max(0, seq_len))
    if seq_len <= 0:
        token_mean = reps[1:2].float().mean(0)
        token_max = reps[1:2].float().max(0).values
    else:
        residue = reps[1 : seq_len + 1].float()
        token_mean = residue.mean(0)
        token_max = residue.max(0).values
    cls = reps[0].float()
    parts = []
    if pooling in {"mean", "mean_cls", "mean_max", "mean_cls_max"}:
        parts.append(token_mean)
    if pooling in {"cls", "mean_cls", "mean_cls_max"}:
        parts.append(cls)
    if pooling in {"mean_max", "mean_cls_max"}:
        parts.append(token_max)
    if not parts:
        parts.append(token_mean)
    return torch.cat(parts, dim=0).detach().cpu().numpy().astype(np.float32)


def compute_esm2_embeddings(args: argparse.Namespace, sequences: Sequence[str], has_sequence: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    import torch
    import esm

    model_name = resolve_model(args.model)
    if not hasattr(esm.pretrained, model_name):
        raise RuntimeError(f"fair-esm does not expose esm.pretrained.{model_name}")
    model_fn = getattr(esm.pretrained, model_name)
    model, alphabet = model_fn()
    repr_layer = esm2_repr_layer(model_name)
    repr_layers = parse_esm_layers(args.esm_layers, repr_layer)
    batch_converter = alphabet.get_batch_converter()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    model = model.eval().to(device)
    if args.fp16 and device.type == "cuda":
        model = model.half()

    acc: np.ndarray | None = None
    counts = np.zeros(len(sequences), dtype=np.float32)
    total_windows = 0
    with torch.no_grad():
        for parents, windows in iter_window_batches(
            sequences,
            args.batch_size,
            args.max_seq_len,
            args.window_stride,
            args.long_strategy,
            args.max_windows_per_sequence,
        ):
            batch = [(str(p), w) for p, w in zip(parents, windows)]
            _, _, tokens = batch_converter(batch)
            tokens = tokens.to(device)
            result = model(tokens, repr_layers=repr_layers, return_contacts=False)
            arrs: List[np.ndarray] = []
            for i, window in enumerate(windows):
                pooled_layers = [
                    pool_esm_representations(result["representations"][layer][i], len(window), args.esm_pooling)
                    for layer in repr_layers
                ]
                arrs.append(np.concatenate(pooled_layers, axis=0).astype(np.float32))
            mat = np.vstack(arrs)
            if acc is None:
                acc = np.zeros((len(sequences), mat.shape[1]), dtype=np.float32)
            for parent, window, emb in zip(parents, windows, mat):
                weight = float(len(window)) if args.window_aggregation == "length_weighted" else 1.0
                acc[parent] += emb * weight
                counts[parent] += weight
            total_windows += len(windows)

    if acc is None:
        raise RuntimeError("No protein windows were encoded.")
    embeddings = finalize_embeddings(acc, counts)
    embeddings[~has_sequence] = 0.0
    return embeddings, {
        "fallback_mode": "esm2",
        "backend": "esm2",
        "model": model_name,
        "repr_layer": repr_layer,
        "repr_layers": repr_layers,
        "esm_pooling": args.esm_pooling,
        "embedding_dim": int(embeddings.shape[1]),
        "total_encoded_windows": int(total_windows),
    }


def compute_prott5_embeddings(args: argparse.Namespace, sequences: Sequence[str], has_sequence: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    import torch
    from transformers import T5EncoderModel, T5Tokenizer

    model_name = resolve_model(args.model)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False, legacy=True)
    dtype = torch.float16 if (args.fp16 and device.type == "cuda") else torch.float32
    model = T5EncoderModel.from_pretrained(model_name, torch_dtype=dtype).eval().to(device)

    acc: np.ndarray | None = None
    counts = np.zeros(len(sequences), dtype=np.float32)
    total_windows = 0
    with torch.no_grad():
        for parents, windows in iter_window_batches(
            sequences,
            args.batch_size,
            args.max_seq_len,
            args.window_stride,
            args.long_strategy,
            args.max_windows_per_sequence,
        ):
            spaced = [" ".join(list(re.sub(r"[UZOB]", "X", w))) for w in windows]
            tokens = tokenizer(
                spaced,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=args.max_seq_len + 1,
                return_tensors="pt",
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            out = model(**tokens)
            h = out.last_hidden_state.float()
            mask = tokens["attention_mask"].float().unsqueeze(-1)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            mat = pooled.detach().cpu().numpy().astype(np.float32)
            if acc is None:
                acc = np.zeros((len(sequences), mat.shape[1]), dtype=np.float32)
            for parent, emb in zip(parents, mat):
                acc[parent] += emb
                counts[parent] += 1.0
            total_windows += len(windows)

    if acc is None:
        raise RuntimeError("No protein windows were encoded.")
    embeddings = finalize_embeddings(acc, counts)
    embeddings[~has_sequence] = 0.0
    return embeddings, {
        "fallback_mode": "prott5",
        "backend": "prott5",
        "model": model_name,
        "embedding_dim": int(embeddings.shape[1]),
        "total_encoded_windows": int(total_windows),
    }


def compute_ankh_embeddings(args: argparse.Namespace, sequences: Sequence[str], has_sequence: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    import torch
    from transformers import AutoTokenizer, T5EncoderModel

    model_name = resolve_model(args.model)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if (args.fp16 and device.type == "cuda") else torch.float32
    model = T5EncoderModel.from_pretrained(model_name, torch_dtype=dtype).eval().to(device)

    acc: np.ndarray | None = None
    counts = np.zeros(len(sequences), dtype=np.float32)
    total_windows = 0
    with torch.no_grad():
        for parents, windows in iter_window_batches(
            sequences,
            args.batch_size,
            args.max_seq_len,
            args.window_stride,
            args.long_strategy,
            args.max_windows_per_sequence,
        ):
            tokens = tokenizer(
                list(windows),
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=args.max_seq_len + 1,
                return_tensors="pt",
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            out = model(**tokens)
            h = out.last_hidden_state.float()
            arrs: List[np.ndarray] = []
            for i, window in enumerate(windows):
                residue_len = min(len(window), h.shape[1])
                emb = h[i, :residue_len].mean(0).detach().cpu().numpy().astype(np.float32)
                arrs.append(emb)
            mat = np.vstack(arrs)
            if acc is None:
                acc = np.zeros((len(sequences), mat.shape[1]), dtype=np.float32)
            for parent, emb in zip(parents, mat):
                acc[parent] += emb
                counts[parent] += 1.0
            total_windows += len(windows)

    if acc is None:
        raise RuntimeError("No protein windows were encoded.")
    embeddings = finalize_embeddings(acc, counts)
    embeddings[~has_sequence] = 0.0
    return embeddings, {
        "fallback_mode": "ankh",
        "backend": "ankh",
        "model": model_name,
        "embedding_dim": int(embeddings.shape[1]),
        "total_encoded_windows": int(total_windows),
    }


def saprot_unknown_structure_tokens(seq: str) -> str:
    # SaProt tokens combine amino-acid and Foldseek 3Di structure symbols.
    # "#" is the SaProt unknown/masked structure placeholder.
    return "".join(f"{aa}#" for aa in clean_protein_sequence(seq))


def compute_saprot_embeddings(args: argparse.Namespace, sequences: Sequence[str], has_sequence: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = resolve_model(args.model)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dtype = torch.float16 if (args.fp16 and device.type == "cuda") else torch.float32
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True).eval().to(device)

    acc: np.ndarray | None = None
    counts = np.zeros(len(sequences), dtype=np.float32)
    total_windows = 0
    with torch.no_grad():
        for parents, windows in iter_window_batches(
            sequences,
            args.batch_size,
            args.max_seq_len,
            args.window_stride,
            args.long_strategy,
            args.max_windows_per_sequence,
        ):
            encoded = [saprot_unknown_structure_tokens(w) for w in windows]
            tokens = tokenizer(
                encoded,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=args.max_seq_len + 2,
                return_tensors="pt",
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            out = model(**tokens)
            h = out.last_hidden_state.float()
            arrs: List[np.ndarray] = []
            for i, window in enumerate(windows):
                residue_len = min(len(window), max(h.shape[1] - 2, 0))
                if residue_len <= 0:
                    emb = np.zeros(h.shape[-1], dtype=np.float32)
                else:
                    emb = h[i, 1 : residue_len + 1].mean(0).detach().cpu().numpy().astype(np.float32)
                arrs.append(emb)
            mat = np.vstack(arrs)
            if acc is None:
                acc = np.zeros((len(sequences), mat.shape[1]), dtype=np.float32)
            for parent, emb in zip(parents, mat):
                acc[parent] += emb
                counts[parent] += 1.0
            total_windows += len(windows)

    if acc is None:
        raise RuntimeError("No protein windows were encoded.")
    embeddings = finalize_embeddings(acc, counts)
    embeddings[~has_sequence] = 0.0
    return embeddings, {
        "fallback_mode": "saprot_unknown_structure",
        "backend": "saprot",
        "model": model_name,
        "embedding_dim": int(embeddings.shape[1]),
        "total_encoded_windows": int(total_windows),
        "structure_token_mode": "unknown_hash_placeholder",
        "structure_aware_saprot": False,
        "note": "SaProt was run with '#' unknown structure placeholders because Foldseek/3Di structure tokens were not provided.",
    }


def compute_aa_kmer_embeddings(_args: argparse.Namespace, sequences: Sequence[str], has_sequence: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    from cgp_common import amino_acid_kmer_embedding

    embeddings = np.vstack([amino_acid_kmer_embedding(seq) for seq in sequences]).astype(np.float32)
    embeddings[~has_sequence] = 0.0
    return embeddings, {
        "fallback_mode": "amino_acid_composition_kmer",
        "backend": "aa_kmer",
        "model": "aa_kmer",
        "embedding_dim": int(embeddings.shape[1]),
        "total_encoded_windows": int(has_sequence.sum()),
    }


def compute_esmc_embeddings(args: argparse.Namespace, sequences: Sequence[str], has_sequence: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    import sys
    import torch
    from pathlib import Path as LocalPath

    esmc_package_dir = os.environ.get("ESMC_PACKAGE_DIR", "")
    if esmc_package_dir:
        sys.modules.pop("esm", None)
        sys.path.insert(0, esmc_package_dir)

    try:
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
    except Exception as exc:
        raise RuntimeError(
            "ESM-C requires the EvolutionaryScale esm package, which is separate from fair-esm. "
            "Run this backend in an isolated environment that provides esm.models.esmc."
        ) from exc

    model_name = resolve_model(args.model)
    local_snapshot_dir = os.environ.get("ESMC_LOCAL_SNAPSHOT_DIR", "").strip()
    if local_snapshot_dir:
        import esm.pretrained as esmc_pretrained
        import esm.utils.constants.esm3 as esmc_constants

        local_root = LocalPath(local_snapshot_dir)
        old_pretrained_data_root = esmc_pretrained.data_root
        old_constants_data_root = esmc_constants.data_root

        def local_data_root(model: str):
            if str(model).startswith("esmc-300"):
                return local_root
            return old_pretrained_data_root(model)

        esmc_pretrained.data_root = local_data_root
        esmc_constants.data_root = local_data_root

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    client = ESMC.from_pretrained(model_name).to(device)
    acc: np.ndarray | None = None
    counts = np.zeros(len(sequences), dtype=np.float32)
    total_windows = 0
    with torch.no_grad():
        for parent, seq in enumerate(sequences):
            for window in make_windows(seq, args.max_seq_len, args.window_stride, args.long_strategy, args.max_windows_per_sequence):
                protein = ESMProtein(sequence=window)
                protein_tensor = client.encode(protein)
                logits = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
                emb_tensor = logits.embeddings
                emb = emb_tensor.float().mean(dim=-2).detach().cpu().numpy().reshape(-1)
                if acc is None:
                    acc = np.zeros((len(sequences), emb.shape[0]), dtype=np.float32)
                acc[parent] += emb.astype(np.float32)
                counts[parent] += 1.0
                total_windows += 1
    if acc is None:
        raise RuntimeError("No protein windows were encoded.")
    embeddings = finalize_embeddings(acc, counts)
    embeddings[~has_sequence] = 0.0
    return embeddings, {
        "fallback_mode": "esmc",
        "backend": "esmc",
        "model": model_name,
        "embedding_dim": int(embeddings.shape[1]),
        "total_encoded_windows": int(total_windows),
        "local_snapshot_dir": local_snapshot_dir or None,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    genes_path = args.data_dir / "genes.parquet"
    if not genes_path.exists():
        raise SystemExit(f"Missing genes.parquet: {genes_path}")
    genes = pd.read_parquet(genes_path)
    if args.smoke_test:
        genes = genes.head(128).copy()
    sequences = genes.get("protein_sequence", pd.Series([""] * len(genes))).fillna("").astype(str).map(clean_protein_sequence).tolist()
    has_sequence = genes.get("has_protein_sequence", pd.Series([True] * len(genes))).astype(bool).to_numpy()
    has_sequence = has_sequence & np.asarray([len(s) > 0 for s in sequences], dtype=bool)

    backend = infer_backend(args.model, args.backend)
    if backend == "esm2":
        embeddings, meta = compute_esm2_embeddings(args, sequences, has_sequence)
    elif backend == "prott5":
        embeddings, meta = compute_prott5_embeddings(args, sequences, has_sequence)
    elif backend == "ankh":
        embeddings, meta = compute_ankh_embeddings(args, sequences, has_sequence)
    elif backend == "saprot":
        embeddings, meta = compute_saprot_embeddings(args, sequences, has_sequence)
    elif backend == "esmc":
        embeddings, meta = compute_esmc_embeddings(args, sequences, has_sequence)
    elif backend == "aa_kmer":
        embeddings, meta = compute_aa_kmer_embeddings(args, sequences, has_sequence)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    if args.smoke_test:
        full = np.zeros((len(pd.read_parquet(genes_path)), embeddings.shape[1]), dtype=np.float32)
        full[: len(embeddings)] = embeddings
        embeddings = full
        full_has = np.zeros(len(full), dtype=bool)
        full_has[: len(has_sequence)] = has_sequence
        has_sequence = full_has

    np.save(args.output_dir / "protein_sequence_embeddings.npy", embeddings.astype(np.float32))
    summary = {
        **meta,
        "num_genes": int(len(embeddings)),
        "num_sequences": int(has_sequence.sum()),
        "genes_with_protein_embedding": int(has_sequence.sum()),
        "missing_sequences": int((~has_sequence).sum()),
        "embedding_shape": list(embeddings.shape),
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "max_seq_len": int(args.max_seq_len),
        "window_stride": int(args.window_stride),
        "max_windows_per_sequence": int(args.max_windows_per_sequence),
        "long_strategy": args.long_strategy,
        "window_aggregation": args.window_aggregation,
        "fp16": bool(args.fp16),
    }
    write_json(args.output_dir / "protein_embedding_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
