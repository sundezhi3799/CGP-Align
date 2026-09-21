from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
LOCAL_PACKAGE_DIR = Path(__file__).resolve().parents[1] / ".python_packages"
os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "torch"))
if LOCAL_PACKAGE_DIR.exists() and str(LOCAL_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_PACKAGE_DIR))

import numpy as np
import pandas as pd

try:
    import requests
except Exception:  # pragma: no cover - optional at runtime
    requests = None

try:
    from rdkit import Chem, DataStructs
    from rdkit import RDLogger
    from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
    RDLogger.DisableLog("rdApp.warning")
except Exception:  # pragma: no cover - optional at runtime
    Chem = None
    DataStructs = None
    AllChem = None
    Crippen = None
    Descriptors = None
    Lipinski = None
    QED = None
    rdMolDescriptors = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - training requires torch
    torch = None
    nn = None
    F = None


OUTPUT_ROOT = Path("output/cgp_align")
RAW_DIR = OUTPUT_ROOT / "raw"
DATA_DIR = OUTPUT_ROOT / "data"
PROTEIN_EMB_DIR = OUTPUT_ROOT / "protein_embeddings"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
LOG_DIR = OUTPUT_ROOT / "logs"
INTRINSIC_DIR = OUTPUT_ROOT / "intrinsic_alignment"
TARGET_RETRIEVAL_DIR = OUTPUT_ROOT / "target_retrieval"
LINK_PREDICTION_DIR = OUTPUT_ROOT / "link_prediction"
MISSING_MODALITY_DIR = OUTPUT_ROOT / "missing_modality"
CASE_STUDY_DIR = OUTPUT_ROOT / "case_studies"

PROFILE_SOURCE_TO_ID = {"compound_perturbation": 0, "gene_perturbation": 1}
PROFILE_TYPE_TO_SOURCE = {"compound": "compound_perturbation", "gene": "gene_perturbation"}
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def ensure_output_dirs(root: Path = OUTPUT_ROOT) -> None:
    for rel in [
        "raw",
        "data",
        "protein_embeddings",
        "checkpoints",
        "logs",
        "intrinsic_alignment",
        "target_retrieval",
        "link_prediction",
        "missing_modality",
        "case_studies",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=json_default) + "\n")


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for this command but is not installed.")


def as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def slug(text: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def read_table(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, nrows=nrows)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t", nrows=nrows)
    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            return pd.read_json(path, lines=True, nrows=nrows)
        df = pd.read_json(path)
        if nrows is not None:
            return df.head(nrows)
        return df
    if suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
        if isinstance(df, pd.DataFrame):
            return df.head(nrows) if nrows is not None else df
        raise ValueError(f"Pickle file is not a DataFrame: {path}")
    raise ValueError(f"Unsupported table format: {path}")


def safe_read_table(path: Path, nrows: int = 5) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        return normalize_columns(read_table(path, nrows=nrows)), None
    except Exception as exc:
        return None, str(exc)


def infer_sep_from_suffix(path: Path) -> str:
    return "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","


def classify_candidate_file(path: Path, columns: Sequence[str], shape: Optional[Sequence[int]]) -> List[str]:
    name = path.name.lower()
    path_text = str(path).replace("\\", "/").lower()
    col_text = " ".join(c.lower() for c in columns)
    roles: List[str] = []
    if "/data/" in path_text:
        if name == "source.parquet":
            roles.append("compound_cell_painting_features")
        if name == "target.parquet":
            roles.append("gene_cell_painting_features")
        if name == "source_map.parquet":
            roles.append("compound_node_table")
        if name == "target_map.parquet":
            roles.append("gene_node_table")
        if name == "s_t_labels.parquet":
            roles.append("compound_target_edges")
            if "/random/" in path_text:
                roles.append("random_split")
            if "/source/" in path_text:
                roles.append("cold_compound_split")
            if "/target/" in path_text:
                roles.append("cold_gene_split")
    has_compound = any(x in name + " " + col_text for x in ["compound", "drug", "smiles", "inchikey", "pubchem", "cid"])
    has_gene = any(x in name + " " + col_text for x in ["gene", "protein", "target", "uniprot", "entrez", "ensembl"])
    has_profile = any(x in name + " " + col_text for x in ["profile", "cell_paint", "cellpainting", "morpholog", "feature"])
    has_edge = any(x in name + " " + col_text for x in ["edge", "link", "interaction", "dti", "target", "source", "destination"])
    has_split = any(x in name + " " + col_text for x in ["split", "train", "valid", "val", "test", "cold", "random"])

    if has_split and "random" in name:
        roles.append("random_split")
    if has_split and any(x in name for x in ["cold_source", "cold-source", "cold_compound", "cold-compound", "source"]):
        roles.append("cold_compound_split")
    if has_split and any(x in name for x in ["cold_target", "cold-target", "cold_gene", "cold-gene", "target"]):
        roles.append("cold_gene_split")
    if has_split and not roles:
        roles.append("split")

    if has_edge and has_compound and has_gene:
        roles.append("compound_target_edges")
    elif has_edge and has_compound:
        roles.append("compound_compound_edges")
    elif has_edge and has_gene:
        roles.append("gene_gene_edges")

    if has_profile and has_compound:
        roles.append("compound_cell_painting_features")
    if has_profile and has_gene:
        roles.append("gene_cell_painting_features")

    if has_compound and not has_profile and not has_edge and not has_split:
        roles.append("compound_node_table")
    if has_gene and not has_profile and not has_edge and not has_split:
        roles.append("gene_node_table")

    numeric_cols = [c for c in columns if re.search(r"feature|feat|Cells_|Nuclei_|Cytoplasm_|Texture_|Intensity_", c, re.I)]
    if shape and len(shape) == 2 and shape[1] > 50 and numeric_cols and has_compound:
        roles.append("compound_cell_painting_features")
    if shape and len(shape) == 2 and shape[1] > 50 and numeric_cols and has_gene:
        roles.append("gene_cell_painting_features")

    return sorted(set(roles)) or ["unknown"]


def inspect_repository_files(root: Path) -> List[Dict[str, Any]]:
    supported = {".csv", ".tsv", ".tab", ".txt", ".parquet", ".json", ".jsonl", ".pkl", ".pickle", ".npy", ".npz"}
    inventory: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in supported:
            continue
        rel = path.relative_to(root)
        item: Dict[str, Any] = {
            "path": str(rel).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
            "suffix": path.suffix.lower(),
        }
        if path.suffix.lower() == ".npy":
            try:
                arr = np.load(path, mmap_mode="r")
                item["shape"] = list(arr.shape)
                item["dtype"] = str(arr.dtype)
                roles = classify_candidate_file(path, [], arr.shape)
            except Exception as exc:
                item["read_error"] = str(exc)
                roles = ["unknown"]
        elif path.suffix.lower() == ".npz":
            try:
                archive = np.load(path, allow_pickle=False)
                item["arrays"] = {k: list(archive[k].shape) for k in archive.files}
                roles = classify_candidate_file(path, list(archive.files), None)
            except Exception as exc:
                item["read_error"] = str(exc)
                roles = ["unknown"]
        else:
            df, err = safe_read_table(path, nrows=5)
            if df is not None:
                item["columns"] = list(map(str, df.columns))
                item["sample_shape"] = list(df.shape)
                roles = classify_candidate_file(path, item["columns"], item["sample_shape"])
            else:
                item["read_error"] = err
                roles = classify_candidate_file(path, [], None)
        item["candidate_roles"] = roles
        inventory.append(item)
    return inventory


def extract_external_data_urls(root: Path) -> List[str]:
    urls: List[str] = []
    for name in ["README.md", "README.rst", "README.txt"]:
        path = root / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in re.findall(r"https?://[^\s)>\"]+", text):
            if re.search(r"zenodo|figshare|osf|dropbox|drive\.google|box\.com|s3|data|download", match, re.I):
                urls.append(match.rstrip(".,"))
    return sorted(set(urls))


def github_zip_candidates(url: str) -> List[str]:
    match = re.match(r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if not match:
        return []
    owner, repo = match.group(1), match.group(2)
    return [
        f"https://github.com/{owner}/{repo}/archive/refs/heads/main.zip",
        f"https://github.com/{owner}/{repo}/archive/refs/heads/master.zip",
    ]


def download_github_zip(raw_dir: Path, target: Path, url: str, summary: Dict[str, Any]) -> bool:
    if requests is None:
        summary["zip_fallback_error"] = "requests is unavailable"
        return False
    for zip_url in github_zip_candidates(url):
        try:
            response = requests.get(zip_url, timeout=60)
            summary.setdefault("zip_attempts", []).append({"url": zip_url, "status_code": response.status_code})
            if response.status_code != 200:
                continue
            zip_path = raw_dir / "motive_repo.zip"
            zip_path.write_bytes(response.content)
            extract_dir = raw_dir / "motive_zip_extract"
            if extract_dir.exists():
                resolved_extract = extract_dir.resolve()
                resolved_raw = raw_dir.resolve()
                if resolved_raw not in resolved_extract.parents and resolved_extract != resolved_raw:
                    raise RuntimeError(f"Refusing to remove unexpected extract dir: {extract_dir}")
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(extract_dir)
            children = [p for p in extract_dir.iterdir() if p.is_dir()]
            if not children:
                raise RuntimeError(f"Zip fallback did not contain a repository directory: {zip_url}")
            if target.exists():
                if any(target.iterdir()):
                    summary["zip_fallback_error"] = f"Target exists and is not empty: {target}"
                    return False
                target.rmdir()
            shutil.move(str(children[0]), str(target))
            summary["zip_fallback_url"] = zip_url
            summary["zip_fallback_used"] = True
            return True
        except Exception as exc:
            summary.setdefault("zip_attempts", []).append({"url": zip_url, "error": str(exc)})
    return False


def clone_or_use_motive_repo(raw_dir: Path, source_dir: Optional[Path], url: str) -> Tuple[Path, Dict[str, Any]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {"source_url": url, "used_existing": False, "cloned": False}
    if source_dir is not None:
        if not source_dir.exists():
            raise FileNotFoundError(f"Specified source_dir does not exist: {source_dir}")
        summary["source_dir"] = str(source_dir)
        summary["used_source_dir"] = True
        return source_dir, summary

    target = raw_dir / "motive"
    if target.exists() and any(target.iterdir()):
        summary["used_existing"] = True
        return target, summary
    if target.exists():
        target.rmdir()

    cmd = ["git", "clone", "--depth", "1", url, str(target)]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    summary["git_stdout"] = proc.stdout[-4000:]
    summary["git_stderr"] = proc.stderr[-4000:]
    summary["git_returncode"] = proc.returncode
    if proc.returncode != 0:
        summary["clone_failed_trying_zip_fallback"] = True
        if download_github_zip(raw_dir, target, url, summary):
            return target, summary
        raise RuntimeError(f"Failed to clone MOTIVE repository from {url}: {proc.stderr.strip()}")
    summary["cloned"] = True
    return target, summary


def canonicalize_smiles(smiles: str, remove_stereo: bool = True) -> Optional[str]:
    if not smiles or not isinstance(smiles, str) or Chem is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return None
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if fragments:
            mol = max(fragments, key=lambda m: m.GetNumHeavyAtoms())
        if remove_stereo:
            Chem.RemoveStereochemistry(mol)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=not remove_stereo)
    except Exception:
        return None


def smiles_from_inchi(inchi: str, remove_stereo: bool = True) -> Optional[str]:
    if not inchi or not isinstance(inchi, str) or Chem is None:
        return None
    try:
        mol = Chem.MolFromInchi(inchi, sanitize=True)
        if mol is None:
            return None
        if remove_stereo:
            Chem.RemoveStereochemistry(mol)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=not remove_stereo)
    except Exception:
        return None


def compound_features_from_smiles(smiles_values: Sequence[Optional[str]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    dim = 2048 + 11
    features = np.zeros((len(smiles_values), dim), dtype=np.float32)
    failures: List[Dict[str, Any]] = []
    for i, smiles in enumerate(smiles_values):
        if not smiles or Chem is None or AllChem is None:
            failures.append({"row_index": i, "smiles": smiles, "reason": "missing_rdkit_or_smiles"})
            continue
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            failures.append({"row_index": i, "smiles": smiles, "reason": "canonicalization_failed"})
            continue
        mol = Chem.MolFromSmiles(canonical)
        if mol is None:
            failures.append({"row_index": i, "smiles": smiles, "reason": "mol_from_smiles_failed"})
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            arr = np.zeros((2048,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            desc = np.asarray(
                [
                    Descriptors.MolWt(mol),
                    Crippen.MolLogP(mol),
                    rdMolDescriptors.CalcTPSA(mol),
                    Lipinski.NumHAcceptors(mol),
                    Lipinski.NumHDonors(mol),
                    Lipinski.NumRotatableBonds(mol),
                    rdMolDescriptors.CalcNumRings(mol),
                    rdMolDescriptors.CalcNumAromaticRings(mol),
                    Chem.GetFormalCharge(mol),
                    rdMolDescriptors.CalcFractionCSP3(mol),
                    QED.qed(mol),
                ],
                dtype=np.float32,
            )
            features[i] = np.concatenate([arr, desc], axis=0)
        except Exception as exc:
            failures.append({"row_index": i, "smiles": smiles, "reason": str(exc)})
    if len(features):
        desc = features[:, 2048:]
        mean = desc.mean(axis=0, keepdims=True)
        std = desc.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        features[:, 2048:] = (desc - mean) / std
    return features, failures


def zscore_matrix(x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = np.asarray(x, dtype=np.float32)
    mean = np.nanmean(x, axis=0, keepdims=True)
    std = np.nanstd(x, axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    y = (x - mean) / std
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y.astype(np.float32), {"mean_shape": list(mean.shape), "std_shape": list(std.shape)}


def query_pubchem_smiles(identifier: str, namespace: str = "name", pause: float = 0.2) -> Optional[Dict[str, Any]]:
    if requests is None or not identifier:
        return None
    safe_id = str(identifier).strip()
    if not safe_id:
        return None
    namespace = namespace.lower()
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    if namespace in {"cid", "pubchem_cid"}:
        url = f"{base}/compound/cid/{safe_id}/property/CanonicalSMILES,IsomericSMILES,InChIKey/JSON"
    elif namespace == "inchikey":
        url = f"{base}/compound/inchikey/{safe_id}/property/CanonicalSMILES,IsomericSMILES,InChIKey/JSON"
    else:
        url = f"{base}/compound/name/{safe_id}/property/CanonicalSMILES,IsomericSMILES,InChIKey/JSON"
    time.sleep(pause)
    try:
        response = requests.get(url, timeout=20)
        if response.status_code != 200:
            return {"identifier": safe_id, "namespace": namespace, "status_code": response.status_code, "smiles": None}
        props = response.json().get("PropertyTable", {}).get("Properties", [])
        if not props:
            return {"identifier": safe_id, "namespace": namespace, "status_code": 200, "smiles": None}
        item = props[0]
        smiles = item.get("CanonicalSMILES") or item.get("IsomericSMILES")
        return {
            "identifier": safe_id,
            "namespace": namespace,
            "status_code": 200,
            "smiles": smiles,
            "inchikey": item.get("InChIKey"),
            "cid": item.get("CID"),
        }
    except Exception as exc:
        return {"identifier": safe_id, "namespace": namespace, "error": str(exc), "smiles": None}


def query_uniprot_sequence(gene_symbol: str, pause: float = 0.2, organism_id: str = "9606") -> Optional[Dict[str, Any]]:
    if requests is None or not gene_symbol:
        return None
    query = f"(gene_exact:{gene_symbol}) AND (organism_id:{organism_id})"
    params = {
        "query": query,
        "fields": "accession,gene_primary,protein_name,sequence,length,reviewed,cc_alternative_products",
        "format": "json",
        "size": "10",
    }
    time.sleep(pause)
    try:
        response = requests.get("https://rest.uniprot.org/uniprotkb/search", params=params, timeout=30)
        if response.status_code != 200:
            return {"gene_symbol": gene_symbol, "status_code": response.status_code, "sequence": None}
        results = response.json().get("results", [])
        best = None
        best_key = None
        for result in results:
            seq = result.get("sequence", {}).get("value")
            if not seq:
                continue
            reviewed = 1 if result.get("entryType", "").lower().startswith("reviewed") else 0
            length = int(result.get("sequence", {}).get("length", len(seq)))
            canonical_bonus = 1 if "-1" not in result.get("primaryAccession", "") else 0
            key = (reviewed, canonical_bonus, length)
            if best_key is None or key > best_key:
                best = result
                best_key = key
        if best is None:
            return {"gene_symbol": gene_symbol, "status_code": 200, "sequence": None}
        return {
            "gene_symbol": gene_symbol,
            "status_code": 200,
            "uniprot_id": best.get("primaryAccession"),
            "protein_sequence": best.get("sequence", {}).get("value"),
            "length": best.get("sequence", {}).get("length"),
        }
    except Exception as exc:
        return {"gene_symbol": gene_symbol, "error": str(exc), "sequence": None}


def _make_synthetic_smiles(i: int) -> str:
    templates = [
        "CCO",
        "CCN",
        "CCC(=O)O",
        "CCOC(=O)C",
        "c1ccccc1",
        "c1ccncc1",
        "CC(C)O",
        "CCS",
        "CCCl",
        "CN(C)C",
        "CC(=O)N",
        "CC(C)C(=O)O",
        "c1ccc(O)cc1",
        "c1ccc(N)cc1",
        "CCOC",
        "CC(C)N",
    ]
    base = templates[i % len(templates)]
    chain = "C" * (1 + (i // len(templates)) % 5)
    if i % 7 == 0:
        return chain + "N"
    if i % 11 == 0:
        return chain + "O"
    return base


def _make_synthetic_sequence(rng: np.random.Generator, latent: np.ndarray, min_len: int = 80, max_len: int = 260) -> str:
    weights = np.abs(np.resize(latent, len(AA_ALPHABET))) + 0.05
    probs = weights / weights.sum()
    length = int(rng.integers(min_len, max_len + 1))
    return "".join(rng.choice(list(AA_ALPHABET), size=length, p=probs).tolist())


def build_synthetic_dataset(
    data_dir: Path,
    n_compounds: int = 500,
    n_genes: int = 1000,
    n_edges: int = 2000,
    profile_dim: int = 256,
    seed: int = 13,
) -> Dict[str, Any]:
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    latent_dim = 64
    compound_latent = rng.normal(size=(n_compounds, latent_dim)).astype(np.float32)
    gene_latent = rng.normal(size=(n_genes, latent_dim)).astype(np.float32)

    smiles = [canonicalize_smiles(_make_synthetic_smiles(i)) or _make_synthetic_smiles(i) for i in range(n_compounds)]
    structure_features, failed = compound_features_from_smiles(smiles)
    if structure_features.shape[1] == 0:
        structure_features = rng.normal(size=(n_compounds, 2059)).astype(np.float32)

    projection_c = rng.normal(scale=0.15, size=(latent_dim, profile_dim)).astype(np.float32)
    projection_g = projection_c + rng.normal(scale=0.02, size=(latent_dim, profile_dim)).astype(np.float32)
    compound_profiles = compound_latent @ projection_c + rng.normal(scale=0.25, size=(n_compounds, profile_dim))
    gene_profiles = gene_latent @ projection_g + rng.normal(scale=0.25, size=(n_genes, profile_dim))
    compound_profiles, _ = zscore_matrix(compound_profiles)
    gene_profiles, _ = zscore_matrix(gene_profiles)
    profile_features = np.vstack([compound_profiles, gene_profiles]).astype(np.float32)

    compounds: List[Dict[str, Any]] = []
    profiles: List[Dict[str, Any]] = []
    for i in range(n_compounds):
        compound_id = f"C{i:06d}"
        profile_id = f"CP{i:06d}"
        compounds.append(
            {
                "compound_id": compound_id,
                "motive_node_id": f"compound:{i}",
                "canonical_smiles": smiles[i],
                "inchikey": "",
                "pubchem_cid": "",
                "has_structure": True,
                "has_compound_profile": True,
                "structure_feature_index": i,
                "compound_profile_id": profile_id,
            }
        )
        profiles.append(
            {
                "profile_id": profile_id,
                "perturbation_type": "compound",
                "perturbation_id": compound_id,
                "motive_node_id": f"compound:{i}",
                "feature_index": i,
                "replicate_count": 1,
                "source_dataset": "synthetic_smoke",
                "has_profile": True,
            }
        )

    genes: List[Dict[str, Any]] = []
    for i in range(n_genes):
        gene_id = f"G{i:06d}"
        profile_id = f"GP{i:06d}"
        symbol = f"SYN{i:05d}"
        seq = _make_synthetic_sequence(rng, gene_latent[i])
        genes.append(
            {
                "gene_id": gene_id,
                "motive_node_id": f"gene:{i}",
                "gene_symbol": symbol,
                "entrez_id": "",
                "ensembl_id": "",
                "uniprot_id": f"UP{i:06d}",
                "protein_sequence": seq,
                "has_protein_sequence": True,
                "has_gene_profile": True,
                "protein_embedding_index": -1,
                "gene_profile_id": profile_id,
            }
        )
        profiles.append(
            {
                "profile_id": profile_id,
                "perturbation_type": "gene",
                "perturbation_id": gene_id,
                "motive_node_id": f"gene:{i}",
                "feature_index": n_compounds + i,
                "replicate_count": 1,
                "source_dataset": "synthetic_smoke",
                "has_profile": True,
            }
        )

    sim = compound_latent @ gene_latent.T
    flat_top = np.argpartition(sim.reshape(-1), -max(n_edges * 2, n_edges))[-max(n_edges * 2, n_edges):]
    rng.shuffle(flat_top)
    edges_seen = set()
    edges: List[Dict[str, Any]] = []
    for flat in flat_top:
        c = int(flat // n_genes)
        g = int(flat % n_genes)
        key = (c, g)
        if key in edges_seen:
            continue
        edges_seen.add(key)
        edges.append(
            {
                "compound_id": f"C{c:06d}",
                "gene_id": f"G{g:06d}",
                "edge_type": "synthetic_target",
                "source_database": "synthetic_smoke",
                "label": 1,
            }
        )
        if len(edges) >= n_edges:
            break

    compounds_df = pd.DataFrame(compounds)
    genes_df = pd.DataFrame(genes)
    profiles_df = pd.DataFrame(profiles)
    edges_df = pd.DataFrame(edges)
    compounds_df.to_parquet(data_dir / "compounds.parquet", index=False)
    genes_df.to_parquet(data_dir / "genes.parquet", index=False)
    profiles_df.to_parquet(data_dir / "profiles.parquet", index=False)
    edges_df.to_parquet(data_dir / "compound_target_edges.parquet", index=False)
    np.save(data_dir / "compound_structure_features.npy", structure_features.astype(np.float32))
    np.save(data_dir / "profile_features.npy", profile_features.astype(np.float32))
    write_json(data_dir / "pubchem_smiles_cache.json", {})
    write_json(data_dir / "uniprot_sequence_cache.json", {})
    pd.DataFrame(failed).to_csv(data_dir / "failed_smiles.csv", index=False)
    pd.DataFrame(columns=["gene_id", "gene_symbol"]).to_csv(data_dir / "missing_protein_sequences.csv", index=False)

    splits = make_edge_splits(edges_df, seed=seed)
    for name, payload in splits.items():
        write_json(data_dir / f"splits_{name}.json", payload)
    summary = dataset_summary(data_dir, compounds_df, genes_df, profiles_df, edges_df, structure_features, profile_features, splits)
    summary["synthetic_smoke"] = True
    write_json(data_dir / "dataset_summary.json", summary)
    return summary


def make_edge_records(df: pd.DataFrame) -> List[Dict[str, str]]:
    return [
        {"compound_id": str(row.compound_id), "gene_id": str(row.gene_id)}
        for row in df[["compound_id", "gene_id"]].itertuples(index=False)
    ]


def make_edge_splits(edges_df: pd.DataFrame, seed: int = 13) -> Dict[str, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n = len(edges_df)
    order = rng.permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    random_split = {
        "split_type": "random_edge",
        "train": make_edge_records(edges_df.iloc[order[:n_train]]),
        "val": make_edge_records(edges_df.iloc[order[n_train : n_train + n_val]]),
        "test": make_edge_records(edges_df.iloc[order[n_train + n_val :]]),
    }

    def cold_by(column: str, split_type: str) -> Dict[str, Any]:
        values = np.asarray(sorted(edges_df[column].astype(str).unique()))
        rng.shuffle(values)
        n_test = max(1, int(0.15 * len(values)))
        n_val_groups = max(1, int(0.10 * len(values)))
        test_values = set(values[:n_test])
        val_values = set(values[n_test : n_test + n_val_groups])
        test = edges_df[edges_df[column].astype(str).isin(test_values)]
        val = edges_df[edges_df[column].astype(str).isin(val_values)]
        train = edges_df[~edges_df[column].astype(str).isin(test_values | val_values)]
        return {
            "split_type": split_type,
            "cold_column": column,
            "train": make_edge_records(train),
            "val": make_edge_records(val),
            "test": make_edge_records(test),
        }

    return {
        "random": random_split,
        "cold_compound": cold_by("compound_id", "cold_compound"),
        "cold_gene": cold_by("gene_id", "cold_gene"),
    }


def dataset_summary(
    data_dir: Path,
    compounds_df: pd.DataFrame,
    genes_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    structure_features: np.ndarray,
    profile_features: Optional[np.ndarray],
    splits: Optional[Dict[str, Dict[str, Any]]] = None,
    feature_dim_by_source: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    splits = splits or {}
    split_sizes: Dict[str, Dict[str, int]] = {}
    for name, payload in splits.items():
        split_sizes[name] = {k: len(payload.get(k, [])) for k in ["train", "val", "test"]}
    profile_dim: Any = None
    if profile_features is not None:
        profile_dim = int(profile_features.shape[1]) if profile_features.ndim == 2 else None
    failed_smiles_count = 0
    failed_smiles_path = data_dir / "failed_smiles.csv"
    if failed_smiles_path.exists() and failed_smiles_path.stat().st_size > 0:
        try:
            failed_smiles_count = int(len(pd.read_csv(failed_smiles_path)))
        except Exception:
            failed_smiles_count = 0
    return {
        "num_compounds": int(len(compounds_df)),
        "num_genes": int(len(genes_df)),
        "num_profiles": int(len(profiles_df)),
        "num_compound_profiles": int((profiles_df["perturbation_type"] == "compound").sum()) if "perturbation_type" in profiles_df else 0,
        "num_gene_profiles": int((profiles_df["perturbation_type"] == "gene").sum()) if "perturbation_type" in profiles_df else 0,
        "num_edges": int(len(edges_df)),
        "compounds_with_smiles": int(compounds_df.get("has_structure", pd.Series(False, index=compounds_df.index)).sum()),
        "genes_with_protein_sequence": int(genes_df.get("has_protein_sequence", pd.Series(False, index=genes_df.index)).sum()),
        "compounds_with_profile": int(compounds_df.get("has_compound_profile", pd.Series(False, index=compounds_df.index)).sum()),
        "genes_with_profile": int(genes_df.get("has_gene_profile", pd.Series(False, index=genes_df.index)).sum()),
        "structure_feature_dim": int(structure_features.shape[1]) if structure_features.ndim == 2 else None,
        "profile_feature_dim": profile_dim,
        "feature_dim_by_source": feature_dim_by_source,
        "split_sizes": split_sizes,
        "missing_smiles_count": int((~compounds_df.get("has_structure", pd.Series(False, index=compounds_df.index)).astype(bool)).sum()),
        "missing_protein_sequence_count": int((~genes_df.get("has_protein_sequence", pd.Series(False, index=genes_df.index)).astype(bool)).sum()),
        "failed_smiles_count": failed_smiles_count,
    }


def amino_acid_kmer_embedding(sequence: str, k: int = 2) -> np.ndarray:
    seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", str(sequence).upper())
    aa_to_idx = {aa: i for i, aa in enumerate(AA_ALPHABET)}
    comp = np.zeros(len(AA_ALPHABET), dtype=np.float32)
    for aa in seq:
        if aa in aa_to_idx:
            comp[aa_to_idx[aa]] += 1.0
    if comp.sum() > 0:
        comp /= comp.sum()
    kmer = np.zeros(len(AA_ALPHABET) ** k, dtype=np.float32)
    if len(seq) >= k:
        for i in range(len(seq) - k + 1):
            token = seq[i : i + k]
            idx = 0
            ok = True
            for aa in token:
                if aa not in aa_to_idx:
                    ok = False
                    break
                idx = idx * len(AA_ALPHABET) + aa_to_idx[aa]
            if ok:
                kmer[idx] += 1.0
        if kmer.sum() > 0:
            kmer /= kmer.sum()
    length_feats = np.asarray([len(seq) / 1000.0, math.log1p(len(seq)) / 10.0], dtype=np.float32)
    return np.concatenate([comp, kmer, length_feats], axis=0)


def load_split(data_dir: Path, split_name: str) -> Dict[str, Any]:
    path = data_dir / f"splits_{split_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return read_json(path)


def read_cgp_tables(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def read_table(stem: str) -> pd.DataFrame:
        parquet_path = data_dir / f"{stem}.parquet"
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            csv_path = data_dir / f"{stem}.csv.gz"
            if csv_path.exists():
                return pd.read_csv(csv_path, keep_default_na=False)
            raise

    return (
        read_table("compounds"),
        read_table("genes"),
        read_table("profiles"),
        read_table("compound_target_edges"),
    )


@dataclass
class CGPBatch:
    compound_indices: np.ndarray
    gene_indices: np.ndarray
    intrinsic_compound_indices: Optional[np.ndarray] = None
    intrinsic_gene_indices: Optional[np.ndarray] = None


class CGPBundle:
    def __init__(self, data_dir: Path = DATA_DIR, protein_embedding_dir: Path = PROTEIN_EMB_DIR):
        self.data_dir = data_dir
        self.protein_embedding_dir = protein_embedding_dir
        self.compounds, self.genes, self.profiles, self.edges = read_cgp_tables(data_dir)
        summary_path = data_dir / "dataset_summary.json"
        self.dataset_summary = read_json(summary_path) if summary_path.exists() else {}
        self.feature_dim_by_source = self.dataset_summary.get("feature_dim_by_source")
        self.structure_features = np.load(data_dir / "compound_structure_features.npy").astype(np.float32)
        self.profile_features = np.load(data_dir / "profile_features.npy").astype(np.float32)
        protein_path = protein_embedding_dir / "protein_sequence_embeddings.npy"
        if protein_path.exists():
            self.protein_embeddings = np.load(protein_path).astype(np.float32)
            if self.protein_embeddings.shape[0] != len(self.genes):
                raise ValueError(
                    f"Protein embedding row count mismatch: {protein_path} has "
                    f"{self.protein_embeddings.shape[0]} rows but genes.parquet has {len(self.genes)} rows."
                )
        else:
            dim = 20 + 20 * 20 + 2
            self.protein_embeddings = np.zeros((len(self.genes), dim), dtype=np.float32)

        self.compound_id_to_idx = {str(v): i for i, v in enumerate(self.compounds["compound_id"].astype(str))}
        self.gene_id_to_idx = {str(v): i for i, v in enumerate(self.genes["gene_id"].astype(str))}
        self.profile_id_to_row = {str(v): i for i, v in enumerate(self.profiles["profile_id"].astype(str))}
        self.compound_profile_row = np.full(len(self.compounds), -1, dtype=np.int64)
        self.gene_profile_row = np.full(len(self.genes), -1, dtype=np.int64)
        for i, row in self.compounds.iterrows():
            pid = str(row.get("compound_profile_id", ""))
            if pid in self.profile_id_to_row:
                self.compound_profile_row[i] = self.profile_id_to_row[pid]
        for i, row in self.genes.iterrows():
            pid = str(row.get("gene_profile_id", ""))
            if pid in self.profile_id_to_row:
                self.gene_profile_row[i] = self.profile_id_to_row[pid]

        self.profile_feature_index = self.profiles["feature_index"].to_numpy(dtype=np.int64)
        self.profile_source_ids = np.asarray(
            [
                PROFILE_SOURCE_TO_ID[PROFILE_TYPE_TO_SOURCE.get(str(t), "compound_perturbation")]
                for t in self.profiles["perturbation_type"].astype(str)
            ],
            dtype=np.int64,
        )
        self.has_structure = self.compounds.get("has_structure", pd.Series(True, index=self.compounds.index)).astype(bool).to_numpy()
        self.has_compound_profile = self.compounds.get("has_compound_profile", pd.Series(True, index=self.compounds.index)).astype(bool).to_numpy()
        self.has_protein = self.genes.get("has_protein_sequence", pd.Series(True, index=self.genes.index)).astype(bool).to_numpy()
        self.has_gene_profile = self.genes.get("has_gene_profile", pd.Series(True, index=self.genes.index)).astype(bool).to_numpy()
        self.edge_pairs = [
            (self.compound_id_to_idx[str(r.compound_id)], self.gene_id_to_idx[str(r.gene_id)])
            for r in self.edges[["compound_id", "gene_id"]].itertuples(index=False)
            if str(r.compound_id) in self.compound_id_to_idx and str(r.gene_id) in self.gene_id_to_idx
        ]
        self.known_positive_pairs = set(self.edge_pairs)
        self.compound_to_genes: Dict[int, set[int]] = {}
        self.gene_to_compounds: Dict[int, set[int]] = {}
        for c, g in self.edge_pairs:
            self.compound_to_genes.setdefault(c, set()).add(g)
            self.gene_to_compounds.setdefault(g, set()).add(c)

    @property
    def structure_dim(self) -> int:
        return int(self.structure_features.shape[1])

    @property
    def protein_dim(self) -> int:
        return int(self.protein_embeddings.shape[1])

    @property
    def profile_dim(self) -> int:
        return int(self.profile_features.shape[1])

    def split_pairs(self, split_name: str = "random", fold: str = "train") -> List[Tuple[int, int]]:
        split = load_split(self.data_dir, split_name)
        pairs = []
        for item in split.get(fold, []):
            c = str(item["compound_id"])
            g = str(item["gene_id"])
            if c in self.compound_id_to_idx and g in self.gene_id_to_idx:
                pairs.append((self.compound_id_to_idx[c], self.gene_id_to_idx[g]))
        return pairs

    def iter_edge_batches(self, pairs: List[Tuple[int, int]], batch_size: int, shuffle: bool = True) -> Iterable[CGPBatch]:
        order = np.arange(len(pairs))
        if shuffle:
            np.random.shuffle(order)
        for start in range(0, len(order), batch_size):
            selected = [pairs[i] for i in order[start : start + batch_size]]
            if not selected:
                continue
            yield CGPBatch(
                compound_indices=np.asarray([p[0] for p in selected], dtype=np.int64),
                gene_indices=np.asarray([p[1] for p in selected], dtype=np.int64),
            )

    def profile_rows_for_compounds(self, compound_indices: np.ndarray) -> np.ndarray:
        return self.compound_profile_row[compound_indices]

    def profile_rows_for_genes(self, gene_indices: np.ndarray) -> np.ndarray:
        return self.gene_profile_row[gene_indices]

    def profile_features_for_rows(self, profile_rows: np.ndarray) -> np.ndarray:
        feature_indices = self.profile_feature_index[profile_rows]
        return self.profile_features[feature_indices]

    def source_ids_for_rows(self, profile_rows: np.ndarray) -> np.ndarray:
        return self.profile_source_ids[profile_rows]


if torch is not None:

    class MLPEncoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dims: Sequence[int], embed_dim: int, dropout: float = 0.1):
            super().__init__()
            layers: List[nn.Module] = []
            prev = input_dim
            for hidden in hidden_dims:
                layers.extend([nn.Linear(prev, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)])
                prev = hidden
            layers.append(nn.Linear(prev, embed_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.normalize(self.net(x), dim=-1)


    class ProfileEncoder(nn.Module):
        def __init__(
            self,
            profile_dim: int,
            embed_dim: int,
            hidden_dim: int = 1024,
            dropout: float = 0.1,
            feature_dim_by_source: Optional[Dict[str, int]] = None,
        ):
            super().__init__()
            self.profile_dim = profile_dim
            self.feature_dim_by_source = feature_dim_by_source or {}
            compound_dim = self.feature_dim_by_source.get("compound")
            gene_dim = self.feature_dim_by_source.get("gene")
            self.use_source_specific_projection = bool(compound_dim and gene_dim and compound_dim != gene_dim)
            if self.use_source_specific_projection:
                self.input_proj_by_source = nn.ModuleList(
                    [
                        nn.Sequential(nn.Linear(int(compound_dim), hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()),
                        nn.Sequential(nn.Linear(int(gene_dim), hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()),
                    ]
                )
                self.input_proj = None
            else:
                self.input_proj = nn.Sequential(nn.Linear(profile_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
                self.input_proj_by_source = None
            self.source_embedding = nn.Embedding(2, hidden_dim)
            self.shared = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, embed_dim),
            )

        def forward(self, x: torch.Tensor, source_ids: torch.Tensor) -> torch.Tensor:
            if self.use_source_specific_projection:
                h = x.new_zeros((x.shape[0], self.source_embedding.embedding_dim))
                for source_id, raw_dim in [(0, int(self.feature_dim_by_source["compound"])), (1, int(self.feature_dim_by_source["gene"]))]:
                    mask = source_ids == source_id
                    if bool(mask.any()):
                        h[mask] = self.input_proj_by_source[source_id](x[mask, :raw_dim])
            else:
                h = self.input_proj(x)
            h = h + self.source_embedding(source_ids)
            return F.normalize(self.shared(h), dim=-1)


    class CGPAlignModel(nn.Module):
        def __init__(
            self,
            structure_dim: int,
            protein_dim: int,
            profile_dim: int,
            embed_dim: int = 256,
            dropout: float = 0.1,
            feature_dim_by_source: Optional[Dict[str, int]] = None,
        ):
            super().__init__()
            self.compound_encoder = MLPEncoder(structure_dim, [1024, 512], embed_dim, dropout)
            self.protein_encoder = MLPEncoder(protein_dim, [512], embed_dim, dropout)
            self.profile_encoder = ProfileEncoder(
                profile_dim,
                embed_dim,
                hidden_dim=1024,
                dropout=dropout,
                feature_dim_by_source=feature_dim_by_source,
            )

        def encode_compound(self, x: torch.Tensor) -> torch.Tensor:
            return self.compound_encoder(x)

        def encode_protein(self, x: torch.Tensor) -> torch.Tensor:
            return self.protein_encoder(x)

        def encode_profile(self, x: torch.Tensor, source_ids: torch.Tensor) -> torch.Tensor:
            return self.profile_encoder(x, source_ids)


def bidirectional_infonce(a: "torch.Tensor", b: "torch.Tensor", temperature: float) -> "torch.Tensor":
    if a.shape[0] < 2 or b.shape[0] < 2:
        return a.new_tensor(0.0)
    logits = a @ b.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def multipositive_contrastive(
    query: "torch.Tensor", gallery: "torch.Tensor", pos_mask: "torch.Tensor", temperature: float
) -> "torch.Tensor":
    if query.shape[0] == 0 or gallery.shape[0] == 0:
        return query.new_tensor(0.0)
    logits = query @ gallery.T / temperature
    valid = pos_mask.any(dim=1)
    if not bool(valid.any()):
        return query.new_tensor(0.0)
    logits = logits[valid]
    mask = pos_mask[valid]
    neg_inf = torch.finfo(logits.dtype).min
    pos_logits = logits.masked_fill(~mask, neg_inf)
    numerator = torch.logsumexp(pos_logits, dim=1)
    denominator = torch.logsumexp(logits, dim=1)
    return -(numerator - denominator).mean()


def bce_link_loss(pos_scores: "torch.Tensor", neg_scores: "torch.Tensor") -> "torch.Tensor":
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return (pos_scores.sum() + neg_scores.sum()) * 0.0
    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)
    scores = torch.cat([pos_scores, neg_scores], dim=0)
    return F.binary_cross_entropy_with_logits(scores, labels)


def pair_score(
    z_c: "torch.Tensor",
    z_p: "torch.Tensor",
    z_cp: Optional["torch.Tensor"],
    z_gp: Optional["torch.Tensor"],
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.2,
    delta: float = 0.1,
) -> "torch.Tensor":
    score = alpha * (z_c * z_p).sum(dim=-1)
    if z_cp is not None:
        score = score + delta * (z_cp * z_p).sum(dim=-1)
    if z_gp is not None:
        score = score + gamma * (z_c * z_gp).sum(dim=-1)
    if z_cp is not None and z_gp is not None:
        score = score + beta * (z_cp * z_gp).sum(dim=-1)
    return score


def unique_preserve(values: np.ndarray) -> np.ndarray:
    seen = set()
    out = []
    for v in values.tolist():
        if int(v) not in seen:
            seen.add(int(v))
            out.append(int(v))
    return np.asarray(out, dtype=np.int64)


def tensor_from_numpy(x: np.ndarray, device: "torch.device") -> "torch.Tensor":
    return torch.from_numpy(np.asarray(x)).float().to(device)


def long_tensor(x: np.ndarray, device: "torch.device") -> "torch.Tensor":
    return torch.from_numpy(np.asarray(x, dtype=np.int64)).long().to(device)


def build_pos_mask(
    row_compounds: np.ndarray, col_genes: np.ndarray, known_positive_pairs: set[Tuple[int, int]], device: "torch.device"
) -> "torch.Tensor":
    mask = np.zeros((len(row_compounds), len(col_genes)), dtype=bool)
    gene_to_col = {int(g): j for j, g in enumerate(col_genes.tolist())}
    for i, c in enumerate(row_compounds.tolist()):
        for g in col_genes.tolist():
            if (int(c), int(g)) in known_positive_pairs:
                mask[i, gene_to_col[int(g)]] = True
    return torch.from_numpy(mask).to(device)


def sample_negative_pairs(
    n: int,
    num_compounds: int,
    num_genes: int,
    positives: set[Tuple[int, int]],
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    negatives: List[Tuple[int, int]] = []
    attempts = 0
    max_attempts = max(1000, n * 50)
    while len(negatives) < n and attempts < max_attempts:
        c = int(rng.integers(0, num_compounds))
        g = int(rng.integers(0, num_genes))
        if (c, g) not in positives:
            negatives.append((c, g))
        attempts += 1
    return negatives


def encode_pair_components(
    model: "CGPAlignModel",
    bundle: CGPBundle,
    compound_indices: np.ndarray,
    gene_indices: np.ndarray,
    device: "torch.device",
) -> Tuple["torch.Tensor", "torch.Tensor", Optional["torch.Tensor"], Optional["torch.Tensor"]]:
    c_x = tensor_from_numpy(bundle.structure_features[compound_indices], device)
    p_x = tensor_from_numpy(bundle.protein_embeddings[gene_indices], device)
    z_c = model.encode_compound(c_x)
    z_p = model.encode_protein(p_x)

    cp_rows = bundle.profile_rows_for_compounds(compound_indices)
    valid_cp = cp_rows >= 0
    z_cp = None
    if bool(valid_cp.all()):
        cp_x = tensor_from_numpy(bundle.profile_features_for_rows(cp_rows), device)
        cp_s = long_tensor(bundle.source_ids_for_rows(cp_rows), device)
        z_cp = model.encode_profile(cp_x, cp_s)

    gp_rows = bundle.profile_rows_for_genes(gene_indices)
    valid_gp = gp_rows >= 0
    z_gp = None
    if bool(valid_gp.all()):
        gp_x = tensor_from_numpy(bundle.profile_features_for_rows(gp_rows), device)
        gp_s = long_tensor(bundle.source_ids_for_rows(gp_rows), device)
        z_gp = model.encode_profile(gp_x, gp_s)
    return z_c, z_p, z_cp, z_gp


def compute_training_losses(
    model: "CGPAlignModel",
    bundle: CGPBundle,
    batch: CGPBatch,
    device: "torch.device",
    mode: str,
    temperature: float,
    lambdas: Dict[str, float],
    modality_dropout_prob: float,
    rng: np.random.Generator,
    score_weights: Dict[str, float],
) -> Tuple["torch.Tensor", Dict[str, float]]:
    require_torch()
    zero = torch.tensor(0.0, device=device)
    losses: Dict[str, "torch.Tensor"] = {
        "compound_profile": zero,
        "protein_profile": zero,
        "compound_protein": zero,
        "profile_bridge": zero,
        "link": zero,
    }

    active = mode_loss_mask(mode)
    drop_compound_structure = rng.random() < modality_dropout_prob
    drop_compound_profile = rng.random() < modality_dropout_prob
    drop_protein = rng.random() < modality_dropout_prob
    drop_gene_profile = rng.random() < modality_dropout_prob

    unique_c = unique_preserve(batch.compound_indices)
    unique_g = unique_preserve(batch.gene_indices)
    intrinsic_c = unique_preserve(
        batch.intrinsic_compound_indices
        if batch.intrinsic_compound_indices is not None
        else batch.compound_indices
    )
    intrinsic_g = unique_preserve(
        batch.intrinsic_gene_indices
        if batch.intrinsic_gene_indices is not None
        else batch.gene_indices
    )

    if active["compound_profile"] and not drop_compound_structure and not drop_compound_profile:
        c_mask = bundle.has_structure[intrinsic_c] & bundle.has_compound_profile[intrinsic_c] & (bundle.profile_rows_for_compounds(intrinsic_c) >= 0)
        c_sel = intrinsic_c[c_mask]
        if len(c_sel) >= 2:
            cp_rows = bundle.profile_rows_for_compounds(c_sel)
            z_c = model.encode_compound(tensor_from_numpy(bundle.structure_features[c_sel], device))
            z_cp = model.encode_profile(
                tensor_from_numpy(bundle.profile_features_for_rows(cp_rows), device),
                long_tensor(bundle.source_ids_for_rows(cp_rows), device),
            )
            losses["compound_profile"] = bidirectional_infonce(z_c, z_cp, temperature)

    if active["protein_profile"] and not drop_protein and not drop_gene_profile:
        g_mask = bundle.has_protein[intrinsic_g] & bundle.has_gene_profile[intrinsic_g] & (bundle.profile_rows_for_genes(intrinsic_g) >= 0)
        g_sel = intrinsic_g[g_mask]
        if len(g_sel) >= 2:
            gp_rows = bundle.profile_rows_for_genes(g_sel)
            z_p = model.encode_protein(tensor_from_numpy(bundle.protein_embeddings[g_sel], device))
            z_gp = model.encode_profile(
                tensor_from_numpy(bundle.profile_features_for_rows(gp_rows), device),
                long_tensor(bundle.source_ids_for_rows(gp_rows), device),
            )
            losses["protein_profile"] = bidirectional_infonce(z_p, z_gp, temperature)

    if active["compound_protein"] and not drop_compound_structure and not drop_protein:
        c_sel = unique_c[bundle.has_structure[unique_c]]
        g_sel = unique_g[bundle.has_protein[unique_g]]
        if len(c_sel) and len(g_sel):
            z_c = model.encode_compound(tensor_from_numpy(bundle.structure_features[c_sel], device))
            z_p = model.encode_protein(tensor_from_numpy(bundle.protein_embeddings[g_sel], device))
            mask = build_pos_mask(c_sel, g_sel, bundle.known_positive_pairs, device)
            losses["compound_protein"] = 0.5 * (
                multipositive_contrastive(z_c, z_p, mask, temperature)
                + multipositive_contrastive(z_p, z_c, mask.T, temperature)
            )

    if active["profile_bridge"] and not drop_compound_profile and not drop_gene_profile:
        c_mask = bundle.has_compound_profile[unique_c] & (bundle.profile_rows_for_compounds(unique_c) >= 0)
        g_mask = bundle.has_gene_profile[unique_g] & (bundle.profile_rows_for_genes(unique_g) >= 0)
        c_sel = unique_c[c_mask]
        g_sel = unique_g[g_mask]
        if len(c_sel) and len(g_sel):
            cp_rows = bundle.profile_rows_for_compounds(c_sel)
            gp_rows = bundle.profile_rows_for_genes(g_sel)
            z_cp = model.encode_profile(
                tensor_from_numpy(bundle.profile_features_for_rows(cp_rows), device),
                long_tensor(bundle.source_ids_for_rows(cp_rows), device),
            )
            z_gp = model.encode_profile(
                tensor_from_numpy(bundle.profile_features_for_rows(gp_rows), device),
                long_tensor(bundle.source_ids_for_rows(gp_rows), device),
            )
            mask = build_pos_mask(c_sel, g_sel, bundle.known_positive_pairs, device)
            losses["profile_bridge"] = multipositive_contrastive(z_cp, z_gp, mask, temperature)

    if active["link"]:
        pos_pairs = list(zip(batch.compound_indices.tolist(), batch.gene_indices.tolist()))
        neg_pairs = sample_negative_pairs(len(pos_pairs), len(bundle.compounds), len(bundle.genes), bundle.known_positive_pairs, rng)
        if pos_pairs and neg_pairs:
            pos_c = np.asarray([p[0] for p in pos_pairs], dtype=np.int64)
            pos_g = np.asarray([p[1] for p in pos_pairs], dtype=np.int64)
            neg_c = np.asarray([p[0] for p in neg_pairs], dtype=np.int64)
            neg_g = np.asarray([p[1] for p in neg_pairs], dtype=np.int64)
            z_c, z_p, z_cp, z_gp = encode_pair_components(model, bundle, pos_c, pos_g, device)
            pos_scores = pair_score(z_c, z_p, z_cp, z_gp, **score_weights)
            z_c, z_p, z_cp, z_gp = encode_pair_components(model, bundle, neg_c, neg_g, device)
            neg_scores = pair_score(z_c, z_p, z_cp, z_gp, **score_weights)
            losses["link"] = bce_link_loss(pos_scores, neg_scores)

    total = sum(lambdas[k] * losses[k] for k in losses)
    log_values = {k: float(v.detach().cpu().item()) for k, v in losses.items()}
    log_values["total"] = float(total.detach().cpu().item())
    return total, log_values


def mode_loss_mask(mode: str) -> Dict[str, bool]:
    modes = {
        "mocop_style": {
            "compound_profile": True,
            "protein_profile": False,
            "compound_protein": False,
            "profile_bridge": False,
            "link": False,
        },
        "protein_profile_only": {
            "compound_profile": False,
            "protein_profile": True,
            "compound_protein": False,
            "profile_bridge": False,
            "link": False,
        },
        "motive_style": {
            "compound_profile": False,
            "protein_profile": False,
            "compound_protein": False,
            "profile_bridge": True,
            "link": True,
        },
        "compound_protein_only": {
            "compound_profile": False,
            "protein_profile": False,
            "compound_protein": True,
            "profile_bridge": False,
            "link": True,
        },
        "cgp_no_link": {
            "compound_profile": True,
            "protein_profile": True,
            "compound_protein": True,
            "profile_bridge": True,
            "link": False,
        },
        "cgp_full": {
            "compound_profile": True,
            "protein_profile": True,
            "compound_protein": True,
            "profile_bridge": True,
            "link": True,
        },
    }
    if mode not in modes:
        raise ValueError(f"Unknown training mode: {mode}. Valid modes: {sorted(modes)}")
    return modes[mode]


def load_model_from_checkpoint(
    run_name: str,
    bundle: CGPBundle,
    checkpoint_root: Path = CHECKPOINT_DIR,
    device: Optional["torch.device"] = None,
    checkpoint_name: str = "best_model.pt",
) -> Tuple["CGPAlignModel", Dict[str, Any]]:
    require_torch()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = checkpoint_root / run_name / "config.json"
    ckpt_path = checkpoint_root / run_name / checkpoint_name
    if not config_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint/config for run_name={run_name} under {checkpoint_root}")
    config = read_json(config_path)
    model = CGPAlignModel(
        structure_dim=bundle.structure_dim,
        protein_dim=bundle.protein_dim,
        profile_dim=bundle.profile_dim,
        embed_dim=int(config.get("embed_dim", 256)),
        dropout=float(config.get("dropout", 0.1)),
        feature_dim_by_source=config.get("feature_dim_by_source") or bundle.feature_dim_by_source,
    ).to(device)
    try:
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(ckpt_path, map_location=device)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state)
    model.eval()
    return model, config


def encode_all_embeddings(
    model: "CGPAlignModel",
    bundle: CGPBundle,
    device: "torch.device",
    batch_size: int = 1024,
) -> Dict[str, np.ndarray]:
    require_torch()
    model.eval()
    out: Dict[str, List[np.ndarray]] = {"compound": [], "protein": [], "profile": []}
    with torch.no_grad():
        for start in range(0, len(bundle.compounds), batch_size):
            x = tensor_from_numpy(bundle.structure_features[start : start + batch_size], device)
            out["compound"].append(model.encode_compound(x).cpu().numpy())
        for start in range(0, len(bundle.genes), batch_size):
            x = tensor_from_numpy(bundle.protein_embeddings[start : start + batch_size], device)
            out["protein"].append(model.encode_protein(x).cpu().numpy())
        for start in range(0, len(bundle.profiles), batch_size):
            rows = np.arange(start, min(start + batch_size, len(bundle.profiles)), dtype=np.int64)
            x = tensor_from_numpy(bundle.profile_features_for_rows(rows), device)
            s = long_tensor(bundle.source_ids_for_rows(rows), device)
            out["profile"].append(model.encode_profile(x, s).cpu().numpy())
    return {k: np.vstack(v).astype(np.float32) if v else np.zeros((0, 0), dtype=np.float32) for k, v in out.items()}


def retrieval_ranks_single_positive(
    query: np.ndarray,
    gallery: np.ndarray,
    positive_indices: np.ndarray,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    ranks = np.zeros(query.shape[0], dtype=np.int64)
    gaps = np.zeros(query.shape[0], dtype=np.float32)
    for start in range(0, query.shape[0], batch_size):
        q = query[start : start + batch_size]
        sim = q @ gallery.T
        for i in range(sim.shape[0]):
            pos = int(positive_indices[start + i])
            row = sim[i]
            pos_score = row[pos]
            ranks[start + i] = int((row > pos_score).sum() + 1)
            if len(row) > 1:
                max_neg = np.max(np.delete(row, pos))
                gaps[start + i] = float(pos_score - max_neg)
            else:
                gaps[start + i] = float("nan")
    return ranks, gaps


def metrics_from_ranks(ranks: Sequence[int], gaps: Optional[Sequence[float]] = None, prefix: str = "") -> Dict[str, Any]:
    arr = np.asarray(ranks, dtype=np.float64)
    if arr.size == 0:
        return {
            f"{prefix}num_queries": 0,
            f"{prefix}Recall@1": None,
            f"{prefix}Recall@5": None,
            f"{prefix}Recall@10": None,
            f"{prefix}Recall@50": None,
            f"{prefix}MRR": None,
            f"{prefix}mean_rank": None,
            f"{prefix}median_rank": None,
            f"{prefix}positive_negative_cosine_gap": None,
        }
    out = {
        f"{prefix}num_queries": int(arr.size),
        f"{prefix}Recall@1": float(np.mean(arr <= 1)),
        f"{prefix}Recall@5": float(np.mean(arr <= 5)),
        f"{prefix}Recall@10": float(np.mean(arr <= 10)),
        f"{prefix}Recall@50": float(np.mean(arr <= 50)),
        f"{prefix}Hit@1": float(np.mean(arr <= 1)),
        f"{prefix}Hit@5": float(np.mean(arr <= 5)),
        f"{prefix}Hit@10": float(np.mean(arr <= 10)),
        f"{prefix}Hit@50": float(np.mean(arr <= 50)),
        f"{prefix}MRR": float(np.mean(1.0 / arr)),
        f"{prefix}mean_rank": float(np.mean(arr)),
        f"{prefix}median_rank": float(np.median(arr)),
    }
    if gaps is not None and len(gaps):
        out[f"{prefix}positive_negative_cosine_gap"] = float(np.nanmean(np.asarray(gaps, dtype=np.float32)))
    else:
        out[f"{prefix}positive_negative_cosine_gap"] = None
    return out


def ranks_for_multi_positive_scores(
    scores: np.ndarray,
    positives: Dict[int, set[int]],
    query_indices: Sequence[int],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    ranks: List[int] = []
    rows: List[Dict[str, Any]] = []
    for i, qid in enumerate(query_indices):
        pos_set = positives.get(int(qid), set())
        if not pos_set:
            continue
        row = scores[i]
        best_rank = min(int((row > row[p]).sum() + 1) for p in pos_set if p < len(row))
        ranks.append(best_rank)
        rows.append({"query_index": int(qid), "rank": int(best_rank), "num_positives": int(len(pos_set))})
    return np.asarray(ranks, dtype=np.int64), rows


def score_compound_gene_matrix(
    encoded: Dict[str, np.ndarray],
    bundle: CGPBundle,
    compound_indices: np.ndarray,
    gene_indices: np.ndarray,
    score_mode: str = "combined",
    alpha: float = 0.4,
    beta: float = 0.3,
    gamma: float = 0.2,
    delta: float = 0.1,
    chunk_size: int = 256,
) -> np.ndarray:
    z_c_all = encoded["compound"]
    z_p_all = encoded["protein"]
    z_prof = encoded["profile"]
    cp_rows = bundle.profile_rows_for_compounds(compound_indices)
    gp_rows = bundle.profile_rows_for_genes(gene_indices)
    z_gp = z_prof[gp_rows]
    z_p = z_p_all[gene_indices]
    out = np.zeros((len(compound_indices), len(gene_indices)), dtype=np.float32)
    for start in range(0, len(compound_indices), chunk_size):
        cidx = compound_indices[start : start + chunk_size]
        z_c = z_c_all[cidx]
        z_cp = z_prof[cp_rows[start : start + chunk_size]]
        if score_mode == "structure_only":
            scores = z_c @ z_p.T
        elif score_mode == "compound_profile_only":
            scores = z_cp @ z_p.T
        elif score_mode == "profile_bridge_only":
            scores = z_cp @ z_gp.T
        elif score_mode == "compound_gene_profile_only":
            scores = z_c @ z_gp.T
        elif score_mode == "combined":
            scores = (
                alpha * (z_c @ z_p.T)
                + beta * (z_cp @ z_gp.T)
                + gamma * (z_c @ z_gp.T)
                + delta * (z_cp @ z_p.T)
            )
        else:
            raise ValueError(f"Unknown score_mode: {score_mode}")
        out[start : start + len(cidx)] = scores.astype(np.float32)
    return out


def build_positive_dict_from_pairs(pairs: Sequence[Tuple[int, int]]) -> Dict[int, set[int]]:
    out: Dict[int, set[int]] = {}
    for c, g in pairs:
        out.setdefault(int(c), set()).add(int(g))
    return out


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    out: Dict[str, Any] = {"num_examples": int(len(labels))}
    if len(np.unique(labels)) < 2:
        out.update({"AUROC": None, "AUPRC": None, "accuracy_at_0": None})
        return out
    try:
        from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

        out["AUROC"] = float(roc_auc_score(labels, scores))
        out["AUPRC"] = float(average_precision_score(labels, scores))
        out["accuracy_at_0"] = float(accuracy_score(labels, scores >= 0.0))
    except Exception as exc:
        out["metric_error"] = str(exc)
        out["AUROC"] = None
        out["AUPRC"] = None
        out["accuracy_at_0"] = None
    return out


def collect_split_query_pairs(bundle: CGPBundle, split_name: str, fold: str = "test") -> Tuple[np.ndarray, np.ndarray, Dict[int, set[int]]]:
    pairs = bundle.split_pairs(split_name, fold)
    positives = build_positive_dict_from_pairs(pairs)
    compounds = np.asarray(sorted(positives.keys()), dtype=np.int64)
    genes = np.arange(len(bundle.genes), dtype=np.int64)
    return compounds, genes, positives


def write_dataframe(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--protein_embedding_dir", type=Path, default=PROTEIN_EMB_DIR)
    parser.add_argument("--output_root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--seed", type=int, default=13)


def checkpoint_exists(run_name: str, checkpoint_root: Path = CHECKPOINT_DIR) -> bool:
    return (checkpoint_root / run_name / "best_model.pt").exists() and (checkpoint_root / run_name / "config.json").exists()


def resolve_protein_embedding_dir(
    run_name: str,
    requested_dir: Optional[Path] = None,
    checkpoint_root: Path = CHECKPOINT_DIR,
) -> Path:
    if requested_dir is not None:
        return requested_dir
    config_path = checkpoint_root / run_name / "config.json"
    if config_path.exists():
        config = read_json(config_path)
        value = config.get("protein_embedding_dir")
        if value:
            return Path(value)
    return PROTEIN_EMB_DIR


def save_final_report_summary(output_root: Path = OUTPUT_ROOT) -> Dict[str, Any]:
    data_dir = output_root / "data"
    summary: Dict[str, Any] = {}
    dataset_path = data_dir / "dataset_summary.json"
    if dataset_path.exists():
        summary["dataset"] = read_json(dataset_path)
    for section, folder in [
        ("intrinsic_alignment", output_root / "intrinsic_alignment"),
        ("target_retrieval", output_root / "target_retrieval"),
        ("link_prediction", output_root / "link_prediction"),
        ("missing_modality", output_root / "missing_modality"),
    ]:
        section_payload = {}
        for path in sorted(folder.glob("*_metrics.json")):
            section_payload[path.stem.replace("_metrics", "")] = read_json(path)
        summary[section] = section_payload
    case_dir = output_root / "case_studies"
    summary["case_studies"] = [str(p.name) for p in sorted(case_dir.glob("*_case_studies.*"))]
    write_json(output_root / "final_report_summary.json", summary)
    return summary
