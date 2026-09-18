from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import hypergeom
except Exception:  # pragma: no cover
    hypergeom = None


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/cgp_align/paper/manuscript_cgp_align_profile"
PRISM_DIR = BASE / "prism_functional_analogue_v1"
PRISM_EMB = PRISM_DIR / "cgp_align_tri_cgp_calib_adapter_only_compound_favored_seed17_prism_overlap_embeddings.npz"
REPOSITION_EMB = BASE / "drug_repositioning_v1/repositioning_cgp_embeddings.npz"
OUT = BASE / "phenotype_gene_mechanism_interpretation_v1"


CASE_SETTING = "low_morgan_lt_0.20"
CASE_LOW_TANIMOTO = 0.20
CASE_CORR = 0.30
CASE_TOPK = 50
GENE_TOPK = 50


METHOD_ORDER = ["CGP-Align", "RDKit2D", "NYAN", "Morgan", "AtomPair", "Avalon", "Random"]
METHOD_COLORS = {
    "CGP-Align": "#0f4c81",
    "RDKit2D": "#9aa5af",
    "NYAN": "#c75b39",
    "Morgan": "#aab3bd",
    "AtomPair": "#0f8b83",
    "Avalon": "#b8c1cc",
    "Random": "#d7dde4",
}


MECHANISM_RULES: list[tuple[str, str, tuple[str, ...]]] = [
    ("DNA damage/repair", "DNA damage, replication stress and repair", ("TP53", "ATM", "ATR", "CHEK", "BRCA", "PARP", "RAD", "TOP", "WEE1")),
    ("Cell cycle/mitosis", "cell-cycle and mitotic regulation", ("CDK", "CCN", "PLK", "AURK", "BUB", "MAD", "CDC", "MCM")),
    ("Proteostasis", "chaperone and ubiquitin-proteasome stress", ("HSP", "PSMA", "PSMB", "PSMC", "PSMD", "UBE", "UBB", "UBC")),
    ("Mitochondria", "mitochondrial respiration and apoptosis", ("NDUF", "UQCR", "COX", "ATP5", "SDH", "VDAC", "CYCS", "BAX", "BAK")),
    ("Microtubule", "microtubule and cytoskeletal organization", ("TUBA", "TUBB", "MAPT", "KIF", "DYNC", "ACT", "MYH")),
    ("PI3K/AKT/mTOR", "PI3K-AKT-mTOR signalling", ("PIK3", "AKT", "MTOR", "RPTOR", "RICTOR", "PTEN")),
    ("MAPK/RTK", "receptor tyrosine kinase and MAPK signalling", ("EGFR", "ERBB", "FGFR", "PDGFR", "KIT", "MET", "MAPK", "MAP2K", "MAP3K", "RAF")),
    ("JAK/STAT", "JAK-STAT signalling", ("JAK", "STAT", "TYK2")),
    ("Apoptosis", "cell death and apoptotic signalling", ("BCL", "CASP", "FAS", "BID", "BAX", "BAK")),
    ("Lipid/transport state", "lipid remodelling, solute transport and membrane-state markers", ("ELOVL", "SLC", "AQP", "CLDN", "TSPAN", "MUC", "PTGES", "ABCB", "ABCC")),
    ("Nuclear receptor/xenobiotic", "nuclear receptor and xenobiotic metabolism", ("PPAR", "ESR", "AR", "NR", "CYP", "UGT", "SULT")),
    ("Epigenetic/chromatin", "chromatin regulation and epigenetic control", ("HDAC", "BRD", "KDM", "DNMT", "EZH", "HAT", "SMAR", "CHD")),
]


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)).astype(np.float32)


def split_tokens(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).replace(",", ";").replace("|", ";")
    out: list[str] = []
    for part in text.split(";"):
        token = part.strip().upper()
        if token and token not in {"NA", "NAN", "NONE", "NULL"}:
            out.append(token)
    return sorted(set(out))


def compact_label(value: Any, max_len: int = 26) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "..."


def gene_categories(symbol: str) -> list[str]:
    symbol = str(symbol).upper()
    hits: list[str] = []
    for name, _, prefixes in MECHANISM_RULES:
        if any(symbol.startswith(prefix) or prefix in symbol for prefix in prefixes):
            hits.append(name)
    return hits


def category_descriptions() -> pd.DataFrame:
    rows = []
    for name, desc, prefixes in MECHANISM_RULES:
        rows.append({"mechanism_category": name, "description": desc, "prefix_rules": ";".join(prefixes)})
    return pd.DataFrame(rows)


def topk_indices(score: np.ndarray, k: int) -> np.ndarray:
    k = int(min(k, score.shape[1]))
    idx = np.argpartition(-score, kth=k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(score, idx, axis=1)
    order = np.argsort(-vals, axis=1)
    return np.take_along_axis(idx, order, axis=1)


def hypergeom_tail(universe: int, k1: int, k2: int, overlap: int) -> float:
    if overlap <= 0:
        return 1.0
    if hypergeom is None:
        return float("nan")
    return float(hypergeom.sf(overlap - 1, universe, k1, k2))


def pair_key(i: int, j: int) -> tuple[int, int]:
    i = int(i)
    j = int(j)
    return (i, j) if i < j else (j, i)


def build_pair_sets() -> pd.DataFrame:
    pairs = pd.read_csv(PRISM_DIR / "prism_high_confidence_hit_pairs.csv")
    sub = pairs[
        pairs["setting"].eq(CASE_SETTING)
        & np.isclose(pairs["low_tanimoto"].astype(float), CASE_LOW_TANIMOTO)
        & np.isclose(pairs["corr_threshold"].astype(float), CASE_CORR)
        & pairs["topk"].eq(CASE_TOPK)
    ].copy()
    if sub.empty:
        raise RuntimeError("No high-confidence PRISM pair rows matched the requested setting.")
    sub["pair_a"] = sub[["query_idx", "retrieved_idx"]].min(axis=1).astype(int)
    sub["pair_b"] = sub[["query_idx", "retrieved_idx"]].max(axis=1).astype(int)
    sub = sub.sort_values(["method", "rank", "prism_response_corr"], ascending=[True, True, False])
    sub = sub.drop_duplicates(["method", "pair_a", "pair_b"]).reset_index(drop=True)
    return sub


def annotate_pair_rows(
    pair_rows: pd.DataFrame,
    meta: pd.DataFrame,
    top_genes: np.ndarray,
    gene_symbols: np.ndarray,
    gene_scores: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = int(len(gene_symbols))
    metric_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []

    cat_by_gene = {g: gene_categories(g) for g in gene_symbols}
    for rec in pair_rows.itertuples(index=False):
        i = int(rec.pair_a)
        j = int(rec.pair_b)
        gi = set(map(int, top_genes[i, :GENE_TOPK]))
        gj = set(map(int, top_genes[j, :GENE_TOPK]))
        shared = sorted(gi & gj, key=lambda x: -(gene_scores[i, x] + gene_scores[j, x]))
        union = gi | gj
        p = hypergeom_tail(universe, GENE_TOPK, GENE_TOPK, len(shared))
        shared_symbols = [str(gene_symbols[x]) for x in shared]
        query_targets = set(split_tokens(meta.loc[i, "prism_targets"]))
        retrieved_targets = set(split_tokens(meta.loc[j, "prism_targets"]))
        query_top_symbols = {str(gene_symbols[x]) for x in top_genes[i, :GENE_TOPK]}
        retrieved_top_symbols = {str(gene_symbols[x]) for x in top_genes[j, :GENE_TOPK]}
        target_hits = sorted((query_targets & query_top_symbols) | (retrieved_targets & retrieved_top_symbols))
        shared_target_hits = sorted((query_targets | retrieved_targets) & set(shared_symbols))

        cats: dict[str, int] = {}
        for g in shared_symbols:
            for cat in cat_by_gene.get(g, []):
                cats[cat] = cats.get(cat, 0) + 1
        dominant = sorted(cats.items(), key=lambda x: (-x[1], x[0]))
        dominant_text = "; ".join(f"{k}:{v}" for k, v in dominant[:4])
        for cat, count in dominant:
            category_rows.append(
                {
                    "method": rec.method,
                    "pair_a": i,
                    "pair_b": j,
                    "mechanism_category": cat,
                    "shared_gene_count": int(count),
                    "pair_shared_genes": int(len(shared)),
                }
            )

        metric = {
            "method": rec.method,
            "query_idx": int(rec.query_idx),
            "retrieved_idx": int(rec.retrieved_idx),
            "pair_a": i,
            "pair_b": j,
            "rank": int(rec.rank),
            "method_score": float(rec.method_score),
            "prism_response_corr": float(rec.prism_response_corr),
            "top_gene_k": int(GENE_TOPK),
            "shared_top_gene_count": int(len(shared)),
            "shared_top_gene_jaccard": float(len(shared) / max(1, len(union))),
            "shared_gene_hypergeom_p": p,
            "shared_gene_neglog10p": float(-math.log10(max(p, 1e-300))) if np.isfinite(p) else float("nan"),
            "num_category_hits": int(sum(cats.values())),
            "num_unique_categories": int(len(cats)),
            "dominant_categories": dominant_text,
            "query_name": meta.loc[i, "prism_name"],
            "retrieved_name": meta.loc[j, "prism_name"],
            "query_moa": meta.loc[i, "prism_moa"],
            "retrieved_moa": meta.loc[j, "prism_moa"],
            "query_targets": meta.loc[i, "prism_targets"],
            "retrieved_targets": meta.loc[j, "prism_targets"],
            "known_target_hits_top_gene": ";".join(target_hits),
            "known_target_hits_shared_gene": ";".join(shared_target_hits),
            "shared_gene_symbols_top20": ";".join(shared_symbols[:20]),
        }
        metric_rows.append(metric)

        if str(rec.method) == "CGP-Align" and int(rec.rank) <= 10:
            case_rows.append(metric)

    metric_df = pd.DataFrame(metric_rows)
    cat_df = pd.DataFrame(category_rows)
    case_df = pd.DataFrame(case_rows)
    if not case_df.empty:
        case_df = (
            case_df.sort_values(
                ["shared_gene_neglog10p", "prism_response_corr", "shared_top_gene_count"],
                ascending=[False, False, False],
            )
            .drop_duplicates(["pair_a", "pair_b"])
            .head(30)
        )
    return metric_df, cat_df, case_df


def summarize(metrics: pd.DataFrame, categories: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for method, sub in metrics.groupby("method", sort=False):
        rows.append(
            {
                "method": method,
                "num_pairs": int(len(sub)),
                "mean_prism_response_corr": float(sub["prism_response_corr"].mean()),
                "median_shared_top50_genes": float(sub["shared_top_gene_count"].median()),
                "mean_shared_top50_genes": float(sub["shared_top_gene_count"].mean()),
                "mean_shared_gene_jaccard": float(sub["shared_top_gene_jaccard"].mean()),
                "mean_shared_gene_neglog10p": float(sub["shared_gene_neglog10p"].mean()),
                "fraction_with_category_hit": float((sub["num_category_hits"] > 0).mean()),
                "mean_unique_categories": float(sub["num_unique_categories"].mean()),
                "fraction_with_known_target_hit": float(sub["known_target_hits_top_gene"].astype(str).str.len().gt(0).mean()),
            }
        )
    summary = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    summary = summary.assign(_order=summary["method"].map(order).fillna(999)).sort_values("_order").drop(columns="_order")

    if categories.empty:
        cat_summary = pd.DataFrame()
    else:
        cat_summary = (
            categories.groupby(["method", "mechanism_category"], as_index=False)
            .agg(
                num_pairs_with_category=("pair_a", "nunique"),
                total_shared_gene_category_hits=("shared_gene_count", "sum"),
            )
            .merge(metrics.groupby("method").size().rename("num_method_pairs").reset_index(), on="method", how="left")
        )
        cat_summary["pair_fraction"] = cat_summary["num_pairs_with_category"] / cat_summary["num_method_pairs"].clip(lower=1)
        cat_summary = cat_summary.sort_values(["method", "pair_fraction", "total_shared_gene_category_hits"], ascending=[True, False, False])
    return summary, cat_summary


def make_plot(summary: pd.DataFrame, cat_summary: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    order = [m for m in METHOD_ORDER if m in set(summary["method"])]
    summary = summary.set_index("method").loc[order].reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), dpi=150)

    y = np.arange(len(summary))
    colors = [METHOD_COLORS.get(m, "#9aa5af") for m in summary["method"]]
    axes[0].barh(y, summary["mean_shared_top50_genes"], color=colors, height=0.65)
    axes[0].set_yticks(y, summary["method"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Shared top-50 gene neighbours")
    axes[0].set_title("A. Gene-neighbour coherence", loc="left", fontweight="bold")

    axes[1].barh(y, summary["mean_shared_gene_neglog10p"], color=colors, height=0.65)
    axes[1].set_yticks(y, [""] * len(summary))
    axes[1].invert_yaxis()
    axes[1].set_xlabel("-log10 overlap P")
    axes[1].set_title("B. Overlap significance", loc="left", fontweight="bold")

    cgp_cat = cat_summary[cat_summary["method"].eq("CGP-Align")].copy()
    cgp_cat = cgp_cat.sort_values("pair_fraction", ascending=False).head(8)
    axes[2].barh(np.arange(len(cgp_cat)), cgp_cat["pair_fraction"], color="#0f8b83", height=0.65)
    axes[2].set_yticks(np.arange(len(cgp_cat)), cgp_cat["mechanism_category"])
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Fraction of CGP pairs")
    axes[2].set_title("C. Shared mechanism categories", loc="left", fontweight="bold")

    for ax in axes:
        ax.grid(axis="x", color="#d9dee5", lw=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=8.5)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#8b96a3")
    fig.suptitle("Gene-perturbation branch provides mechanism hypotheses for structure-diverse PRISM neighbours", x=0.02, y=1.03, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "prism_gene_mechanism_interpretation_summary.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "prism_gene_mechanism_interpretation_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, case_df: pd.DataFrame) -> None:
    cgp = summary[summary["method"].eq("CGP-Align")].iloc[0]
    rdkit = summary[summary["method"].eq("RDKit2D")].iloc[0] if "RDKit2D" in set(summary["method"]) else None
    random = summary[summary["method"].eq("Random")].iloc[0] if "Random" in set(summary["method"]) else None
    lines = [
        "# PRISM gene-mechanism interpretation experiment",
        "",
        "## Purpose",
        "",
        "This experiment explains CGP-Align retrieval results rather than establishing an independent classifier. It asks whether structure-diverse PRISM response-neighbour pairs retrieved by CGP-Align share nearby gene-perturbation neighbours in the same phenotype-anchored latent space.",
        "",
        "## Design",
        "",
        f"- Pair set: PRISM high-confidence response-neighbour pairs with Morgan Tanimoto < {CASE_LOW_TANIMOTO:.2f}, PRISM response correlation >= {CASE_CORR:.2f}, top-{CASE_TOPK} retrieval.",
        f"- Gene interpretation: each compound was linked to its top-{GENE_TOPK} nearest CGP-Align gene neighbours.",
        "- Readouts: shared top-gene count, Jaccard overlap, hypergeometric overlap P value, known PRISM target hits and lightweight mechanism-category tags.",
        "",
        "## Main result",
        "",
        f"- CGP-Align pairs: {int(cgp['num_pairs'])} unique high-confidence pairs.",
        f"- Mean shared top-50 gene neighbours: CGP-Align {cgp['mean_shared_top50_genes']:.2f}",
        f"  versus RDKit2D {rdkit['mean_shared_top50_genes']:.2f}" if rdkit is not None else "",
        f"  and random {random['mean_shared_top50_genes']:.2f}." if random is not None else "",
        f"- Mean -log10 overlap P: CGP-Align {cgp['mean_shared_gene_neglog10p']:.2f}",
        f"  versus RDKit2D {rdkit['mean_shared_gene_neglog10p']:.2f}" if rdkit is not None else "",
        f"  and random {random['mean_shared_gene_neglog10p']:.2f}." if random is not None else "",
        "",
        "## Interpretation",
        "",
        "The result supports the interpretability advantage of the four-branch design: CGP-Align does not only retrieve structure-diverse compounds with related PRISM responses; those retrieved compounds can also be connected to shared nearby gene-perturbation neighbourhoods. This provides mechanism hypotheses for follow-up, but it should be presented as computational prioritization rather than experimental mechanism confirmation.",
        "",
        "## Representative CGP-Align cases",
        "",
        case_df[
            [
                "query_name",
                "retrieved_name",
                "prism_response_corr",
                "shared_top_gene_count",
                "dominant_categories",
                "shared_gene_symbols_top20",
            ]
        ]
        .head(10)
        .to_markdown(index=False, floatfmt=".3g"),
        "",
        "## Output files",
        "",
        "- `pair_gene_mechanism_metrics.csv`",
        "- `pair_gene_mechanism_summary.csv`",
        "- `pair_mechanism_category_summary.csv`",
        "- `representative_cgp_gene_mechanism_cases.csv`",
        "- `prism_gene_mechanism_interpretation_summary.png`",
    ]
    (OUT / "prism_gene_mechanism_interpretation_report.md").write_text("\n".join([x for x in lines if x is not None]) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(PRISM_DIR / "prism_cgp_overlap_compounds.csv")
    prism_z = np.load(PRISM_EMB, allow_pickle=True)
    z_cgp = l2_normalize(prism_z["z_cgp"])
    gene_z = np.load(REPOSITION_EMB, allow_pickle=True)
    z_gene = l2_normalize(gene_z["z_gene"])
    gene_symbols = gene_z["gene_symbols"].astype(str)
    if z_cgp.shape[1] != z_gene.shape[1]:
        raise ValueError(f"Dimension mismatch: compounds {z_cgp.shape}, genes {z_gene.shape}")

    score = z_cgp @ z_gene.T
    top_genes = topk_indices(score, max(GENE_TOPK, 100))

    pairs = build_pair_sets()
    metrics, categories, cases = annotate_pair_rows(pairs, meta, top_genes, gene_symbols, score)
    summary, cat_summary = summarize(metrics, categories)

    category_descriptions().to_csv(OUT / "mechanism_category_rules.csv", index=False)
    pairs.to_csv(OUT / "input_high_confidence_pair_set.csv", index=False)
    metrics.to_csv(OUT / "pair_gene_mechanism_metrics.csv", index=False)
    categories.to_csv(OUT / "pair_mechanism_category_hits.csv", index=False)
    summary.to_csv(OUT / "pair_gene_mechanism_summary.csv", index=False)
    cat_summary.to_csv(OUT / "pair_mechanism_category_summary.csv", index=False)
    cases.to_csv(OUT / "representative_cgp_gene_mechanism_cases.csv", index=False)

    make_plot(summary, cat_summary)
    write_report(summary, cases)
    manifest = {
        "num_prism_compounds": int(len(meta)),
        "num_gene_embeddings": int(len(gene_symbols)),
        "case_setting": CASE_SETTING,
        "low_tanimoto": CASE_LOW_TANIMOTO,
        "response_corr_threshold": CASE_CORR,
        "retrieval_topk": CASE_TOPK,
        "gene_topk": GENE_TOPK,
        "output_dir": str(OUT.relative_to(ROOT)),
    }
    (OUT / "prism_gene_mechanism_interpretation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
