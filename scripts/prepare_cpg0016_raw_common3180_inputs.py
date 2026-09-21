from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


META_COLS = ["Metadata_Source", "Metadata_Plate", "Metadata_Well", "Metadata_JCP2022"]
CP_PREFIXES = ("Cells_", "Cytoplasm_", "Nuclei_", "Image_")


def cp_features(path: Path) -> list[str]:
    cols = pq.read_schema(path).names
    return [c for c in cols if c.startswith(CP_PREFIXES)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compound_parquet", type=Path, required=True)
    parser.add_argument("--orf_parquet", type=Path, required=True)
    parser.add_argument("--crispr_parquet", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--row_group_size", type=int, default=8192)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    compound_features = cp_features(args.compound_parquet)
    orf_features = set(cp_features(args.orf_parquet))
    crispr_features = set(cp_features(args.crispr_parquet))
    common = [c for c in compound_features if c in orf_features and c in crispr_features]
    if len(common) != len(compound_features):
        raise RuntimeError(
            f"Compound features are not all shared: compound={len(compound_features)} common={len(common)}"
        )

    gene_out = args.output_dir / "gene_orf_crispr_raw_common3180.parquet"
    columns = META_COLS + common
    writer = None
    total_rows = 0
    try:
        for path in [args.orf_parquet, args.crispr_parquet]:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=args.row_group_size, columns=columns):
                table = pa.Table.from_batches([batch])
                if writer is None:
                    writer = pq.ParquetWriter(gene_out, table.schema, compression="snappy")
                writer.write_table(table)
                total_rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    report = {
        "compound_parquet": str(args.compound_parquet),
        "orf_parquet": str(args.orf_parquet),
        "crispr_parquet": str(args.crispr_parquet),
        "gene_common_parquet": str(gene_out),
        "metadata_columns": META_COLS,
        "compound_feature_count": len(compound_features),
        "orf_feature_count": len(orf_features),
        "crispr_feature_count": len(crispr_features),
        "common_feature_count": len(common),
        "gene_common_rows_written": total_rows,
    }
    (args.output_dir / "raw_common3180_schema_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "feature_columns_common3180.json").write_text(
        json.dumps({"feature_columns": common}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
