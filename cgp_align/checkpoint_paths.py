"""Explicit relocation of inference inputs without modifying model architecture."""
from pathlib import Path

DATA_KEYS = ('compound_data_dir', 'gene_data_dir', 'gene_protein_embedding_dir')
INITIALIZER_KEYS = ('init_compound_checkpoint', 'init_gene_checkpoint',
                    'init_orf_gene_checkpoint', 'init_crispr_gene_checkpoint')


def add_data_arguments(parser):
    for key in DATA_KEYS:
        parser.add_argument('--' + key.replace('_', '-'), dest=key, type=Path,
                            help='Override this data directory in the saved checkpoint config.')


def configure_inference(model_args, overrides=None):
    """Full state_dict is loaded strictly afterwards; initializers are unnecessary."""
    for key in INITIALIZER_KEYS:
        setattr(model_args, key, None)
    if overrides is not None:
        for key in DATA_KEYS:
            value = getattr(overrides, key, None)
            if value is not None:
                setattr(model_args, key, Path(value))
    return model_args
