"""Fit project-level profile corrections exclusively on supplied training rows."""
import numpy as np
import pandas as pd


def fit_transform(x, replicates, train_rows, kind, passes=2):
    """Preserve historical correction order; return fitted parameters for replay."""
    x = np.asarray(x, dtype=np.float32)
    train_rows = np.asarray(train_rows, dtype=np.int64)
    if len(train_rows) == 0 or len(np.unique(train_rows)) != len(train_rows):
        raise ValueError('Training rows must be nonempty and unique')
    if len(x) != len(replicates) or train_rows.min() < 0 or train_rows.max() >= len(x):
        raise ValueError('Invalid feature/replicate/training-row alignment')
    modalities = replicates['perturbation_modality'].astype(str).to_numpy() if kind == 'gene' else np.repeat('compound', len(x))
    params = {'fit_feature_rows': train_rows.copy()}
    corrected = np.empty_like(x)
    steps = []
    for index, modality in enumerate(sorted(set(modalities))):
        fit = train_rows[modalities[train_rows] == modality]
        apply = np.flatnonzero(modalities == modality)
        if not len(fit): raise ValueError('No training data for modality ' + modality)
        mean = np.nanmean(x[fit], axis=0).astype(np.float32)
        std = np.nanstd(x[fit], axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError('A feature has no finite training statistic')
        corrected[apply] = (x[apply] - mean) / std
        params[f'zscore_{index}_modality'] = np.array(modality)
        params[f'zscore_{index}_mean'], params[f'zscore_{index}_std'] = mean, std
        steps.append(dict(step='zscore', modality=modality, fit_rows=len(fit)))
    np.nan_to_num(corrected, copy=False, nan=0., posinf=0., neginf=0.)
    for cycle in range(passes):
        for column in ('Metadata_Plate', 'Metadata_Well'):
            labels = replicates[column].astype(str).to_numpy()
            fit_groups = pd.DataFrame({'label': labels[train_rows]}).groupby('label', sort=True).indices
            apply_groups = pd.DataFrame({'label': labels}).groupby('label', sort=True).indices
            keys = list(fit_groups)
            means = np.stack([corrected[train_rows[fit_groups[key]]].mean(axis=0) for key in keys]).astype(np.float32)
            for key, mean in zip(keys, means): corrected[apply_groups[key]] -= mean
            prefix = f'pass{cycle + 1}_{column}'
            params[prefix + '_keys'] = np.asarray(keys, dtype=str)
            params[prefix + '_means'] = means
            steps.append(dict(step=prefix, fit_rows=len(train_rows), fit_labels=len(keys),
                              unseen_rows=sum(len(v) for k, v in apply_groups.items() if k not in fit_groups)))
            np.nan_to_num(corrected, copy=False, nan=0., posinf=0., neginf=0.)
    return corrected, params, steps
