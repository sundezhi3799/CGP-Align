"""Entity representations from individually encoded biological replicates."""
import numpy as np


def encode_replicate_mean(features, feature_rows, owners, entity_count, encode, batch_size=512):
    feature_rows = np.asarray(feature_rows, dtype=np.int64)
    owners = np.asarray(owners, dtype=np.int64)
    if len(feature_rows) != len(owners) or entity_count <= 0:
        raise ValueError('Invalid replicate mapping')
    if (owners < 0).any() or (owners >= entity_count).any():
        raise ValueError('Invalid entity index')
    if (feature_rows < 0).any() or (feature_rows >= len(features)).any():
        raise ValueError('Invalid feature index')
    counts = np.bincount(owners, minlength=entity_count)
    if (counts == 0).any():
        raise ValueError('Every evaluated entity must have a replicate')
    total = None
    for start in range(0, len(owners), batch_size):
        z = np.asarray(encode(np.array(features[feature_rows[start:start + batch_size]])))
        if z.ndim != 2 or len(z) != len(owners[start:start + batch_size]) or not np.isfinite(z).all():
            raise ValueError('Invalid encoded replicates')
        if total is None:
            total = np.zeros((entity_count, z.shape[1]), dtype=np.float64)
        np.add.at(total, owners[start:start + batch_size], z)
    mean = (total / counts[:, None]).astype(np.float32)
    norms = np.linalg.norm(mean, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= 1e-12).any():
        raise ValueError('Nonfinite or zero entity representation')
    return mean / norms, counts
