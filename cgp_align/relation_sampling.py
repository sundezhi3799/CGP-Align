"""Stable sampling seeds and mutually exclusive negative relation pairs."""
import numpy as np


def sampling_seed(base_seed, model_seed, task):
    # Preserve the published 31/37/41 offsets regardless of command-line order.
    offsets = {31: 0, 37: 1009, 41: 2018}
    offset = offsets.get(int(model_seed), int(model_seed) * 1009)
    return int(base_seed) + offset + sum(ord(c) for c in task)


def separate_negative_pairs(negatives, positives, all_pos, n_genes, rng):
    """Keep existing draws, replacing only overlaps with earlier splits."""
    arrays = [np.array(x, dtype=np.int64, copy=True) for x in negatives]
    reserved = set(all_pos)
    for array in arrays:
        pairs = list(map(tuple, array.tolist()))
        if len(set(pairs)) != len(pairs) or set(pairs) & all_pos:
            raise ValueError('Negative pairs contain positives or within-split duplicates')
        reserved.update(pairs)
    claimed = set()
    replacements = []
    for array, pos in zip(arrays, positives):
        changed = 0
        for index, pair in enumerate(map(tuple, array.tolist())):
            if pair in claimed:
                for _ in range(max(1000, len(array) * 200)):
                    candidate = (int(pos[int(rng.integers(len(pos))), 0]), int(rng.integers(n_genes)))
                    if candidate not in reserved:
                        break
                else:
                    raise RuntimeError('Insufficient unused negative pairs')
                array[index] = candidate
                reserved.add(candidate)
                pair = candidate
                changed += 1
            claimed.add(pair)
        replacements.append(changed)
    return arrays, replacements
