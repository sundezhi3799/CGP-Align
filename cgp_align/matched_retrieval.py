"""One-positive entity-mean retrieval with frozen, unique negative candidates."""
import numpy as np
import torch


def summarize(ranks):
    ranks = np.asarray(ranks)
    return {'Recall@1': float(np.mean(ranks <= 1)), 'Recall@5': float(np.mean(ranks <= 5)),
            'Recall@10': float(np.mean(ranks <= 10)), 'MRR': float(np.mean(1.0 / ranks)),
            'mean_rank': float(ranks.mean()), 'median_rank': float(np.median(ranks))}


def evaluate(query, gallery, candidates, device='cpu', batch_size=256, ratios=(100, 1000)):
    """Aligned diagonal positives; ties are broken by ascending gallery row index."""
    assert query.shape == gallery.shape and candidates.shape[1] == len(query)
    assert np.isfinite(query).all() and np.isfinite(gallery).all()
    q = torch.nn.functional.normalize(torch.as_tensor(query, device=device), dim=1)
    g = torch.nn.functional.normalize(torch.as_tensor(gallery, device=device), dim=1)
    n = len(q)
    assert all(0 < k <= candidates.shape[2] for k in ratios)
    full = np.empty(n, dtype=np.int32)
    sampled = {k: np.empty((len(candidates), n), dtype=np.int32) for k in ratios}
    columns = torch.arange(n, device=device)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            ids = torch.arange(start, stop, device=device)
            sim = q[start:stop] @ g.T
            positive = sim[torch.arange(stop-start, device=device), ids].unsqueeze(1)
            better = (sim > positive) | ((sim == positive) & (columns.unsqueeze(0) < ids.unsqueeze(1)))
            full[start:stop] = (1 + better.sum(1)).cpu().numpy()
            for repeat in range(len(candidates)):
                indices = torch.as_tensor(np.array(candidates[repeat,start:stop]), device=device, dtype=torch.long)
                wins = better.gather(1, indices)
                for k in ratios: sampled[k][repeat,start:stop] = (1 + wins[:,:k].sum(1)).cpu().numpy()
    result = dict(full_gallery=summarize(full), sampled={})
    for k, ranks in sampled.items():
        repeats = [summarize(r) for r in ranks]
        result['sampled'][str(k)] = {metric: {'mean': float(np.mean([r[metric] for r in repeats])),
                                              'repeat_sd_ddof0': float(np.std([r[metric] for r in repeats]))}
                                      for metric in repeats[0]}
    return result, full, sampled
