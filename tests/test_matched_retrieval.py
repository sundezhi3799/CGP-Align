"""Check exact full and sampled ranks against independent stable sorting."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgp_align.matched_retrieval import evaluate


class MatchedRetrievalTest(unittest.TestCase):
    def test_rank_agreement_including_ties(self):
        rng = np.random.default_rng(31)
        n = 11
        for q, g in [(np.eye(n, dtype='float32'), np.eye(n, dtype='float32')),
                     (np.ones((n, 3), dtype='float32'), np.ones((n, 3), dtype='float32')),
                     (rng.normal(size=(n,4)).astype('float32'), rng.normal(size=(n,4)).astype('float32'))]:
            bank = np.empty((2,n,7), dtype='int32')
            for rep in range(2):
                for i in range(n): bank[rep,i] = rng.choice(np.delete(np.arange(n),i),7,replace=False)
            result, full, sampled = evaluate(q,g,bank,batch_size=4,ratios=(3,7))
            qn = q / np.linalg.norm(q,axis=1,keepdims=True)
            gn = g / np.linalg.norm(g,axis=1,keepdims=True)
            scores = qn @ gn.T
            for i in range(n):
                order = np.lexsort((np.arange(n), -scores[i]))
                self.assertEqual(full[i], np.where(order == i)[0][0]+1)
                for rep in range(2):
                    for k in (3,7):
                        ids = np.r_[i,bank[rep,i,:k]]
                        order = ids[np.lexsort((ids,-scores[i,ids]))]
                        self.assertEqual(sampled[k][rep,i],np.where(order == i)[0][0]+1)
            self.assertAlmostEqual(result['full_gallery']['MRR'], float(np.mean(1/full)))


if __name__ == '__main__': unittest.main()
