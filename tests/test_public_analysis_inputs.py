"""Public compact inputs preserve matched identities but not full gene coverage."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgp_align.matched_inputs import validated_profile_input, sha256, query_rows
from cgp_align.profile_aggregation import encode_replicate_mean
from cgp_align.analysis_inputs import identity_digest, require_full_universe
from cgp_align.relation_sampling import sampling_seed, separate_negative_pairs


class PublicInputs(unittest.TestCase):
    def test_compact_and_full_profiles_have_identical_query_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ids = ['train', 'B', 'val', 'A']
            raw = np.array([[2., 1.], [1., 3.], [4., 2.], [3., 1.]])
            outputs = []
            audit = {'seed':41, 'test_ids':['A', 'B']}
            for compact in [False, True]:
                directory = base / str(compact); directory.mkdir()
                local_ids = ['B', 'A'] if compact else ids
                pd.DataFrame({'entity_index':[1,3] if compact else range(len(local_ids)), 'compound_id':local_ids}).to_parquet(directory/'compound_mocop_entities.parquet')
                # Compact feature rows are deliberately reordered independently of entities.
                features = raw[[3, 0, 2, 1]] if compact else raw
                owners = [1, 0, 1, 0] if compact else [1, 1, 3, 3]
                pd.DataFrame({'entity_index':owners,'feature_index':range(4)}).to_parquet(directory/'compound_mocop_replicates.parquet')
                np.save(directory/'compound_mocop_replicate_features.npy',features)
                splits = {'seed':41,'cold_compound':{'test':[0, 1] if compact else [3, 1]}}
                if compact:splits['split_index_type']='compact_test_entities'
                (directory/'splits_compound_mocop.json').write_text(json.dumps(splits))
                if compact:
                    entries=[{'path':'data/seed41/compound/'+p.name,'sha256':sha256(p)} for p in directory.iterdir()]
                    manifest=base/'manifest.json';manifest.write_text(json.dumps({'groups':[{'files':entries}]}))
                else:
                    audit.update(entity_table_sha256=sha256(directory/'compound_mocop_entities.parquet'),split_sha256=sha256(directory/'splits_compound_mocop.json'),fit_audit={'output_sha256':sha256(directory/'compound_mocop_replicate_features.npy')})
                    manifest=None
                table, rows, _, _ = validated_profile_input(directory,audit,manifest)
                self.assertEqual(table.iloc[rows].compound_id.tolist(),['A','B'])
                lookup=np.full(len(table),-1);lookup[rows]=np.arange(2)
                reps=pd.read_parquet(directory/'compound_mocop_replicates.parquet')
                outputs.append(encode_replicate_mean(features,reps.feature_index,lookup[reps.entity_index],2,lambda x:x*x)[0])
            np.testing.assert_array_equal(*outputs)
            with (directory/'compound_mocop_replicate_features.npy').open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):validated_profile_input(directory,audit,manifest)

    def test_wrong_query_identity_rejected(self):
        with self.assertRaisesRegex(ValueError,'differ'):
            query_rows(pd.DataFrame({'compound_id':['A','B']}),[0],['B'])

    def test_universe_checks_identities_not_only_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);table=pd.DataFrame({'entity_id':['G1','G2'],'gene_symbol':['A','B']})
            info={'gene':{'rows':2,'columns':list(table),'identity_sha256':identity_digest(table,list(table))}}
            config=root/'config.json';config.write_text(json.dumps(info))
            table.to_parquet(root/'gene_entities.parquet');np.save(root/'gene.npy',np.eye(2))
            require_full_universe(root,compound=False,config=config)
            table.iloc[::-1].to_parquet(root/'gene_entities.parquet')
            with self.assertRaisesRegex(ValueError,'candidate universe'):
                require_full_universe(root,compound=False,config=config)

    def test_disjoint_negatives_preserve_unaffected_pairs(self):
        original=[np.array([[0,0],[1,1]]),np.array([[0,0],[1,2]]),np.array([[1,2],[0,3]])]
        positives=[np.array([[0,8],[1,9]])]*3
        out, counts=separate_negative_pairs(original,positives,{(0,8),(1,9)},20,np.random.default_rng(123))
        self.assertEqual(counts,[0,1,1]);np.testing.assert_array_equal(out[0],original[0])
        np.testing.assert_array_equal(out[1][1],original[1][1]);np.testing.assert_array_equal(out[2][1],original[2][1])
        sets=[set(map(tuple,a.tolist())) for a in out]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertFalse(set.union(*sets)&{(0,8),(1,9)})

    def test_sampling_seed_independent_of_run_order(self):
        task='holdout_dgidb'
        batch={s:sampling_seed(13,s,task) for s in [31,37,41]}
        reverse={s:sampling_seed(13,s,task) for s in [41,37,31]}
        self.assertEqual(batch,reverse)
        self.assertEqual(sampling_seed(13,41,task),13+2018+sum(map(ord,task)))
        for index,seed in enumerate([31,37,41]):
            self.assertEqual(batch[seed],13+index*1009+sum(map(ord,task)))


if __name__=='__main__':unittest.main()
