"""Group-safe target statistics stay inside every current training fold."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mlmom.persistence import tree_hash
from mlmom.worker import train
from tests import test_inner_validation as inner_fixture


ENCODER = '''from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import GroupKFold
MLMOM_PREPROCESSOR_V1 = True
class Encoder:
    def fit_transform(self,X,y,context):
        assert context.validation_y is None
        ids=np.asarray(X['x'],dtype=int)
        np.testing.assert_array_equal(np.asarray(y),2*ids+7)
        np.testing.assert_array_equal(context.sample_weight,ids+1)
        output=np.empty(len(X));audit=[]
        for a,b in GroupKFold(2).split(X,y,context.groups):
            assert not set(context.groups[a]) & set(context.groups[b])
            output[b]=np.mean(np.asarray(y)[a])
            audit.append(dict(train=ids[a].tolist(),valid=ids[b].tolist()))
        self.mean=float(np.mean(y))
        Path(context.work_dir,'encoding-audit.json').write_text(json.dumps(dict(train=ids.tolist(),folds=audit)))
        return X.assign(encoded=output)
    def transform(self,X):return X.assign(encoded=self.mean)
def create_preprocessor(config):return Encoder()
'''


class TargetEncodingBoundaryTests(unittest.TestCase):
    def test_group_oof_and_metadata_are_local_to_each_validation_fold(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame,context,code,job,request=inner_fixture.InnerExecutionTests().request(Path(tmp))
            frame['t0']=2*frame['x']+7
            frame['group']=np.arange(len(frame))//5
            frame['weight']=frame['x']+1
            frame.to_csv(Path(tmp)/'train.csv',index=False)
            context.task=type(context.task).model_validate(dict(context.task.model_dump(),
                validation_strategy='group_kfold',group_columns=['group'],weight_column='weight'))
            request['context']=context.model_dump(mode='json')
            candidate=code/'candidate.py'
            candidate.write_text(candidate.read_text(encoding='utf-8').replace('.fit(X,y)', '.fit(X,y,sample_weight=context.sample_weight)'),encoding='utf-8')
            (code/'preprocessing.py').write_text(ENCODER,encoding='utf-8');request['code_hash']=tree_hash(code)
            result=train(request,job)
            protocol=json.loads(Path(result['validation_protocol_ref']).read_text())
            for outer in protocol['folds']:
                audits=list((job/f"trial-0/fold-{outer['number']}").rglob('encoding-audit.json'))
                self.assertEqual(len(audits),1)
                audit=json.loads(audits[0].read_text())
                self.assertEqual(audit['train'],outer['train_indices'])
                outer_valid=set(outer['validation_indices'])
                seen=[]
                for inner in audit['folds']:
                    self.assertFalse(outer_valid & set(inner['train']))
                    self.assertFalse({i//5 for i in inner['train']} & {i//5 for i in inner['valid']})
                    seen.extend(inner['valid'])
                self.assertEqual(sorted(seen),sorted(outer['train_indices']))
            self.assertTrue(result['full_refit'])


if __name__=='__main__':unittest.main()
