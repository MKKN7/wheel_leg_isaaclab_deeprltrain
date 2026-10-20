from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rsl_rl'))
import torch,pytest
from rmuc_distill_sampling import repair_pools,sample_batch

def test_filtered_offsets_and_holdout_exclusion():
    paths=[Path('/a' if i%2==0 else '/b')/f'{i}.pt' for i in range(12)]
    data=[{'observations':torch.zeros(i+1,25)} for i in range(12)]
    pools=repair_pools(paths,data,['/a','/b'],'cpu');expected=[[],[]];offset=0
    for i in range(12):
        if i==9:continue
        expected[i%2]+=list(range(offset,offset+i+1));offset+=i+1
    assert [p.tolist() for p in pools]==expected
    ids=sample_batch(offset,pools,batch_size=100)
    assert len(ids)==100 and int(ids.max())<offset
    assert torch.isin(ids[80:90],pools[0]).all()
    assert torch.isin(ids[90:],pools[1]).all()
    with pytest.raises(ValueError):repair_pools(paths,data,['/missing'],'cpu')
    with pytest.raises(ValueError):sample_batch(offset,pools,fraction=.5)
