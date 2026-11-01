import torch
from rmuc_world_model.retention import phase_labels,balanced_indices,phase_accepts,phase_maxima

def test_phase_labels_use_pre_action_age_boundaries():
    ages=torch.tensor([0,249,250,549,550,749,750,1549,1550,1875,2124,2125,2174,2175,2499])
    assert phase_labels(ages).tolist()==[1,1,0,0,2,2,0,0,2,0,0,3,3,2,2]

def test_rare_early_stop_not_diluted_by_long_idle():
    labels=torch.tensor([0]*100+[1]*200+[2]*400+[3]*2)
    ids=balanced_indices(labels,8,torch.Generator().manual_seed(7))
    assert torch.bincount(labels[ids]).tolist()==[8,8,8,8]
    delta=torch.zeros(len(labels),6);delta[-1,0]=.009
    assert phase_accepts(phase_maxima(delta,labels))
    delta[101,0]=.0006
    assert not phase_accepts(phase_maxima(delta,labels))
