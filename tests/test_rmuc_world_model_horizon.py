import pytest
from rmuc_world_model.horizon import require_complete_intervention

def test_baseline_bootstrap_does_not_omit_remaining_pulse():
    with pytest.raises(ValueError,match='before external pulse'):require_complete_intervention(32,.008,.4)
    require_complete_intervention(50,.008,.4)
    require_complete_intervention(64,.008,.4)

@pytest.mark.parametrize('h,dt,duration',[(True,.008,.4),(0,.008,.4),(64,0.,.4),(64,.008,float('nan'))])
def test_invalid_horizon_timing_rejected(h,dt,duration):
    with pytest.raises(ValueError):require_complete_intervention(h,dt,duration)
