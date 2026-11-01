import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rmuc_world_model'))
from warmup_guard import check_warmup

def test_reject_divergence_and_nonfinite():
 assert not check_warmup([{'critic_loss':2000,'bootstrap_return':-150}]*10,100,50)['passed']
 assert not check_warmup([{'critic_loss':float('nan'),'bootstrap_return':0}],100,50)['passed']
 assert check_warmup([{'critic_loss':.5,'bootstrap_return':-4}]*10,100,50)['passed']

def test_recent_window_only():
 old=[{'critic_loss':1000,'bootstrap_return':-150}]*20
 recent=[{'critic_loss':1,'bootstrap_return':-5}]*10
 assert check_warmup(old+recent,100,50)['passed']
