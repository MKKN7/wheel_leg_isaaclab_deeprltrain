"""Standard post-report hook for RMUC experiment pipelines (CPU only)."""
import argparse,json
from pathlib import Path
from plot_rmuc_training_delivery import export
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--desktop-root',type=Path,default=Path('/home/kellen/桌面/RMUC训练曲线'));a=p.parse_args()
 if not (a.experiment/'evaluation.leg_tracking.npz').exists():raise RuntimeError('Missing physical trace; evaluation must use --record-leg-tracking')
 export(a.experiment,a.desktop_root/a.experiment.resolve().name)
