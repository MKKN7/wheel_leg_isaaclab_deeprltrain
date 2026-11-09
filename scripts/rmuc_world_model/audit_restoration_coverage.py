"""Report exact restoration coverage without turning weak history into mastery."""
import argparse,ast,json
from pathlib import Path

def audit(manifest_path, command_path, grids):
 manifest=json.loads(manifest_path.read_text());required={(float(x['vx']),float(x['yaw'])) for x in manifest['points']};spec=json.loads(command_path.read_text());cmd=spec['command'];cfg={}
 for value in cmd:
  if value.startswith('env.') and '=' in value:
   k,v=value[4:].split('=',1)
   try:cfg[k]=ast.literal_eval(v)
   except (ValueError,SyntaxError):cfg[k]=v
 active=int(cmd[cmd.index('--command_level')+1]);ids=set(cfg['command_rehearsal_priority_stage_ids'])|{active};pool=set()
 for i in ids:
  vx=float(cfg['command_stage_linear_x_mps'][i]);yaw=float(cfg['command_stage_yaw_rate_rad_s'][i]);pool.update([(vx,yaw),(vx,-yaw)])
 pool.update((float(row[1]),float(row[2])) for row in cfg.get('command_training_sequence',[]))
 evaluated=set()
 for grid in grids:evaluated.update(tuple(map(float,row)) for row in json.loads(grid.read_text()))
 return dict(required_points=len(required),training_declared_points=len(pool),required_in_training=len(required&pool),missing_training=sorted(required-pool),required_in_eval=len(required&evaluated),missing_evaluation=sorted(required-evaluated),all_restoration_training_covered=required<=pool,all_restoration_eval_covered=required<=evaluated,scope='Declared pools only; actual rollout counts/runtime gates and behavior required. Partial batches allowed but must not claim all-domain coverage.',promoted=False)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('manifest',type=Path);parser.add_argument('command',type=Path);parser.add_argument('output',type=Path);parser.add_argument('--grids',nargs='+',type=Path,required=True);a=parser.parse_args();result=audit(a.manifest,a.command,a.grids);a.output.write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if not isinstance(v,list)})
