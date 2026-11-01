"""Measure critic input scale using real replay, without modifying actor or WM."""
import argparse,json,torch
from pathlib import Path
from rich_replay import load_capture
from queue_schema import align_delay_queues
from replay_window import prefix_sequences
from transition_reward import step_metadata
p=argparse.ArgumentParser();p.add_argument('--command',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();torch.set_num_threads(2)
s=json.loads(a.command.read_text());v=s['command'];ck=torch.load(v[v.index('--model')+1],map_location='cpu',weights_only=False);sd=ck['model'];paths=v[v.index('--captures')+1:v.index('--batch-size')];prefix=json.loads(Path(v[v.index('--capture-prefix-config')+1]).read_text()) if '--capture-prefix-config' in v else {};seq=[]
for path in paths:
 q,meta=load_capture(path,4,33,True,True)
 if str(Path(path).resolve()) in prefix:q=prefix_sequences(q,prefix[str(Path(path).resolve())],horizon=33)
 q,meta=align_delay_queues(q,meta,ck['state_columns'][16:]);seq.extend(x for x in q if x['env']%4!=3)
x=torch.cat([torch.cat([q['state'],q['context'],step_metadata(q['context'][:,-3])],-1) for q in seq]);mean=torch.cat([sd['mean'],sd['context_mean'],torch.zeros(2)]);scale=torch.cat([sd['scale'],sd['context_scale'],torch.ones(2)]);active=x.amax(0)!=x.amin(0);z=(x-mean)/scale;z[:,~active]=0
names=ck['state_columns']+[f'context:{i}' for i in range(len(sd['context_mean']))]+['step_direction','step_size'];mx=z.abs().amax(0);ids=mx.argsort(descending=True)[:20];rows=[dict(feature=names[i],max_abs_normalized=float(mx[i]),source_scale=float(scale[i]),replay_std=float(x[:,i].std()),min=float(x[:,i].min()),max=float(x[:,i].max())) for i in ids.tolist()]
a.output.write_text(json.dumps(dict(samples=len(x),active_features=int(active.sum()),largest=rows,scope='Real training split; large normalized input is a risk indicator, not proof of divergence cause'),indent=2));print(json.dumps(rows[:10],indent=2))
