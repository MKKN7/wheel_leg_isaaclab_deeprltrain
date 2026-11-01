"""Summarize completed physics only; no inferred success from training status."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
labels={2:'0→0.25起步',4:'0.25→0停车',6:'0→0.5起步',9:'0.5→0.35减速',12:'0.35→0停车',15:'0→0.5再次起步',17:'0.5→0停车'}
rows=[];pending=[]
for d in sorted(a.root.glob('cycle_*')):
 if not d.is_dir():continue
 report=d/'retention_audit.json'
 if not report.exists():pending.append(d.name);continue
 r=json.loads(report.read_text())
 for condition,events in r['comparison'].items():
  for e in events:
   rows.append(dict(cycle=d.name,condition=condition,event=labels[e['start']],start_s=e['start'],baseline_pass=e['before'],candidate_pass=e['after'],completed=e['complete'],replicas=32,scope='seed7891 fixed20s sequence; not all skills or multiseed'))
out=dict(rows=rows,pending=pending,promoted=False,scope='Per-event deterministic physics counts; overlapping events in same rollout are NOT independent episodes.')
(a.root/'all_cycle_evidence.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
lines=['# 已完成真实验收对照','', '每格为通过数/32；同一序列内事件相关，不能相加当成独立回合。所有候选均未通过完整保留门槛，未晋升。','', '| 候选 | 延迟 | '+ ' | '.join(labels.values())+' |','|---|---|'+ '|'.join(['---']*len(labels))+'|']
for cycle in dict.fromkeys(r['cycle'] for r in rows):
 for condition in ['joint10','sensor8']:
  selected=[r for r in rows if r['cycle']==cycle and r['condition']==condition]
  lines.append('| '+cycle+' | '+condition+' | '+' | '.join(str(r['candidate_pass'])+'/32' for r in selected)+' |')
lines+=['','待完成：'+', '.join(pending),'','joint10 = obs8ms + wheel2ms/leg2ms并行；sensor8 = obs8ms、无额外轮/腿运输延迟。action额外延迟0，电机响应另计。']
(a.root/'BEHAVIOR_COMPARISON.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(completed_cycles=len(set(r['cycle'] for r in rows)),pending=pending),ensure_ascii=False))
