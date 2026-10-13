"""Static H723 student export, gated by an explicit closed-loop acceptance report."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort
from rmuc_student_runtime import StudentActor

def main():
 p=argparse.ArgumentParser();p.add_argument('--diagnostic-only',action='store_true',help='Export explicitly uncertified artifact for sim2sim diagnostics; never deployment release');p.add_argument('--student',type=Path,required=True);p.add_argument('--acceptance',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);a=p.parse_args()
 sha=hashlib.sha256(a.student.read_bytes()).hexdigest();gate=json.loads(a.acceptance.read_text())
 assert gate['student_sha256']==sha
 if not a.diagnostic_only:
  assert gate['passed'] and gate['minimum_episodes']>=32 and gate['retention_fraction']>=.9 and not gate['survival_regressions']
 actor=StudentActor(torch.load(a.student,map_location='cpu',weights_only=False)).eval();net=copy.deepcopy(actor.net).eval()
 with torch.no_grad():
  w=net[0].weight.clone();net[0].weight.copy_(w/actor.std);net[0].bias.sub_((w*(actor.mean/actor.std)).sum(1))
 x=torch.tensor(np.load(a.inputs),dtype=torch.float32);assert x.shape==(100,25) and torch.isfinite(x).all()
 with torch.no_grad():expected=actor({'policy':x});folded=net(x)
 assert torch.allclose(expected,folded,atol=2e-5,rtol=2e-5)
 a.output.mkdir(parents=True,exist_ok=False)
 torch.jit.trace(net,x[:1]).save(str(a.output/'policy.pt'))
 torch.onnx.export(net,x[:1],str(a.output/'policy.onnx'),input_names=['observation'],output_names=['action'],opset_version=13,dynamo=False)
 model=onnx.load(a.output/'policy.onnx');onnx.checker.check_model(model);assert [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]==[1,25]
 assert set(n.op_type for n in model.graph.node)<= {'Gemm','Relu'}
 session=ort.InferenceSession(str(a.output/'policy.onnx'),providers=['CPUExecutionProvider']);actual=np.concatenate([session.run(None,{'observation':z[None].numpy()})[0] for z in x]);np.testing.assert_allclose(actual,expected.numpy(),atol=2e-5,rtol=2e-5)
 np.savez(a.output/'golden_100.npz',observations=x.numpy(),pytorch=expected.numpy(),onnx=actual)
 report=dict(diagnostic_only=a.diagnostic_only,deployable=False,student_sha256=sha,onnx_sha256=hashlib.sha256((a.output/'policy.onnx').read_bytes()).hexdigest(),parameters=20614,weight_bytes=82456,batch=1,ops=[n.op_type for n in model.graph.node],max_error=float(np.abs(actual-expected.numpy()).max()),normalizer_folded=True,cubeai_validated=False,hardware_validated=False)
 (a.output/'export_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
