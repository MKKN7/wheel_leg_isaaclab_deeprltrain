import json,hashlib,torch,pytest
from rmuc_optimizer_resume import validate_optimizer_resume

def test_resume_requires_manifest_and_matching_hash(tmp_path):
 p=tmp_path/'checkpoint.pt';p.write_bytes(b'placeholder');m=tmp_path/'manifest.json';m.write_text(json.dumps({'checkpoint_sha256':'wrong'}))
 with pytest.raises(ValueError,match='manifest'):validate_optimizer_resume(str(p),['x'],None)
 with pytest.raises(ValueError,match='hash'):validate_optimizer_resume(str(p),['x'],str(m))

def test_resume_rejects_different_modules(tmp_path):
 p=tmp_path/'checkpoint.pt';torch.save({'optimizer_state_dict':{'param_groups':[{'rmuc_group':'physics_adaptation','rmuc_physics_modules':['old']},{'rmuc_group':'critic'}]}},p);m=tmp_path/'manifest.json';m.write_text(json.dumps({'checkpoint_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
 with pytest.raises(ValueError,match='module provenance'):validate_optimizer_resume(str(p),['new'],str(m))
