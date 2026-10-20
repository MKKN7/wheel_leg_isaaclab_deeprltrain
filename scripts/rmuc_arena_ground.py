"""Generate isolated RMUC arena-ground MJCF candidates; never change the default model."""
import argparse,hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path

def build(source,profile_path,name,destination):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError(destination)
    spec=json.loads(Path(profile_path).read_text());p=spec['profiles'][name]
    tree=ET.parse(source);root=tree.getroot();ground=root.find("./worldbody/geom[@name='ground']")
    if ground is None:raise ValueError('Expected RMUC ground geom')
    geom_default=root.find('./default/geom');friction=ground.get('friction',geom_default.get('friction')).split()
    friction[0]=str(p['effective_dynamic_friction']);ground.set('friction',' '.join(friction))
    ground.set('rgba','0.8 0.8 0.8 1')
    contact=root.find('contact')
    if contact is None:contact=ET.SubElement(root,'contact')
    for wheel in ['left_wheel_contact','right_wheel_contact']:
        if root.find(f".//geom[@name='{wheel}']") is None:raise ValueError('Missing wheel contact')
        if any({q.get('geom1'),q.get('geom2')}=={'ground',wheel} for q in contact.findall('pair')):raise ValueError('Existing ground pair must be audited')
        # Explicit pair avoids max(ground, wheel) silently raising the worn mu.
        ET.SubElement(contact,'pair',name=f'arena_{wheel}',geom1='ground',geom2=wheel,condim='3',
            friction=' '.join([friction[0],friction[0],friction[1],friction[2],friction[2]]),
            solref=geom_default.get('solref'),solimp=geom_default.get('solimp'))
    compiler=root.find('compiler');compiler.set('meshdir',str((source.parent/compiler.get('meshdir','')).resolve()))
    destination.parent.mkdir(parents=True,exist_ok=True);tree.write(destination,encoding='utf-8',xml_declaration=True)
    return dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),output=str(destination),output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),profile=name,target_effective_dynamic_friction=p['effective_dynamic_friction'],isaac_effective_static_friction=p['effective_static_friction'],isaac_restitution_target=p['effective_restitution'],mujoco_restitution_equivalence_verified=False,solver_parameters='baseline retained',promoted=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--profile',type=Path,required=True);ap.add_argument('--name',choices=['fresh','worn'],required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();r=build(a.source,a.profile,a.name,a.output);a.output.with_suffix('.manifest.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n');print(json.dumps(r,indent=2,ensure_ascii=False))
