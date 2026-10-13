"""Read authored spherical anchors and measure runtime separation; no mutation."""
import torch


def world_anchor(position, quaternion_xyzw, local):
    xyz=quaternion_xyzw[..., :3]
    cross=torch.linalg.cross(xyz,local,dim=-1)
    return position+local+2*(quaternion_xyzw[...,3:4]*cross+torch.linalg.cross(xyz,cross,dim=-1))


class ClosureAudit:
    def __init__(self, usd_path, robot):
        from pxr import Usd, UsdPhysics
        stage=Usd.Stage.Open(str(usd_path))
        self.robot=robot
        self.loops=[]
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.SphericalJoint):
                continue
            joint=UsdPhysics.SphericalJoint(prim)
            ids=[robot.body_names.index(rel.GetTargets()[0].name) for rel in
                 [joint.GetBody0Rel(),joint.GetBody1Rel()]]
            anchors=torch.tensor([list(joint.GetLocalPos0Attr().Get()),list(joint.GetLocalPos1Attr().Get())],
                                 device=robot.data.body_pos_w.torch.device)
            self.loops.append((prim.GetName(),ids,anchors))
        if len(self.loops)!=4:
            raise ValueError('Expected four closed-loop spherical joints')

    def sample(self):
        data=self.robot.data
        result={}
        for name,ids,anchors in self.loops:
            points=world_anchor(data.body_pos_w.torch[0,ids],data.body_quat_w.torch[0,ids],anchors)
            result[name]=float(torch.linalg.vector_norm(points[0]-points[1]))
        return result
