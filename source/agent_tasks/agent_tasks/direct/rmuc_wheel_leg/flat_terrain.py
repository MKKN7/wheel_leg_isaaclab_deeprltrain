"""Flat mesh refinement with unchanged surface, material and environment origins."""
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils.configclass import configclass


def refine_flat_mesh(mesh, subdivisions):
    if not isinstance(subdivisions, int) or not 0 <= subdivisions <= 6:
        raise ValueError('flat_mesh_subdivisions must be an integer in [0, 6]')
    for _ in range(subdivisions):
        mesh = mesh.subdivide()
    return mesh


class RmucFlatTerrainImporter(TerrainImporter):
    def import_mesh(self, name, mesh):
        mesh = refine_flat_mesh(mesh, self.cfg.flat_mesh_subdivisions)
        super().import_mesh(name, mesh)


class RmucSlopeTerrainImporter(RmucFlatTerrainImporter):
    """One physical plane per run, with an independently configured slope."""
    def import_mesh(self, name, mesh):
        import numpy as np
        angle=np.deg2rad(self.cfg.recovery_slope_degrees)
        azimuth=np.deg2rad(self.cfg.recovery_slope_azimuth_degrees)
        axis=np.array([np.sin(azimuth),-np.cos(azimuth),0.])
        cross=np.array([[0.,-axis[2],axis[1]],
                        [axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
        rotation=np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*(cross@cross)
        mesh=mesh.copy()
        mesh.vertices=mesh.vertices@rotation.T
        super().import_mesh(name,mesh)


@configclass
class RmucFlatTerrainImporterCfg(TerrainImporterCfg):
    class_type = RmucFlatTerrainImporter
    # Zero reproduces the legacy large-triangle collider.
    flat_mesh_subdivisions: int = 5
