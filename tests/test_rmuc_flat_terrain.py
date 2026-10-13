"""Verify refinement preserves the physical surface and rollback semantics."""
import ast
from pathlib import Path

import numpy as np
import pytest
import trimesh


def refine_function():
    path = Path(__file__).resolve().parents[1] / 'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/flat_terrain.py'
    fn = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef))
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['refine_flat_mesh']


def test_refinement_preserves_plane_area_bounds_normals_and_input():
    mesh = trimesh.Trimesh(vertices=[[-64,-64,0],[64,-64,0],[64,64,0],[-64,64,0]],
                           faces=[[0,1,2],[0,2,3]], process=False)
    original = mesh.vertices.copy()
    refined = refine_function()(mesh, 5)
    np.testing.assert_array_equal(refined.bounds, mesh.bounds)
    np.testing.assert_array_equal(mesh.vertices, original)
    assert len(mesh.faces) == 2
    assert refined.area == pytest.approx(mesh.area)
    assert np.all(refined.vertices[:, 2] == 0)
    np.testing.assert_allclose(refined.face_normals, np.tile([0,0,1], (len(refined.faces),1)))
    assert refine_function()(mesh, 0) is mesh


@pytest.mark.parametrize('value', [-1, 7, 1.5])
def test_invalid_refinement_rejected(value):
    with pytest.raises(ValueError):
        refine_function()(None, value)
