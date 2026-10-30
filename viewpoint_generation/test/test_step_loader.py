"""Characterization tests for the STEP loader on a generated box.

The fixture is written by OpenCascade itself at test time, so no CAD file
(and no tenant part) is committed. Skipped where OCP is not installed.
"""

import numpy as np
import pytest

OCP = pytest.importorskip('OCP')


@pytest.fixture(scope='module')
def box_step(tmp_path_factory):
    """A 10 x 20 x 30 mm box exported as STEP AP214."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Interface import Interface_Static
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    path = tmp_path_factory.mktemp('step') / 'box.stp'
    Interface_Static.SetCVal_s('write.step.unit', 'MM')
    writer = STEPControl_Writer()
    writer.Transfer(BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape(), STEPControl_AsIs)
    assert writer.Write(str(path)) == IFSelect_RetDone
    return path


def test_box_loads_with_six_planar_faces_in_metres(box_step):
    from viewpoint_generation.step_loader import load_step

    result = load_step(str(box_step), units='mm')  # units is ignored for STEP

    assert result.units == 'm'
    assert result.face_count() == 6
    assert result.surface_type_counts() == {'plane': 6}
    extent = np.asarray(result.mesh.get_axis_aligned_bounding_box().get_extent())
    np.testing.assert_allclose(sorted(extent), [0.01, 0.02, 0.03], atol=1e-6)


def test_tri_to_brep_is_consistent_after_the_weld(box_step):
    from viewpoint_generation.step_loader import load_step

    result = load_step(str(box_step))
    triangles = np.asarray(result.mesh.triangles)
    vertices = np.asarray(result.mesh.vertices)

    assert len(result.tri_to_brep) == len(triangles)
    # Every triangle belongs to exactly one face, and the per-face lists agree.
    owned = np.concatenate([face.triangle_indices for face in result.brep_faces])
    assert sorted(owned.tolist()) == list(range(len(triangles)))
    for face in result.brep_faces:
        assert set(result.tri_to_brep[face.triangle_indices].tolist()) == {face.face_id}
    # The weld leaves a box with its 8 corners shared across faces.
    assert len(vertices) == 8
    # Each face of a box touches four others.
    assert all(len(neighbours) == 4 for neighbours in result.face_adjacency.values())


def test_read_mesh_file_dispatches_step_and_reports_metres(box_step):
    from viewpoint_generation.mesh_utils import mesh_file_units, read_mesh_file

    mesh, error = read_mesh_file(str(box_step))
    assert error == ''
    assert len(mesh.triangles) > 0
    assert mesh_file_units(str(box_step), 'mm') == 'm'
    assert mesh_file_units('/x/part.stl', 'mm') == 'mm'
