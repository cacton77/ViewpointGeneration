"""Local thumbnail rendering for catalogued parts.

This tenant publishes no per-object preview image -- `documents/{id}` offers
only the platform's generic "Physical Product" type icon, identical for every
item, and the representation objects that hold the actual geometry are not
exposed by its REST API. A catalog of identical icons is useless for picking a
part, so previews are rendered locally from the STEP file the cell has already
downloaded.

Rendering deliberately avoids Open3D's offscreen renderer and matplotlib's
mplot3d: both want a working GL/EGL context, which a headless container may
not have (and mplot3d fails to import in this image). Instead the mesh is
projected and rasterized directly -- triangles sorted back-to-front and filled
with flat Lambertian shading, which is all a 256px thumbnail needs and is
fast, deterministic, and dependency-light.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Viewing direction, a conventional three-quarter CAD view: front-left-above.
_VIEW_DIRECTION = np.array([1.0, -1.0, 0.6])
_UP = np.array([0.0, 0.0, 1.0])
# Headlight offset from the view direction, so faces facing the camera are
# bright without the whole part flattening into one tone.
_LIGHT_DIRECTION = np.array([0.9, -1.0, 1.1])


def _view_basis(direction=_VIEW_DIRECTION, up=_UP):
    """Orthonormal camera basis (right, true-up, forward) for a view direction."""
    forward = np.asarray(direction, dtype=float)
    forward /= np.linalg.norm(forward)
    reference = np.asarray(up, dtype=float)
    if abs(float(np.dot(forward, reference))) > 0.99:
        reference = np.array([0.0, 1.0, 0.0])
    right = np.cross(reference, forward)
    right /= np.linalg.norm(right)
    true_up = np.cross(forward, right)
    return right, true_up, forward


def render_mesh_thumbnail(mesh, dest, size=256, background=(0.11, 0.13, 0.16),
                          base_color=(0.62, 0.68, 0.78)):
    """Render an Open3D triangle mesh to a PNG thumbnail.

    Args:
        mesh: An `o3d.geometry.TriangleMesh`.
        dest: Output path; parent directories are created.
        size: Square image edge, in pixels.
        background: RGB background, 0-1.
        base_color: RGB surface color before shading, 0-1.

    Returns:
        tuple: (bool, str) success flag and message.
    """
    # Imported lazily and with a non-interactive backend so importing this
    # module never requires a display.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
    except ImportError as e:
        return False, f'Thumbnail rendering needs matplotlib: {e}'

    from pathlib import Path

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(vertices) == 0 or len(triangles) == 0:
        return False, 'Cannot render a thumbnail from an empty mesh.'

    right, true_up, forward = _view_basis()

    corners = vertices[triangles]                      # (n_tri, 3, 3)
    # Project to camera space: x across, y up, depth along the view direction.
    projected = np.stack([corners @ right, corners @ true_up], axis=-1)
    depth = (corners @ forward).mean(axis=1)

    # Flat shading from the geometric face normal, two-sided so inward-wound
    # triangles (common in CAD tessellations) do not render black.
    edge_a = corners[:, 1] - corners[:, 0]
    edge_b = corners[:, 2] - corners[:, 0]
    normals = np.cross(edge_a, edge_b)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals),
                        where=lengths > 0)
    light = _LIGHT_DIRECTION / np.linalg.norm(_LIGHT_DIRECTION)
    intensity = np.abs(normals @ light)
    # Lift the floor so unlit faces stay readable rather than crushing to black.
    intensity = 0.35 + 0.65 * np.clip(intensity, 0.0, 1.0)

    colors = np.clip(np.asarray(base_color)[None, :] * intensity[:, None], 0, 1)

    # Painter's algorithm: farthest first, so nearer triangles overdraw them.
    order = np.argsort(-depth)

    figure = plt.figure(figsize=(size / 100.0, size / 100.0), dpi=100)
    try:
        axes = figure.add_axes([0, 0, 1, 1])
        axes.set_facecolor(background)
        figure.patch.set_facecolor(background)
        axes.set_xticks([])
        axes.set_yticks([])
        for spine in axes.spines.values():
            spine.set_visible(False)

        collection = PolyCollection(projected[order], facecolors=colors[order],
                                    edgecolors='none', antialiased=False)
        axes.add_collection(collection)

        # Square, centred, with a small margin so the part never touches the edge.
        flat = projected.reshape(-1, 2)
        centre = (flat.max(axis=0) + flat.min(axis=0)) / 2.0
        extent = float((flat.max(axis=0) - flat.min(axis=0)).max()) or 1.0
        half = extent * 0.58
        axes.set_xlim(centre[0] - half, centre[0] + half)
        axes.set_ylim(centre[1] - half, centre[1] + half)
        axes.set_aspect('equal')

        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(dest, facecolor=background, dpi=100)
    except (OSError, ValueError) as e:
        return False, f'Thumbnail rendering failed: {e}'
    finally:
        plt.close(figure)

    return True, f'Rendered thumbnail to {dest}.'


def render_step_thumbnail(step_path, dest, units='mm', size=256):
    """Render a thumbnail directly from a STEP file.

    Returns:
        tuple: (bool, str) success flag and message.
    """
    try:
        from viewpoint_generation.step_loader import load_step
    except ImportError as e:
        return False, f'STEP thumbnail rendering unavailable: {e}'
    try:
        result = load_step(step_path, units=units)
    except Exception as e:  # noqa: BLE001 - OCC raises a wide range of errors
        return False, f'Could not load {step_path} for thumbnailing: {e}'
    return render_mesh_thumbnail(result.mesh, dest, size=size)
