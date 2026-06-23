"""
PartField-based mesh segmentation.

A drop-in alternative to RegionGrowing that produces the same
``segment() -> (regions, noise_points)`` contract, where each region is a list
of point indices into the sampled point cloud and ``noise_points`` is a list of
unassigned indices.

Internally this shells out to nv-tlabs/PartField (mounted at /models/PartField):

  1. ``partfield_inference.py``  -> per-face geometric feature ``.npy``
  2. ``run_part_clustering.py``  -> per-face part labels ``.npy``

PartField labels are per-face. PartField preserves the input mesh's face count
and ordering (only the vertex coordinates are normalised internally), so the
returned labels align by index with the mesh handed in. Each sampled point is
mapped to its nearest triangle to inherit that face's part label, and points are
grouped by label into regions.

Subprocess invocation keeps the PyTorch-Lightning / DDP machinery out of the ROS
process and matches PartField's intended usage.
"""

import os
import glob
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass
class PartFieldSegmentationConfig:
    """Configuration parameters for PartField segmentation."""

    # Core parameters
    num_parts: int = 12

    # Clustering strategy
    use_agglo: bool = True          # agglomerative -> spatially connected parts
    option: int = 0                 # face adjacency: 0=naive, 1=face-MST, 2=cc-MST
    with_knn: bool = False          # augment adjacency with kNN (agglo only)

    # PartField install / invocation (rarely changed; not exposed to the GUI)
    partfield_root: str = '/models/PartField'
    checkpoint: str = 'model/model_objaverse.ckpt'
    config_file: str = 'configs/final/demo.yaml'
    python_exe: str = 'python3'
    timeout_s: int = 1800

    def to_dict(self):
        return {
            "num_parts": {
                "value": self.num_parts,
                "type": "integer",
                "description": "Number of PartField parts to segment the mesh into",
                "control": "slider",
                "range": [2, 50],
            },
            "use_agglo": {
                "value": self.use_agglo,
                "type": "boolean",
                "description": "Agglomerative clustering (connected parts) instead of KMeans",
                "control": "toggle",
            },
            "option": {
                "value": self.option,
                "type": "integer",
                "description": "Face-adjacency graph for agglomerative clustering: 0=naive, 1=face-MST, 2=cc-MST",
                "control": "slider",
                "range": [0, 2],
            },
            "with_knn": {
                "value": self.with_knn,
                "type": "boolean",
                "description": "Augment the face-adjacency graph with kNN connections (agglomerative only)",
                "control": "toggle",
            },
        }


class PartFieldSegmentation:
    """PartField segmentation with a RegionGrowing-compatible interface."""

    def __init__(self, config: PartFieldSegmentationConfig = None):
        self.config = config or PartFieldSegmentationConfig()
        self.mesh = None
        self.points = None
        self.point_labels = None  # per-point part label, set after segment()

    def segment(self, point_cloud: o3d.geometry.PointCloud,
                mesh: o3d.geometry.TriangleMesh):
        """
        Segment the mesh into parts and project the labels onto the point cloud.

        Args:
            point_cloud: Sampled point cloud (same coordinate frame as ``mesh``).
            mesh: Triangle mesh the points were sampled from. Its faces must be
                  in the same order PartField sees them, which is guaranteed by
                  handing PartField a freshly exported copy of this mesh.

        Returns:
            Tuple of (regions, noise_points). ``regions`` is a list of clusters,
            each a list of point indices; ``noise_points`` is always empty since
            PartField assigns every face to a part.
        """
        start_time = time.time()

        self.mesh = mesh
        self.points = np.asarray(point_cloud.points)

        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        if len(faces) == 0:
            raise ValueError('Mesh has no triangles; cannot run PartField.')

        # Run PartField on an exported copy of this exact mesh so the returned
        # per-face labels align by index with `faces` above.
        face_labels = self._run_partfield(vertices, faces)

        if len(face_labels) != len(faces):
            raise RuntimeError(
                f'PartField returned {len(face_labels)} face labels but the mesh '
                f'has {len(faces)} faces. Mesh preprocessing/remeshing must stay '
                f'disabled for the index mapping to hold.')

        # Map each point to its nearest triangle, inherit that face's part label.
        tri_ids = self._nearest_triangle(vertices, faces, self.points)
        point_labels = face_labels[tri_ids].astype(int)
        self.point_labels = point_labels

        regions = [np.nonzero(point_labels == lab)[0].tolist()
                   for lab in np.unique(point_labels)]
        noise_points = []

        print(f'PartField segmentation: {len(regions)} parts from '
              f'{len(faces)} faces in {time.time() - start_time:.2f}s')
        return regions, noise_points

    # ------------------------------------------------------------------ #
    # PartField subprocess pipeline
    # ------------------------------------------------------------------ #
    def _run_partfield(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Run inference + clustering, return per-face labels aligned to `faces`."""
        cfg = self.config
        root = cfg.partfield_root
        if not os.path.isdir(root):
            raise RuntimeError(
                f'PartField not found at {root}. Is ./models mounted into the '
                f'container and PYTHONPATH set?')

        uid = 'mesh'
        run_name = f'vpg_{os.getpid()}_{int(time.time())}'
        data_dir = tempfile.mkdtemp(prefix='partfield_data_')
        feat_dir = os.path.join(root, 'exp_results', 'partfield_features', run_name)
        clust_dir = os.path.join(root, 'exp_results', 'clustering', run_name)

        try:
            # Stage the mesh as <data_dir>/mesh.obj for PartField to discover.
            mesh_obj = os.path.join(data_dir, f'{uid}.obj')
            staged = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(vertices),
                o3d.utility.Vector3iVector(faces))
            o3d.io.write_triangle_mesh(mesh_obj, staged,
                                       write_vertex_normals=False)

            env = dict(os.environ, PYTHONPATH=root)

            # 1) Feature extraction.
            self._check_call([
                cfg.python_exe, 'partfield_inference.py',
                '-c', cfg.config_file,
                '--opts',
                'continue_ckpt', cfg.checkpoint,
                'result_name', f'partfield_features/{run_name}',
                'dataset.data_path', data_dir,
            ], cwd=root, env=env)

            # 2) Clustering into parts. max_num_clusters must exceed num_parts so
            #    the requested level is produced (KMeans: 2..K-1, agglo: 1..K).
            k = max(cfg.num_parts, 2) + 2
            cmd = [
                cfg.python_exe, 'run_part_clustering.py',
                '--root', feat_dir,
                '--dump_dir', clust_dir,
                '--source_dir', data_dir,
                '--max_num_clusters', str(k),
                '--option', str(cfg.option),
            ]
            # NOTE: these flags use argparse type=bool, so ANY value is truthy;
            # the only way to get False is to omit the flag entirely.
            if cfg.use_agglo:
                cmd += ['--use_agglo', 'True']
                if cfg.with_knn:
                    cmd += ['--with_knn', 'True']
            self._check_call(cmd, cwd=root, env=env)

            labels = self._load_labels(clust_dir, uid, cfg.num_parts)
            return labels.reshape(-1).astype(int)
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)
            shutil.rmtree(feat_dir, ignore_errors=True)
            shutil.rmtree(clust_dir, ignore_errors=True)

    def _load_labels(self, clust_dir: str, uid: str, num_parts: int) -> np.ndarray:
        """Load the per-face label file for the requested part count."""
        cluster_out = os.path.join(clust_dir, 'cluster_out')
        target = os.path.join(cluster_out, f'{uid}_0_{num_parts:02d}.npy')
        if os.path.exists(target):
            return np.load(target)

        # Fall back to the available level closest to the request.
        available = sorted(glob.glob(os.path.join(cluster_out, f'{uid}_0_*.npy')))
        if not available:
            raise RuntimeError(
                f'PartField clustering produced no label files in {cluster_out}.')

        def level(path):
            return int(os.path.basename(path).rsplit('_', 1)[-1].split('.')[0])

        closest = min(available, key=lambda p: abs(level(p) - num_parts))
        print(f'PartField: requested {num_parts} parts not available, using '
              f'{level(closest)} ({os.path.basename(closest)}).')
        return np.load(closest)

    @staticmethod
    def _check_call(cmd, cwd, env):
        """Run a subprocess, raising with captured output on failure."""
        result = subprocess.run(cmd, cwd=cwd, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        if result.returncode != 0:
            tail = '\n'.join(result.stdout.splitlines()[-40:])
            raise RuntimeError(
                f'PartField step failed ({result.returncode}): '
                f'{" ".join(cmd)}\n{tail}')

    @staticmethod
    def _nearest_triangle(vertices: np.ndarray, faces: np.ndarray,
                          points: np.ndarray) -> np.ndarray:
        """Return the index of the nearest mesh triangle for each query point."""
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(
            o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32),
            o3d.core.Tensor(faces, dtype=o3d.core.Dtype.UInt32))
        ans = scene.compute_closest_points(
            o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32))
        return ans['primitive_ids'].numpy().astype(np.int64)
