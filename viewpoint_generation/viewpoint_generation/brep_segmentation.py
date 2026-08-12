"""B-rep topology-aware segmentation.

Uses the B-rep face structure carried by a STEP file to define inspection
regions. Each analytical face -- a plane, cylinder, cone, B-spline patch --
is already a natural surface region, so this needs no seeds, no thresholds,
no GPU, and no sampling: the segmentation is exactly the one the CAD author
drew.

Two optional merge passes fold the raw face set into something better suited
to inspection planning:

* Faces below `min_face_area` (fillets, chamfers, tiny relief cuts) are merged
  into their largest edge-sharing neighbour. Left alone they would each become
  their own region and each earn their own viewpoint, which is wasteful for
  features far smaller than the camera's field of view.
* With `merge_same_type`, adjacent faces sharing an analytical surface type
  are unioned, so a cylinder split into halves by a seam becomes one region.

Presents the same `segment() -> (regions, noise_faces)` contract as
`RegionGrowing` and `PartFieldSegmentation`, where a region is a list of
triangle indices into the mesh.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BRepSegmentationConfig:
    """Configuration for B-rep-based segmentation."""

    # Faces smaller than this (m^2) are merged into an adjacent face.
    # 1e-6 m^2 is a 1mm x 1mm patch. 0 disables small-face merging.
    min_face_area: float = 1e-6

    # Union adjacent faces that share an analytical surface type.
    merge_same_type: bool = False

    # Cap on emitted regions (0 = one region per surviving B-rep face). When
    # exceeded, the smallest regions are merged into neighbours until the cap
    # is met.
    max_regions: int = 0

    def to_dict(self):
        return {
            "min_face_area": {
                "value": self.min_face_area,
                "type": "float",
                "description": "Minimum B-rep face area (m²); smaller faces merge with their largest neighbour",
                "control": "slider",
                "range": [0.0, 0.001],
            },
            "merge_same_type": {
                "value": self.merge_same_type,
                "type": "boolean",
                "description": "Merge adjacent B-rep faces of the same analytical surface type",
                "control": "toggle",
            },
            "max_regions": {
                "value": self.max_regions,
                "type": "integer",
                "description": "Maximum regions (0 = use all B-rep faces)",
                "control": "slider",
                "range": [0, 200],
            },
        }


class BRepSegmentation:
    """B-rep topology segmentation with the same interface as RegionGrowing."""

    def __init__(self, config: BRepSegmentationConfig = None):
        self.config = config or BRepSegmentationConfig()

    def segment(self, mesh, step_data):
        """Segment a tessellated STEP mesh using its B-rep face topology.

        Args:
            mesh: The Open3D TriangleMesh produced by `load_step()`. Only its
                triangle count is consulted -- the topology comes from
                step_data -- but it must be the same mesh the indices in
                step_data address.
            step_data: A `StepLoadResult` carrying brep_faces, tri_to_brep,
                and face_adjacency.

        Returns:
            tuple: (regions, noise_faces) where each region is a list of
            triangle indices into mesh.triangles, and noise_faces lists
            triangles that belong to no region.
        """
        if step_data is None:
            raise ValueError('B-rep segmentation requires STEP data. '
                             'Load a .stp/.step file first.')

        triangle_count = len(np.asarray(mesh.triangles))

        # One region per B-rep face, seeded from the loader's mapping.
        face_regions = {}
        for brep_face in step_data.brep_faces:
            if not brep_face.triangle_indices:
                continue
            face_regions[brep_face.face_id] = {
                'tris': list(brep_face.triangle_indices),
                'area': brep_face.area_m2,
                'surface_type': brep_face.surface_type,
                'brep_face_ids': [brep_face.face_id],
            }

        adjacency = step_data.face_adjacency or {}

        if self.config.min_face_area > 0:
            face_regions = self._merge_small_faces(face_regions, adjacency)

        if self.config.merge_same_type:
            face_regions = self._merge_same_type(face_regions, adjacency)

        if self.config.max_regions > 0:
            face_regions = self._enforce_max_regions(face_regions, adjacency)

        regions = []
        assigned = set()
        for region in face_regions.values():
            triangles = sorted(set(region['tris']))
            if triangles:
                regions.append(triangles)
                assigned.update(triangles)

        noise_faces = [index for index in range(triangle_count)
                       if index not in assigned]

        logger.info('B-rep segmentation: %d B-rep faces -> %d regions '
                    '(%d unassigned triangles)',
                    len(step_data.brep_faces), len(regions), len(noise_faces))
        return regions, noise_faces

    def region_surface_types(self, mesh, step_data):
        """Segment, and report each region's surface type and source faces.

        Companion to `segment()` for callers that want the CAD provenance of
        each region -- the results JSON records it so downstream stages can
        adapt to surface type (lighting angles for cylinders vs. planes, for
        instance).

        Returns:
            tuple: (regions, noise_faces, metadata) where metadata is a list
            of {'surface_type', 'brep_face_ids'} parallel to regions.
        """
        if step_data is None:
            raise ValueError('B-rep segmentation requires STEP data. '
                             'Load a .stp/.step file first.')

        face_regions = {}
        for brep_face in step_data.brep_faces:
            if not brep_face.triangle_indices:
                continue
            face_regions[brep_face.face_id] = {
                'tris': list(brep_face.triangle_indices),
                'area': brep_face.area_m2,
                'surface_type': brep_face.surface_type,
                'brep_face_ids': [brep_face.face_id],
            }

        adjacency = step_data.face_adjacency or {}
        if self.config.min_face_area > 0:
            face_regions = self._merge_small_faces(face_regions, adjacency)
        if self.config.merge_same_type:
            face_regions = self._merge_same_type(face_regions, adjacency)
        if self.config.max_regions > 0:
            face_regions = self._enforce_max_regions(face_regions, adjacency)

        triangle_count = len(np.asarray(mesh.triangles))
        regions = []
        metadata = []
        assigned = set()
        for region in face_regions.values():
            triangles = sorted(set(region['tris']))
            if not triangles:
                continue
            regions.append(triangles)
            metadata.append({
                'surface_type': region['surface_type'],
                'brep_face_ids': sorted(region['brep_face_ids']),
            })
            assigned.update(triangles)

        noise_faces = [index for index in range(triangle_count)
                       if index not in assigned]
        return regions, noise_faces, metadata

    def _absorb(self, face_regions, source_id, target_id):
        """Fold one region into another, keeping triangles, area, and provenance."""
        target = face_regions[target_id]
        source = face_regions[source_id]
        target['tris'].extend(source['tris'])
        target['area'] += source['area']
        target['brep_face_ids'].extend(source['brep_face_ids'])
        del face_regions[source_id]

    def _largest_neighbour(self, face_regions, adjacency, region_id, exclude=()):
        """The largest surviving region adjacent to region_id, or None.

        Adjacency is indexed by original B-rep face id, so a merged region is
        reachable through any of the faces it absorbed.
        """
        owner = {}
        for current_id, region in face_regions.items():
            for face_id in region['brep_face_ids']:
                owner[face_id] = current_id

        best = None
        best_area = -1.0
        for face_id in face_regions[region_id]['brep_face_ids']:
            for neighbour_face in adjacency.get(face_id, ()):  # noqa: B023
                neighbour_id = owner.get(neighbour_face)
                if neighbour_id is None or neighbour_id == region_id:
                    continue
                if neighbour_id in exclude:
                    continue
                if face_regions[neighbour_id]['area'] > best_area:
                    best = neighbour_id
                    best_area = face_regions[neighbour_id]['area']
        return best

    def _merge_small_faces(self, face_regions, adjacency):
        """Merge faces below min_face_area into their largest neighbour.

        Repeats until no undersized region has a viable neighbour, so a chain
        of small faces collapses into one region rather than stopping after a
        single pass. An undersized region with no neighbour at all (an isolated
        face) is kept -- dropping it would silently lose surface area.
        """
        threshold = self.config.min_face_area
        while True:
            small = [region_id for region_id, region in face_regions.items()
                     if region['area'] < threshold]
            if not small:
                break
            merged_any = False
            for region_id in small:
                if region_id not in face_regions:
                    continue
                target = self._largest_neighbour(face_regions, adjacency, region_id)
                if target is None:
                    continue
                self._absorb(face_regions, region_id, target)
                merged_any = True
            if not merged_any:
                break
        return face_regions

    def _merge_same_type(self, face_regions, adjacency):
        """Union adjacent regions that share an analytical surface type."""
        parent = {region_id: region_id for region_id in face_regions}

        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(first, second):
            root_a, root_b = find(first), find(second)
            if root_a != root_b:
                parent[root_b] = root_a

        owner = {}
        for region_id, region in face_regions.items():
            for face_id in region['brep_face_ids']:
                owner[face_id] = region_id

        for region_id, region in face_regions.items():
            for face_id in region['brep_face_ids']:
                for neighbour_face in adjacency.get(face_id, ()):
                    neighbour_id = owner.get(neighbour_face)
                    if neighbour_id is None or neighbour_id == region_id:
                        continue
                    if (face_regions[neighbour_id]['surface_type']
                            == region['surface_type']):
                        union(region_id, neighbour_id)

        groups = {}
        for region_id in face_regions:
            groups.setdefault(find(region_id), []).append(region_id)

        merged = {}
        for root, members in groups.items():
            merged[root] = {
                'tris': [],
                'area': 0.0,
                'surface_type': face_regions[root]['surface_type'],
                'brep_face_ids': [],
            }
            for region_id in members:
                merged[root]['tris'].extend(face_regions[region_id]['tris'])
                merged[root]['area'] += face_regions[region_id]['area']
                merged[root]['brep_face_ids'].extend(
                    face_regions[region_id]['brep_face_ids'])
        return merged

    def _enforce_max_regions(self, face_regions, adjacency):
        """Merge the smallest regions into neighbours until the cap is met."""
        while len(face_regions) > self.config.max_regions:
            smallest = min(face_regions, key=lambda rid: face_regions[rid]['area'])
            target = self._largest_neighbour(face_regions, adjacency, smallest)
            if target is None:
                # Nothing left to merge into; further shrinking would mean
                # discarding surface, so stop above the cap instead.
                logger.warning('Could not reach max_regions=%d: %d regions remain '
                               'with no adjacent neighbour to merge into.',
                               self.config.max_regions, len(face_regions))
                break
            self._absorb(face_regions, smallest, target)
        return face_regions
