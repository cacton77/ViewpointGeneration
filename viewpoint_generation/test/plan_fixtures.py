"""Builders for plan_io tests: minimal results dicts at each pipeline stage."""


def results_for(stage, regions=2, clusters_per_region=2, mesh_file='/models/part.stp'):
    """A minimal ViewpointGeneration results dict at the given pipeline stage."""
    region_list = []
    for r in range(regions):
        region = {'id': r, 'faces': [r], 'clusters': []}
        if stage in ('clustered', 'projected', 'ordered'):
            for c in range(clusters_per_region):
                cluster = {'id': c, 'faces': [r]}
                if stage in ('projected', 'ordered'):
                    cluster['viewpoint'] = {'position': [0, 0, 1],
                                            'orientation': [0, 0, 0, 1]}
                region['clusters'].append(cluster)
        if stage == 'ordered':
            region['order'] = {'greedy': {'order': list(range(clusters_per_region))}}
        region_list.append(region)
    return {'meshes': [{'file': mesh_file, 'units': 'm', 'source_format': 'step',
                        'segmentation_algorithm': 'region_growth',
                        'regions': region_list}]}
