"""Test doubles and builders for the catalog characterization tests.

Nothing here touches the network or ROS: `FakeDXClient` stands in for
`DXClient` with an in-memory tenant, and every catalog lives under pytest's
tmp_path.
"""

import copy
from pathlib import Path

from viewpoint_generation import plan_io
from viewpoint_generation.catalog.backends.dx3.client import DXClient
from viewpoint_generation.catalog.backends.dx3.config import DXConfig


def make_item(item_id, cestamp='c1', revision='A.1', space='Test Space',
              maturity='IN_WORK', title=None):
    """A normalized remote item, shaped like DXClient._normalize_item()."""
    return {
        'eng_item_id': item_id,
        'title': title or f'Part {item_id}',
        'part_number': f'prd-{item_id}',
        'revision': revision,
        'cestamp': cestamp,
        'maturity': maturity,
        'type': 'VPMReference',
        'collab_space': space,
        'description': '',
        'owner': 'tester',
        'modified': None,
        'created': None,
        'thumbnail_url': None,
    }


def eng_member(item_id):
    """A bookmark member that is an engineering item."""
    return {'id': item_id, 'type': 'VPMReference',
            'relative_path': f'/resources/v1/modeler/dseng/dseng:EngItem/{item_id}'}


def doc_member(doc_id):
    """A bookmark member that is a Document, not a part."""
    return {'id': doc_id, 'type': 'Document',
            'relative_path': f'/resources/v1/modeler/documents/{doc_id}'}


class FakeDXClient:
    """In-memory stand-in for DXClient, with the same method names.

    The search endpoint reports `totalItems` as the size of the page it just
    returned, exactly like the real tenant, so paging code cannot lean on it.
    """

    is_eng_item = staticmethod(DXClient.is_eng_item)

    def __init__(self):
        self.config = DXConfig(
            passport_url='https://passport.invalid',
            space_url='https://space.invalid/enovia',
            tenant='TENANT', username='tester', password='not-a-secret',
            security_context='ctx::Role.Org.Space')
        self.items = {}
        self.bookmarks = {}
        self.bookmark_members = {}
        self.step_files = {}
        self.search_calls = []
        self.uploads = []
        self.result_uploads = []
        self.issues = []
        self.space_predicate_unsupported = False

    def ensure_login(self):
        return True, 'Logged in.'

    def search_eng_items(self, query='*', mask=None, top=100, skip=0):
        self.search_calls.append((query, top, skip))
        if self.space_predicate_unsupported and '[ds6w:project]' in query:
            return [], 0
        page = [copy.deepcopy(item) for item in list(self.items.values())[skip:skip + top]]
        return page, len(page)

    def get_eng_item(self, item_id, mask=None):
        item = self.items.get(item_id)
        return copy.deepcopy(item) if item else None

    def find_bookmark(self, title):
        return self.bookmarks.get(title)

    def list_bookmarks(self, query='*', top=50):
        return list(self.bookmarks.values())

    def list_bookmark_items(self, bookmark_id, recursive=True, page_size=1000):
        return list(self.bookmark_members.get(bookmark_id, []))

    def get_thumbnail(self, item_id):
        return None, '', False

    def find_step_output(self, item_id):
        if item_id in self.step_files:
            return f'DO-{item_id}', {'id': f'F-{item_id}', 'format': 'STEP_AP214',
                                     'filename': f'{item_id}.stp', 'downloadable': True}
        return None, None

    def download_derived_output(self, derived_output_id, file_id, dest):
        item_id = derived_output_id[len('DO-'):]
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.step_files[item_id])
        return dest, 'Downloaded.'

    def list_related_documents(self, eng_item_id, title_prefix=None):
        return []

    def upload_inspection_plan(self, eng_item_id, plan_path, title, collab_space=None,
                               link_to_item=True):
        self.uploads.append({'eng_item_id': eng_item_id, 'path': str(plan_path),
                             'title': title, 'collab_space': collab_space,
                             'link_to_item': link_to_item})
        doc_id = f'DOC{len(self.uploads)}'
        return doc_id, f'Uploaded inspection plan as document {doc_id}.'

    def upload_result_bundle(self, eng_item_id, bundle_path, title, collab_space=None,
                             link_to_item=True):
        self.result_uploads.append({'eng_item_id': eng_item_id, 'bundle': str(bundle_path),
                                    'title': title})
        doc_id = f'RES{len(self.result_uploads)}'
        return doc_id, f'Uploaded result bundle as document {doc_id}.'

    def create_issue(self, eng_item_id, title, description, severity='Medium'):
        self.issues.append({'eng_item_id': eng_item_id, 'title': title,
                            'description': description})
        return f'ISS{len(self.issues)}', 'Created issue.'

    def close(self):
        pass


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


def producer_describe(paths):
    """The producer's side of DescribePlans, as the VG node implements it."""
    return [plan_io.describe_plan(path) for path in paths]
