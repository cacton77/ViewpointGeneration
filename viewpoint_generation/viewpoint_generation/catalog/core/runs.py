"""Inspection runs and their result bundles.

A run is one execution of a plan on a part. Its result bundle is a directory
(`results/{storage_key}/{run_id}/`) of images, point clouds and a
`result_manifest.json`; uploading it stores the whole directory as one remote
attachment, and raises an issue on the part when the worst anomaly exceeds
`ncr_threshold` and the backend can raise issues.

Mixed into `Catalog`; uses its `db`, `paths`, `config` and backends.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from viewpoint_generation.catalog.backends.base import (BackendError, Capability,
                                                        part_from_row)
from viewpoint_generation.catalog.core.schema import utc_now

MANIFEST = 'result_manifest.json'


def _file_kind(path):
    """Classify a result-bundle file for the manifest's file list."""
    suffix = path.suffix.lower()
    if suffix in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'):
        return 'image'
    if suffix in ('.ply', '.pcd', '.xyz', '.las'):
        return 'point_cloud'
    if suffix == '.json':
        return 'metadata'
    return 'other'


class RunsMixin:
    """Inspection-run bookkeeping for `Catalog`."""

    def start_run(self, backend_name, part_id, run_id=None, plan_id=None):
        """Record the start of an inspection run.

        Returns:
            tuple: (run_id or None, str) the run identifier and a message.
        """
        if self.db.get_part(backend_name, part_id) is None:
            return None, f'Unknown part: {backend_name}/{part_id}.'
        if run_id is None:
            run_id = 'run_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        if plan_id is None:
            current = self.db.get_current_plan(backend_name, part_id)
            plan_id = current['plan_id'] if current else None
        self.db.create_run({'run_id': run_id, 'backend': backend_name, 'part_id': part_id,
                            'plan_id': plan_id, 'started_at': utc_now(),
                            'status': 'running'})
        self.db.log_sync('run_started', backend_name, part_id, run_id)
        return run_id, f'Started inspection run {run_id}.'

    def result_dir_for(self, run):
        """The bundle directory of a run (where its files belong)."""
        part = self.db.get_part(run['backend'], run['part_id']) or {}
        if run.get('result_bundle_path'):
            return Path(run['result_bundle_path'])
        return self.paths.result_dir_for(part.get('storage_key', run['part_id']), run['run_id'])

    def build_result_manifest(self, run_id, anomalies=None, summary=None):
        """Write the result_manifest.json for a run's bundle.

        Returns:
            tuple: (Path or None, str) the manifest path and a message.
        """
        run = self.db.get_run(run_id)
        if run is None:
            return None, f'Unknown inspection run: {run_id}.'
        part = self.db.get_part(run['backend'], run['part_id']) or {}
        plan = self.db.get_plan(run['plan_id']) if run.get('plan_id') else None

        bundle = self.result_dir_for(run)
        try:
            bundle.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return None, f'Could not create result bundle directory: {e}'

        files = [{'path': str(path.relative_to(bundle)), 'type': _file_kind(path),
                  'size_bytes': path.stat().st_size}
                 for path in sorted(bundle.rglob('*'))
                 if path.is_file() and path.name != MANIFEST]
        anomalies = anomalies or []
        scores = [a.get('score', 0.0) for a in anomalies]
        manifest = {
            'envelope_version': '2.0',
            'type': 'inspection_result',
            'plm_context': {
                'backend': run['backend'],
                'part_id': run['part_id'],
                'part_number': part.get('part_number'),
                'revision': part.get('revision'),
                'change_token': part.get('change_token'),
                'plan_ref': plan.get('remote_ref') if plan else None,
                'result_ref': None,
            },
            'run_context': {
                'run_id': run_id,
                'cell_id': self.config.cell_id,
                'started_at': run.get('started_at'),
                'completed_at': run.get('completed_at') or utc_now(),
                'operator': self._account(run['backend']),
            },
            'summary': summary or {
                'viewpoints_executed': run.get('viewpoints_executed'),
                'viewpoints_skipped': run.get('viewpoints_skipped') or 0,
                'anomalies_detected': len(anomalies),
                'max_anomaly_score': max(scores) if scores else 0.0,
                'overall_result': run.get('overall_result') or ('FAIL' if anomalies else 'PASS'),
                'total_files': len(files),
            },
            'anomalies': anomalies,
            'files': files,
        }
        path = bundle / MANIFEST
        try:
            with open(path, 'w') as handle:
                json.dump(manifest, handle, indent=2)
        except OSError as e:
            return None, f'Could not write result manifest: {e}'

        self.db.update_run(run_id, result_bundle_path=str(bundle),
                           anomalies_found=len(anomalies),
                           max_anomaly_score=max(scores) if scores else 0.0,
                           overall_result=manifest['summary']['overall_result'],
                           completed_at=manifest['run_context']['completed_at'],
                           status='completed')
        return path, f'Wrote result manifest to {path}.'

    def upload_results(self, run_id):
        """Upload a run's result bundle, raising an issue above the NCR threshold.

        Returns:
            tuple: (remote ref id or None, str) and a message.
        """
        run = self.db.get_run(run_id)
        if run is None:
            return None, f'Unknown inspection run: {run_id}.'
        part = self.db.get_part(run['backend'], run['part_id'])
        if part is None:
            return None, f"Unknown part: {run['backend']}/{run['part_id']}."
        bundle = run.get('result_bundle_path')
        if not bundle or not Path(bundle).exists():
            return None, (f'Run {run_id} has no result bundle on disk; '
                          'build one with build_result_manifest() first.')
        backend = self.backend(run['backend'])
        if backend is None or Capability.ATTACH not in backend.capabilities:
            return None, (f"{run['backend']} does not store files remotely; the result "
                          'bundle stays on this cell.')

        self.db.update_run(run_id, upload_status='uploading')
        title = self._title('result', part, run_id)
        ref, message = None, ''
        connected, connect_message = self._connect(backend)
        if not connected:
            message = f'Result upload failed: {connect_message}'
        else:
            try:
                ref = backend.attach_file(part_from_row(part), Path(bundle),
                                          'result_bundle', title)
                if ref is None:
                    message = f'{backend.name} could not store the result bundle.'
            except BackendError as e:
                message = f'Result upload failed: {e}'
        if ref is None:
            self.db.update_run(run_id, upload_status='failed')
            self.db.log_sync('result_upload_failed', run['backend'], run['part_id'], message)
            return None, message

        self.db.update_run(run_id, remote_ref=ref.ref_id, upload_status='uploaded')
        self.db.log_sync('result_uploaded', run['backend'], run['part_id'], ref.ref_id)
        message = ref.note or f'Uploaded result bundle as {ref.ref_id}.'

        max_score = run.get('max_anomaly_score') or 0.0
        if run.get('anomalies_found') and max_score > self.config.ncr_threshold:
            message = f'{message} {self._raise_issue(backend, part, run_id)}'
        return ref.ref_id, message

    def _raise_issue(self, backend, part, run_id):
        """Raise an issue for a failing run, when the backend can; returns a message."""
        if Capability.ISSUES not in backend.capabilities:
            return f'{backend.name} cannot raise issues; none raised.'
        run = self.db.get_run(run_id)
        title = (f"{self.config.title_prefix}Inspection anomaly: "
                 f"{part.get('part_number')} {part.get('revision')}")
        try:
            ref = backend.raise_issue(part_from_row(part), title,
                                      self._format_anomaly_summary(run))
        except BackendError as e:
            return f'Issue creation failed: {e}'
        if ref is None:
            return f'{backend.name} could not raise an issue.'
        self.db.update_run(run_id, issue_ref=ref.ref_id)
        self.db.log_sync('issue_created', part['backend'], part['part_id'], ref.ref_id)
        return f'Raised issue {ref.ref_id}.'

    def _format_anomaly_summary(self, run):
        """Human-readable anomaly summary for an issue description."""
        return (f"Automated surface inspection on cell {self.config.cell_id} "
                f"found {run.get('anomalies_found')} anomaly/anomalies "
                f"(max score {run.get('max_anomaly_score')}). "
                f"Run {run['run_id']} started {run.get('started_at')}, "
                f"result {run.get('overall_result')}. "
                f"Full result bundle: {run.get('remote_ref')}.")
