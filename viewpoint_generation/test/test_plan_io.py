"""Characterization tests for plan_io: file naming, stages, and legacy readers."""

import json
import os

from catalog_fakes import results_for

from viewpoint_generation import plan_io


def test_infer_stage_from_content():
    assert plan_io.infer_stage({'meshes': [{'regions': []}]}) == plan_io.STAGE_EMPTY
    assert plan_io.infer_stage({}) == plan_io.STAGE_EMPTY
    for stage in plan_io.STAGES:
        assert plan_io.infer_stage(results_for(stage)) == stage


def test_stage_rank_orders_known_stages_and_sinks_unknown():
    ranks = [plan_io.stage_rank(s) for s in plan_io.STAGES]
    assert ranks == sorted(ranks) == [0, 1, 2, 3]
    assert plan_io.stage_rank('unknown') == -1
    assert plan_io.stage_rank(None) == -1


def test_summarize_and_describe():
    summary = plan_io.summarize(results_for('projected', regions=3, clusters_per_region=2))
    counts = (summary['num_regions'], summary['num_clusters'], summary['num_viewpoints'])
    assert counts == (3, 6, 6)
    assert summary['mesh_file'] == '/models/part.stp'
    assert plan_io.describe(summary) == '3 regions, 6 clusters, 6 viewpoints'
    assert plan_io.describe(plan_io.summarize(results_for('segmented', regions=4))) == '4 regions'
    assert plan_io.describe(plan_io.summarize({})) == 'empty'


def test_plan_filename_omits_zero_counts():
    assert plan_io.plan_filename('segmented', {'num_regions': 12}, '20260814T162427Z') == \
        'segmented_12r_20260814T162427Z.json'
    assert plan_io.plan_filename(
        'ordered', {'num_regions': 12, 'num_clusters': 75, 'num_viewpoints': 75},
        '20260814T162538Z') == 'ordered_12r_75c_75v_20260814T162538Z.json'


def test_write_and_read_each_stage(tmp_path):
    for stage in plan_io.STAGES:
        path, message = plan_io.write_plan(tmp_path, results_for(stage), stage=stage)
        assert path is not None, message
        assert os.path.basename(path).startswith(stage + '_')
        results, context, _ = plan_io.read_plan(path)
        assert results['schema_version'] == plan_io.SCHEMA_VERSION == '2.0'
        assert context['stage'] == stage
        assert results['summary']['num_regions'] == 2
        assert 'meshes' in results


def test_write_plan_never_overwrites_within_one_second(tmp_path):
    first, _ = plan_io.write_plan(tmp_path, results_for('segmented'), timestamp='T')
    second, _ = plan_io.write_plan(tmp_path, results_for('segmented'), timestamp='T')
    assert first != second
    assert second.endswith('-2.json')


def test_read_schema_2_0_sibling_form(tmp_path):
    data = results_for('clustered')
    data.update({'schema_version': '2.0',
                 'plm_context': {'eng_item_id': 'E1', 'cestamp': 'c9', 'stage': 'clustered'}})
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(data))
    results, context, message = plan_io.read_plan(path)
    assert message == 'Plan loaded.'
    assert context == {'eng_item_id': 'E1', 'cestamp': 'c9', 'stage': 'clustered'}
    assert results['meshes'][0]['regions']


def test_read_1_0_envelope_unwraps_and_infers_stage(tmp_path):
    path = tmp_path / 'envelope.json'
    path.write_text(json.dumps({'plm_context': {'eng_item_id': 'E1', 'cestamp': 'c1'},
                                'plan': results_for('projected')}))
    results, context, message = plan_io.read_plan(path)
    assert 'envelope' in message
    assert context['stage'] == 'projected'
    assert context['eng_item_id'] == 'E1'
    # The unwrapped payload carries its context so a re-save comes out in 2.0 form.
    assert results['plm_context']['eng_item_id'] == 'E1'
    assert results['schema_version'] == '2.0'


def test_read_bare_results_json(tmp_path):
    path = tmp_path / 'bare.json'
    path.write_text(json.dumps(results_for('segmented')))
    results, context, _ = plan_io.read_plan(path)
    assert results is not None
    assert context == {}


def test_read_rejects_garbage(tmp_path):
    path = tmp_path / 'bad.json'
    path.write_text('[1, 2, 3]')
    assert plan_io.read_plan(path)[0] is None
    path.write_text('not json')
    assert plan_io.read_plan(path)[0] is None
    assert plan_io.read_plan(tmp_path / 'missing.json')[0] is None


def test_stamp_adds_only_producer_keys_to_the_catalog_context(tmp_path):
    catalog_context = {'eng_item_id': 'E1', 'cestamp': 'c1', 'cell_id': 'alpha'}
    results = results_for('clustered')
    results['plm_context'] = dict(catalog_context)
    path, _ = plan_io.write_plan(tmp_path, results)
    context = json.loads(open(path).read())['plm_context']
    assert set(context) - set(catalog_context) == {'stage', 'generated_at',
                                                   'software_version'}
    assert context['stage'] == 'clustered'
    assert context['software_version'].startswith('ViewpointGeneration@')


def test_describe_plan_is_the_describeplans_contract(tmp_path):
    results = results_for('projected', regions=3, clusters_per_region=1)
    results['plm_context'] = {'eng_item_id': 'E1', 'cestamp': 'c7'}
    path, _ = plan_io.write_plan(tmp_path, results)
    description = plan_io.describe_plan(path)
    assert description['valid']
    assert (description['stage'], description['stage_rank']) == ('projected', 2)
    assert description['summary_text'] == '3 regions, 3 clusters, 3 viewpoints'
    assert json.loads(description['summary_json'])['num_regions'] == 3
    assert description['change_token'] == 'c7'
    assert description['generated_at'].endswith('Z')


def test_describe_plan_falls_back_to_mtime_and_rejects_non_plans(tmp_path):
    bare = tmp_path / 'bare.json'
    bare.write_text(json.dumps(results_for('segmented')))
    os.utime(bare, (0, 0))
    description = plan_io.describe_plan(bare)
    assert description['valid'] and description['change_token'] == ''
    assert description['generated_at'] == '1970-01-01T00:00:00Z'
    empty = tmp_path / 'empty.json'
    empty.write_text(json.dumps({'meshes': [{'regions': []}]}))
    assert not plan_io.describe_plan(empty)['valid']
    assert not plan_io.describe_plan(tmp_path / 'missing.json')['valid']


def test_describe_cli_prints_json(tmp_path, capsys):
    path, _ = plan_io.write_plan(tmp_path, results_for('ordered'))
    assert plan_io.main(['describe', path]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed[0]['stage'] == 'ordered'
    assert plan_io.main([]) == 2


def test_legacy_stage_from_name():
    assert plan_io.legacy_stage_from_name('/x/part_optimized.json') == 'ordered'
    assert plan_io.legacy_stage_from_name('part_75_clusters.json') == 'clustered'
    assert plan_io.legacy_stage_from_name('part_12_regions.json') == 'segmented'
    assert plan_io.legacy_stage_from_name('whatever.json') is None
