from Tantra.data.auto_growth import select_expert
from Tantra.data.expert_workbench import catalog


def test_growth_requires_ready_data_and_skips_attempts():
    manifest = {'experts': {name: {'state': 'needs_data'} for name in catalog()}}
    manifest['experts']['math']['state'] = 'data_ready'
    assert select_expert(manifest, set()) == 'math'
    assert select_expert(manifest, {'math'}) is None


def test_growth_defers_other_languages():
    manifest = {'experts': {'lang_fr': {'state': 'data_ready'}}}
    assert select_expert(manifest, set()) is None
