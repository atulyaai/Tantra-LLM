from Tantra.data.continue_pretraining import pilot_passed


def test_continuation_requires_finished_pilot_and_both_languages_improving():
    report={'state':'pilot_complete','baseline_validation':{'hi':{'loss':10},'en':{'loss':9}},
            'latest_validation':{'hi':{'loss':8},'en':{'loss':8}}}
    assert pilot_passed(report)
    report['state']='paused_update_limit'
    assert not pilot_passed(report)
    report['state']='pilot_complete';report['latest_validation']['hi']['loss']=11
    assert not pilot_passed(report)
    report['latest_validation']['hi']['loss']=float('nan')
    assert not pilot_passed(report)
