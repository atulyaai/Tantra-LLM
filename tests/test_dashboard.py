from Tantra.dashboard import speed,spark,render
from rich.console import Console
import io


def test_dashboard_speed_uses_real_timestamps():
    assert speed([]) is None
    assert speed([{'timestamp':10,'target_tokens':100},{'timestamp':12,'target_tokens':500}])==200
    assert spark([1,1,1])=='...'


def test_dashboard_renders_empty_stage_without_claiming_training(tmp_path):
    stream=io.StringIO();console=Console(file=stream,width=110,color_system=None)
    console.print(render(tmp_path))
    output=stream.getvalue()
    assert 'WORKER STOPPED' in output and 'pending' in output
    assert 'paused' in output and 'stopped' in output
    assert 'No live training worker' in output


def test_loss_only_probe_does_not_hide_latest_generated_answers(tmp_path):
    import json
    directory=tmp_path/'base-training';directory.mkdir()
    records=[{'update':200,'responses':[{'language':'en','question':'Who made you?','answer':'A recorded reply'}]},
             {'update':250,'responses':[{'language':'en','question':'Who made you?','answer_loss':7}]}]
    (directory/'probe_history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    stream=io.StringIO();console=Console(file=stream,width=110,color_system=None)
    console.print(render(tmp_path))
    assert 'A recorded reply' in stream.getvalue() and 'UPDATE 200' in stream.getvalue()
