from npdna.cli import info_main


def test_info_main_prints_seed_summary(capsys):
    info_main()
    out = capsys.readouterr().out
    assert "NP-DNA seed configuration" in out
    assert "initial_vocab=" in out
