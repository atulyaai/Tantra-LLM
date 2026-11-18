from examples.train_small_sample import train_sample


def test_small_cpu_sample_trains_and_roundtrips_checkpoint(tmp_path):
    report = train_sample(tmp_path / "small_cpu", steps=35)
    assert report["device"] == "cpu"
    assert report["parameters"] > 0
    assert report["final_train_loss"] < report["initial_train_loss"]
    assert report["reload_max_abs_logit_error"] <= 1e-6
    assert (tmp_path / "small_cpu" / "base.pt").is_file()
    assert (tmp_path / "small_cpu" / "tokenizer.json").is_file()
    assert (tmp_path / "small_cpu" / "training_report.json").is_file()
