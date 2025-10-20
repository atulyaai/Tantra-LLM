from pathlib import Path


def fetch(model_name: str = "distilgpt2", out_dir: str = "Model/hf"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name)
    tok.save_pretrained(out_dir)
    mdl.save_pretrained(out_dir)
    print(f"Saved {model_name} to {out_dir}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--out", default="Model/hf")
    a = ap.parse_args()
    fetch(a.model, a.out)


