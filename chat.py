import sys, os, json, torch
sys.stdout.reconfigure(encoding='utf-8')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tokenizers import Tokenizer
from Tantra.utils import safe_load_checkpoint
from Tantra.model import NeuroCoreModel

SYS_PROMPT = "आप तंत्र हैं, एक सहायक, सटीक और विनम्र AI सहायक जो अतुल्य AI द्वारा बनाया गया है। हिंदी में उत्तर दें।"


def _find_latest_checkpoint():
    """Search known checkpoint locations; tolerate non-numeric filenames
    like 'checkpoint_latest.pt' (the previous sort key assumed a numeric
    step suffix and crashed on this exact filename)."""
    search_dirs = ["Model/Checkpoints", "Model/Latest", "Model/Best"]
    candidates = []
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".pt"):
                path = os.path.join(d, f)
                candidates.append((os.path.getmtime(path), path))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


found = _find_latest_checkpoint()
if not found:
    print("No checkpoints found! Training must complete at least one checkpoint save.")
    sys.exit(1)

print(f"Loading checkpoint: {found}")
ckpt = safe_load_checkpoint(found, map_location="cpu")
state_dict = ckpt.get("model_state_dict", ckpt)

# Load the tokenizer from the SAME directory as the checkpoint whenever
# possible, not a hardcoded root-level path. Model/tokenizer.json (root) and
# Model/Latest/tokenizer.json were found to disagree on the token id for
# 1,166 tokens -- using whichever tokenizer wasn't saved alongside this
# checkpoint risks silently feeding it the wrong ids.
ckpt_dir = os.path.dirname(found)
tok_path = os.path.join(ckpt_dir, "tokenizer.json")
if not os.path.exists(tok_path):
    tok_path = "Model/tokenizer.json"
    print(f"WARNING: no tokenizer.json next to {found}; falling back to "
          f"{tok_path}, which may not match this checkpoint's vocabulary.")
tok = Tokenizer.from_file(tok_path)
EOS = tok.token_to_id("[EOS]") or 2

# Build the model from the config saved INSIDE the checkpoint (train.py
# saves the actual NeuroCoreConfig used for training) rather than a
# hand-typed `config.model.dim = 512` block -- NeuroCoreConfig has no
# `.model` attribute, so that line crashed unconditionally before this fix.
config = ckpt.get("config", None)
if config is None:
    raise RuntimeError(
        f"{found} has no saved 'config' -- can't reconstruct a compatible "
        f"model architecture. Re-export this checkpoint with a version of "
        f"train.py that saves config (see CHANGELOG 'Unreleased' fix)."
    )

model = NeuroCoreModel(config, use_mtp=False, use_moe=getattr(config.moe, "num_experts", 1) > 1)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing or unexpected:
    print(f"  load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
model.eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
print(f"Checkpoint step: {ckpt.get('step', ckpt.get('step_count', 'unknown'))}")
print()

USE_SPEAK = "--speak" in sys.argv or "-s" in sys.argv

def _speak_text(text: str):
    """Play speech asynchronously using edge-tts on Windows."""
    if not text.strip():
        return
    try:
        import asyncio, edge_tts, tempfile
        async def _synth():
            comm = edge_tts.Communicate(text, voice="hi-IN-SwaraNeural")
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            await comm.save(tmp_path)
            return tmp_path

        audio_path = asyncio.run(_synth())
        # Play asynchronously using PowerShell SoundPlayer or default media tool
        cmd = f'powershell -c "(New-Object Media.SoundPlayer \'{audio_path}\').PlaySync()" 2>$null'
        os.system(cmd)
        try:
            os.unlink(audio_path)
        except Exception:
            pass
    except Exception as e:
        print(f"  [TTS warning: {e}]")

# Interactive chat
print("=" * 60)
print(f"  तंत्र चैट - Hindi LLM Interactive Mode {'[Voice ON]' if USE_SPEAK else '[Voice OFF (use --speak to enable)]'}")
print("  Type 'quit' to exit")
print("=" * 60)

while True:
    user_input = input("\nआप: ").strip()
    if user_input.lower() in ['quit', 'exit', 'बंद', 'q']:
        print("धन्यवाद! अलविदा!")
        break
    if not user_input:
        continue

    prompt = f"{SYS_PROMPT}\n\nउपयोगकर्ता: {user_input}\nसहायक:"
    ids = tok.encode(prompt).ids
    prompt_tensor = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        out = model.generate(
            prompt_tensor,
            max_new_tokens=60,
            temperature=0.3,
            top_p=0.85,
            repetition_penalty=1.3,
            eos_token_id=EOS,
        )

    gen_ids = out[0].tolist()
    new_ids = gen_ids[len(ids):]
    if EOS in new_ids:
        new_ids = new_ids[:new_ids.index(EOS)]

    response = tok.decode(new_ids).strip()
    print(f"\nतंत्र: {response}")
    if USE_SPEAK:
        _speak_text(response)
