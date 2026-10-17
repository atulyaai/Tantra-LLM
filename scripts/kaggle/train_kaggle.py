"""Resume the 64K continued-pretraining run on a Kaggle GPU.

Copies the read-only dataset into /kaggle/working so the trainer's relative paths
and checkpoint writes work, then resumes the same run (same optimizer/RNG/cursor).
Download /kaggle/working/tantra/runs/pretraining-20m-v1/base-training afterwards.
"""
import glob, os, shutil, subprocess, sys

src = next(os.path.dirname(p) for p in glob.glob('/kaggle/input/**/requirements.txt', recursive=True))
dst = '/kaggle/working/tantra'
if not os.path.exists(dst):
    shutil.copytree(src, dst)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tokenizers', 'psutil', 'rich'], check=False)
subprocess.run(['nvidia-smi'], check=False)
env = dict(os.environ, TANTRA_DEVICE='cuda', PYTHONUNBUFFERED='1')
cmd = [sys.executable, '-m', 'Tantra.pretrain', '--resume', '--microbatch', '16',
       '--threads', '4', '--early-stopping-patience', '10']
sys.exit(subprocess.run(cmd, cwd=dst, env=env).returncode)
