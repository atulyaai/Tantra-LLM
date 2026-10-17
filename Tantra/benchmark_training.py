"""Benchmark microbatches on identical 4096-target backward passes; preserve weights."""
import argparse,json,time
from pathlib import Path
import numpy as np
import psutil,torch
from Tantra.config import ModelConfig
from Tantra.model.tantra_model import TantraModel
from Tantra.data.public_sources import sha256
from Tantra.data.train_base_pilot import tensor_batch,window_order,memory_ok


def choose(results):
    safe=[r for r in results if r.get('safe') and r.get('tokens_per_second',0)>0]
    if not safe:raise MemoryError('No safe candidate')
    return max(safe,key=lambda r:r['tokens_per_second'])['microbatch']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Use a new benchmark report')
    lock=a.stage/'base-training/training.lock'
    if lock.exists() and psutil.pid_exists(int(lock.read_text())):p.error('Pause live training before benchmark')
    torch.set_num_threads(4)
    source=a.stage/'base-training/latest.pt';before=sha256(source)
    ck=torch.load(source,map_location='cpu',weights_only=False)
    model=TantraModel(ModelConfig.from_dict(ck['config']));model.load_state_dict(ck['state']);del ck
    files={lang:np.memmap(a.stage/f'corpus-19m/{lang}-train.bin',mode='r',dtype='<u4') for lang in ('hi','en')}
    positions=window_order(files)[:16];results=[];model.train()
    for batch in (1,2,4,8,16):
        if not memory_ok():break
        rss=0
        try:
            elapsed=[]
            for repeat in range(3):
                started=time.perf_counter();model.zero_grad(set_to_none=True)
                for start in range(0,16,batch):
                    if not memory_ok():raise MemoryError('Memory gate before microbatch')
                    x=tensor_batch(files,positions[start:start+batch],256)
                    loss=model.loss(x,x)*(len(x)/16)
                    if not torch.isfinite(loss):raise FloatingPointError('Non-finite benchmark loss')
                    loss.backward();rss=max(rss,psutil.Process().memory_info().rss)
                elapsed.append(time.perf_counter()-started)
            result={'microbatch':batch,'tokens_per_second':8192/sum(elapsed[1:]),'sampled_rss_bytes':rss,'safe':memory_ok(),
                    'effective_targets':4096,'timed_repeats':2,'warmup_repeats':1}
        except (MemoryError,RuntimeError) as exc:
            if isinstance(exc,RuntimeError) and 'memory' not in str(exc).lower():raise
            result={'microbatch':batch,'safe':False,'error':str(exc)}
        finally:model.zero_grad(set_to_none=True)
        results.append(result);print(json.dumps(result),flush=True)
        if not result['safe']:break
    report={'results':results,'selected_microbatch':choose(results),'checkpoint_sha256':before,
        'checkpoint_unchanged':sha256(source)==before,'threads':4,
        'scope':'Steady-state accumulated backward throughput, excluding optimizer/evaluation; no speedup promise',
        'memory_limit_gib':8,'minimum_free_system_gib':2}
    a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
