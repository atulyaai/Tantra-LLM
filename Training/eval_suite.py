import time
from typing import Callable, Dict, Any, List


def eval_latency(agent_run: Callable[[str], Any], prompts: List[str]) -> Dict[str, float]:
    latencies = []
    for p in prompts:
        t0 = time.time()
        agent_run(p)
        latencies.append(time.time() - t0)
    latencies.sort()
    if not latencies:
        return {"p50": 0.0, "p95": 0.0}
    p50 = latencies[len(latencies)//2]
    p95 = latencies[int(len(latencies)*0.95)-1]
    return {"p50": p50, "p95": p95}


