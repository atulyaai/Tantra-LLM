import json
d = json.load(open("evals/eval_20261012_203346.json", encoding="utf-8"))
print("router_acc:", d.get("router_acc"))
print()
for name, v in d["experts"].items():
    print(f"{name:24s} base {v['loss_base']:.3f}  expert {v['loss_expert']:.3f}  gain {v['gain_pct']:6.1f}%  router {v['router_acc']*100:.0f}%")
