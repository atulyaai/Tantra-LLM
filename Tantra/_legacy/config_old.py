"""
Tantra Config Loader
=====================
Load taxonomy and build configurations.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path


@dataclass
class DomainConfig:
    name: str
    count: int
    rank: int
    params_per_expert: int
    experts: List[str]


def load_taxonomy(taxonomy_path: str) -> Dict:
    """Load taxonomy from YAML file."""
    with open(taxonomy_path, 'r') as f:
        return yaml.safe_load(f)


def parse_taxonomy(taxonomy: Dict) -> Dict[str, DomainConfig]:
    """Parse taxonomy into DomainConfig objects."""
    domains = {}
    for domain_name, info in taxonomy.items():
        if not isinstance(info, dict):
            continue
        if "experts" not in info or "count" not in info:
            continue
        
        domains[domain_name] = DomainConfig(
            name=domain_name,
            count=info["count"],
            rank=info.get("rank", 32),
            params_per_expert=info.get("params_per_expert", 50000),
            experts=info["experts"],
        )
    return domains


def get_training_order(taxonomy: Dict) -> List[str]:
    """Get expert training order by domain priority."""
    domain_priority = [
        "conversation", "safety", "tasks", "reasoning",
        "knowledge", "agentic", "tool_calling", "structured_output",
        "rag", "code_agent", "translation", "multilingual", "long_context"
    ]
    
    order = []
    for domain in domain_priority:
        if domain in taxonomy and "experts" in taxonomy[domain]:
            order.extend(taxonomy[domain]["experts"])
    return order


def get_domain_config_for_router(taxonomy: Dict) -> Dict[str, int]:
    """Get domain -> num_experts mapping for hierarchical router."""
    config = {}
    for domain, info in taxonomy.items():
        if "count" in info:
            config[domain] = info["count"]
    return config


def get_adapter_ranks(taxonomy: Dict) -> Dict[str, int]:
    """Get expert_name -> rank mapping for adapter layer."""
    ranks = {}
    for domain, info in taxonomy.items():
        if "experts" in info and "rank" in info:
            for expert in info["experts"]:
                ranks[expert] = info["rank"]
    return ranks


def calculate_totals(taxonomy: Dict) -> Dict[str, int]:
    """Calculate total parameters and experts."""
    total_experts = 0
    total_adapter_params = 0
    
    for domain, info in taxonomy.items():
        if "experts" not in info:
            continue
        count = info.get("count", 0)
        params = info.get("params_per_expert", 0)
        total_experts += count
        total_adapter_params += count * params
    
    return {
        "total_experts": total_experts,
        "total_adapter_params": total_adapter_params,
        "base_model_params": 540_000_000,  # 1B - adapters - router - reserve
        "router_params": 2_000_000,
        "reserve_params": 1_000_000_000 - 540_000_000 - total_adapter_params - 2_000_000,
        "grand_total": 1_000_000_000,
    }


def validate_taxonomy(taxonomy: Dict) -> List[str]:
    """Validate taxonomy structure."""
    errors = []
    
    for domain, info in taxonomy.items():
        if not isinstance(info, dict):
            errors.append(f"{domain}: not a dict")
            continue
        
        if "count" not in info:
            errors.append(f"{domain}: missing 'count'")
        if "rank" not in info:
            errors.append(f"{domain}: missing 'rank'")
        if "params_per_expert" not in info:
            errors.append(f"{domain}: missing 'params_per_expert'")
        if "experts" not in info:
            errors.append(f"{domain}: missing 'experts'")
        else:
            if len(info["experts"]) != info.get("count", 0):
                errors.append(f"{domain}: expert count mismatch ({len(info['experts'])} vs {info.get('count')})")
            
            for expert in info["experts"]:
                if not expert.replace("_", "").isalnum():
                    errors.append(f"{domain}.{expert}: invalid name (must be alphanumeric + underscore)")
    
    return errors


def print_taxonomy_summary(taxonomy: Dict) -> None:
    """Print human-readable taxonomy summary."""
    totals = calculate_totals(taxonomy)
    
    print("=" * 70)
    print("Tantra EXPERT TAXONOMY SUMMARY")
    print("=" * 70)
    
    for domain, info in taxonomy.items():
        if "experts" not in info:
            continue
        print(f"\n{domain.upper()} ({info['count']} experts, rank {info['rank']}, {info['params_per_expert']:,} params each)")
        for i, expert in enumerate(info["experts"]):
            print(f"  {i+1:2d}. {expert}")
    
    print("\n" + "-" * 70)
    print("TOTALS:")
    print(f"  Total Experts:        {totals['total_experts']}")
    print(f"  Adapter Params:       {totals['total_adapter_params']:,} ({totals['total_adapter_params']/1e6:.1f}M)")
    print(f"  Base Model (frozen):  {totals['base_model_params']:,} ({totals['base_model_params']/1e6:.0f}M)")
    print(f"  Router Params:        {totals['router_params']:,}")
    print(f"  Reserve:              {totals['reserve_params']:,} ({totals['reserve_params']/1e6:.0f}M)")
    print(f"  GRAND TOTAL:          {totals['grand_total']:,} ({totals['grand_total']/1e9:.1f}B)")
    print("=" * 70)


# Compatibility shims for old base model checkpoints (Model/best.pt)
class _LegacyConfig:
    """Legacy config class - allows unpickling old model checkpoints."""
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
    def __setstate__(self, state):
        self.__dict__.update(state)
    def __getstate__(self):
        return self.__dict__


def __getattr__(name):
    """Auto-create legacy config classes on demand for checkpoint unpickling."""
    if name.endswith("Config") or name.endswith("State"):
        cls = type(name, (_LegacyConfig,), {})
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    import sys
    
    path = sys.argv[1] if len(sys.argv) > 1 else "Tantra/config/taxonomy.yaml"
    taxonomy = load_taxonomy(path)
    
    errors = validate_taxonomy(taxonomy)
    if errors:
        print("VALIDATION ERRORS:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    
    print_taxonomy_summary(taxonomy)