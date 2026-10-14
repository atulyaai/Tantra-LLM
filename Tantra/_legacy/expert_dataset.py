"""
Tantra Expert Dataset Builder
===============================
Build per-expert datasets from mixed sources:
- Manual curated (highest quality)
- Auto-classified from general corpus
- Synthetic generation from stronger model
"""

from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator

import torch
from torch.utils.data import Dataset, IterableDataset


@dataclass
class ExpertDataConfig:
    """Configuration for a single expert's data sources."""
    expert_name: str
    domain: str
    manual_path: Optional[str] = None
    auto_classify: Optional[Dict] = None  # {"source": "...", "keywords": [...], "threshold": 0.8}
    synthetic_prompts: Optional[List[str]] = None
    synthetic_n: int = 10000
    quality_rules: Optional[List[str]] = None
    max_samples: Optional[int] = None


class ExpertDataset(Dataset):
    """Dataset for a single expert."""
    
    def __init__(self, samples: List[Dict], tokenizer, seq_len: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.seq_len = seq_len
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        text = sample.get("text", "")
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # Labels = input_ids for causal LM
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "text": text,
        }


class ExpertDatasetBuilder:
    """Build per-expert datasets from mixed sources."""
    
    def __init__(
        self,
        datasets_dir: str,
        taxonomy: Dict,
        tokenizer=None,
    ):
        self.datasets_dir = Path(datasets_dir)
        self.taxonomy = taxonomy
        self.tokenizer = tokenizer
        self.expert_configs = self._build_expert_configs()
    
    def _build_expert_configs(self) -> Dict[str, ExpertDataConfig]:
        """Parse taxonomy into ExpertDataConfig objects."""
        configs = {}
        for domain, info in self.taxonomy.items():
            if "experts" not in info:
                continue
            for expert_name in info["experts"]:
                configs[expert_name] = ExpertDataConfig(
                    expert_name=expert_name,
                    domain=domain,
                    # Paths will be resolved at load time
                )
        return configs
    
    def load_expert_dataset(
        self,
        expert_name: str,
        domain: str,
    ) -> ExpertDataset:
        """Load or build dataset for a single expert."""
        
        # Check for cached dataset
        cache_path = self.datasets_dir / domain / f"{expert_name}.jsonl"
        if cache_path.exists():
            samples = self._load_jsonl(cache_path)
            if self.tokenizer:
                return ExpertDataset(samples, self.tokenizer)
            return samples
        
        # Build from sources
        config = self.expert_configs.get(expert_name)
        if not config:
            return ExpertDataset([], self.tokenizer) if self.tokenizer else []
        
        samples = []
        
        # 1. Manual curated data (highest priority)
        manual_dir = self.datasets_dir / "manual" / domain
        manual_files = list(manual_dir.glob(f"{expert_name}*.jsonl"))
        for f in manual_files:
            samples.extend(self._load_jsonl(f))
            print(f"  {expert_name}: +{len(samples)} manual samples")
        
        # 2. Auto-classified from general corpus
        if config.auto_classify:
            auto_samples = self._auto_classify(config.auto_classify, expert_name)
            samples.extend(auto_samples)
            print(f"  {expert_name}: +{len(auto_samples)} auto-classified")
        
        # 3. Synthetic generation
        if config.synthetic_prompts:
            synth_samples = self._generate_synthetic(config, expert_name)
            samples.extend(synth_samples)
            print(f"  {expert_name}: +{len(synth_samples)} synthetic")
        
        # Deduplicate and quality filter
        samples = self._deduplicate(samples)
        samples = self._quality_filter(samples, config.quality_rules)
        
        if config.max_samples:
            samples = samples[:config.max_samples]
        
        # Save cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_jsonl(samples, cache_path)
        print(f"  {expert_name}: FINAL {len(samples)} samples cached")
        
        if self.tokenizer:
            return ExpertDataset(samples, self.tokenizer)
        return samples
    
    def _load_jsonl(self, path: Path) -> List[Dict]:
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples
    
    def _save_jsonl(self, samples: List[Dict], path: Path) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
    
    def _auto_classify(self, config: Dict, expert_name: str) -> List[Dict]:
        """Classify general corpus samples into this expert."""
        # This would use a classifier - placeholder
        source_path = config.get("source")
        keywords = config.get("keywords", [])
        threshold = config.get("threshold", 0.8)
        
        if not source_path or not os.path.exists(source_path):
            return []
        
        # Simple keyword-based classification for now
        samples = []
        with open(source_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    text = item.get("text", "").lower()
                    
                    # Score by keyword matches
                    score = sum(1 for kw in keywords if kw.lower() in text) / max(1, len(keywords))
                    
                    if score >= threshold:
                        item["expert"] = expert_name
                        item["auto_score"] = score
                        samples.append(item)
                except:
                    continue
        
        return samples
    
    def _generate_synthetic(self, config: ExpertDataConfig, expert_name: str) -> List[Dict]:
        """Generate synthetic samples using a stronger model."""
        # Placeholder - would call API or local model
        prompts = config.synthetic_prompts or []
        n = config.synthetic_n
        
        samples = []
        for i in range(min(n, len(prompts) * 10)):
            prompt = random.choice(prompts)
            # In reality: call teacher model here
            # For now, create template samples
            samples.append({
                "text": f"{prompt} [SYNTHETIC GENERATION NEEDED]",
                "expert": expert_name,
                "synthetic": True,
                "prompt": prompt,
            })
        
        return samples
    
    def _deduplicate(self, samples: List[Dict]) -> List[Dict]:
        """Remove duplicate texts."""
        seen = set()
        unique = []
        for s in samples:
            text = s.get("text", "")
            if text and text not in seen:
                seen.add(text)
                unique.append(s)
        return unique
    
    def _quality_filter(self, samples: List[Dict], rules: Optional[List[str]]) -> List[Dict]:
        """Apply quality filters."""
        if not rules:
            return samples
        
        filtered = []
        for s in samples:
            text = s.get("text", "")
            ok = True
            
            for rule in rules:
                if rule == "min_length" and len(text) < 50:
                    ok = False
                elif rule == "max_length" and len(text) > 4000:
                    ok = False
                elif rule == "no_pii" and self._has_pii(text):
                    ok = False
                elif rule == "english_or_hindi" and not self._is_english_or_hindi(text):
                    ok = False
            
            if ok:
                filtered.append(s)
        
        return filtered
    
    def _has_pii(self, text: str) -> bool:
        """Simple PII detection."""
        import re
        patterns = [
            r'\b\d{10}\b',  # phone
            r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b',  # email
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # credit card
        ]
        return any(re.search(p, text) for p in patterns)
    
    def _is_english_or_hindi(self, text: str) -> bool:
        """Check if text is primarily English or Hindi."""
        # Simple heuristic
        devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097f')
        latin = sum(1 for c in text if 'a' <= c.lower() <= 'z')
        total = devanagari + latin
        return total > len(text) * 0.5 if total > 0 else False


# ── Streaming Dataset for Large Corpora ────────────────────────────────────────

class StreamingExpertDataset(IterableDataset):
    """Memory-efficient streaming dataset for large expert data."""
    
    def __init__(
        self,
        data_paths: List[str],
        tokenizer,
        seq_len: int = 512,
        shuffle_buffer: int = 10000,
    ):
        self.data_paths = data_paths
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.shuffle_buffer = shuffle_buffer
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        import random
        
        buffer = []
        
        for path in self.data_paths:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        buffer.append(item)
                        
                        if len(buffer) >= self.shuffle_buffer:
                            random.shuffle(buffer)
                            for item in buffer:
                                yield self._process_item(item)
                            buffer = []
                    except:
                        continue
        
        # Flush remaining
        if buffer:
            random.shuffle(buffer)
            for item in buffer:
                yield self._process_item(item)
    
    def _process_item(self, item: Dict) -> Dict[str, torch.Tensor]:
        text = item.get("text", "")
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[encoding["attention_mask"].squeeze(0) == 0] = -100
        return {"input_ids": input_ids, "labels": labels}


# ── Data Validation ────────────────────────────────────────────────────────────

def validate_expert_datasets(datasets_dir: str, taxonomy: Dict) -> Dict[str, Any]:
    """Validate all expert datasets exist and have sufficient samples."""
    results = {
        "total_experts": 0,
        "experts_with_data": 0,
        "experts_missing": [],
        "experts_low_samples": [],
        "sample_counts": {},
    }
    
    for domain, info in taxonomy.items():
        if "experts" not in info:
            continue
        
        for expert_name in info["experts"]:
            results["total_experts"] += 1
            cache_path = Path(datasets_dir) / domain / f"{expert_name}.jsonl"
            
            if cache_path.exists():
                samples = []
                with open(cache_path) as f:
                    samples = [line for line in f if line.strip()]
                
                count = len(samples)
                results["sample_counts"][expert_name] = count
                
                if count >= 100:
                    results["experts_with_data"] += 1
                else:
                    results["experts_low_samples"].append({
                        "expert": expert_name,
                        "count": count,
                    })
            else:
                results["experts_missing"].append(expert_name)
    
    return results


if __name__ == "__main__":
    import yaml
    import sys
    
    # Test
    with open("Tantra/config/taxonomy.yaml") as f:
        taxonomy = yaml.safe_load(f)
    
    builder = ExpertDatasetBuilder("Datasets/experts", taxonomy)
    results = validate_expert_datasets("Datasets/experts", taxonomy)
    
    print(json.dumps(results, indent=2))