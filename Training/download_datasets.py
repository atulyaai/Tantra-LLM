import json
import requests
from pathlib import Path
from datasets import load_dataset
import time

def download_reliable_datasets():
    """Download reliable open source datasets that work well"""
    
    datasets_to_download = [
        {
            "name": "alpaca",
            "hf_name": "tatsu-lab/alpaca",
            "split": "train",
            "text_field": "text",
            "max_items": 52000,
            "description": "Alpaca instruction-following dataset"
        },
        {
            "name": "openassistant", 
            "hf_name": "OpenAssistant/oasst1",
            "split": "train",
            "text_field": "text",
            "max_items": 84000,
            "description": "OpenAssistant conversations"
        },
        {
            "name": "ultrachat",
            "hf_name": "HuggingFaceH4/ultrachat_200k", 
            "split": "train_sft",
            "text_field": "messages",
            "max_items": 200000,
            "description": "UltraChat conversations"
        },
        {
            "name": "dolly",
            "hf_name": "databricks/databricks-dolly-15k",
            "split": "train", 
            "text_field": "instruction",
            "max_items": 15000,
            "description": "Dolly instruction dataset"
        },
        {
            "name": "wizardlm",
            "hf_name": "WizardLM/WizardLM_evol_instruct_V2_196k",
            "split": "train",
            "text_field": "conversations", 
            "max_items": 143000,
            "description": "WizardLM evolved instructions"
        },
        {
            "name": "squad",
            "hf_name": "squad",
            "split": "train",
            "text_field": "context",
            "max_items": 87000,
            "description": "SQuAD reading comprehension"
        },
        {
            "name": "gsm8k",
            "hf_name": "gsm8k",
            "split": "train",
            "text_field": "question",
            "max_items": 7500,
            "description": "GSM8K math problems"
        }
    ]
    
    total_samples = 0
    
    for dataset_info in datasets_to_download:
        print(f"\n📥 Downloading {dataset_info['description']}...")
        
        try:
            # Load dataset
            ds = load_dataset(
                dataset_info["hf_name"], 
                split=dataset_info["split"], 
                streaming=True
            )
            
            output_file = f"Dataset/{dataset_info['name']}.jsonl"
            count = 0
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in ds:
                    # Extract text based on field
                    text = example.get(dataset_info["text_field"], "")
                    
                    # Handle different text formats
                    if isinstance(text, list):
                        # For messages/conversations
                        text = " ".join([str(item) for item in text])
                    elif isinstance(text, dict):
                        # For complex structures
                        text = str(text)
                    
                    # Clean and validate text
                    if text and isinstance(text, str) and len(text.strip()) > 10:
                        clean_text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
                        if clean_text.strip():
                            f.write(json.dumps({"text": clean_text.strip()}) + "\n")
                            count += 1
                            
                            if count >= dataset_info["max_items"]:
                                break
                    
                    if count % 10000 == 0 and count > 0:
                        print(f"  Downloaded {count:,} samples...")
            
            print(f"✅ {dataset_info['name']}: {count:,} samples saved to {output_file}")
            total_samples += count
            
        except Exception as e:
            print(f"❌ Failed to download {dataset_info['name']}: {e}")
            continue
    
    print(f"\n🎉 Total downloaded: {total_samples:,} samples")
    return total_samples

if __name__ == "__main__":
    print("🚀 Starting comprehensive dataset download...")
    total = download_reliable_datasets()
    print(f"\n✅ Download complete! Ready for training with {total:,} samples.")
