import json
import glob
from pathlib import Path
import random

def combine_all_datasets():
    """Combine all downloaded datasets into one comprehensive training file"""
    
    all_texts = []
    dataset_stats = {}
    
    # Process each dataset file
    dataset_files = [
        "Dataset/alpaca.jsonl",
        "Dataset/openassistant.jsonl", 
        "Dataset/ultrachat.jsonl",
        "Dataset/dolly.jsonl",
        "Dataset/wizardlm.jsonl",
        "Dataset/squad.jsonl"
    ]
    
    for file_path in dataset_files:
        if not Path(file_path).exists():
            print(f"Skipping {file_path} - file not found")
            continue
            
        filename = Path(file_path).name
        print(f"Processing {filename}...")
        
        count = 0
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    
                    # Extract text
                    text = data.get('text', '')
                    
                    # Clean text
                    if text and isinstance(text, str) and len(text.strip()) > 10:
                        clean_text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
                        if clean_text.strip():
                            all_texts.append(clean_text.strip())
                            count += 1
                            
                except json.JSONDecodeError:
                    continue
        
        dataset_stats[filename] = count
        print(f"  Added {count:,} samples from {filename}")
    
    # Shuffle the combined dataset
    random.shuffle(all_texts)
    
    # Save combined dataset
    output_file = 'Dataset/combined_full_training.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in all_texts:
            f.write(json.dumps({"text": text}) + "\n")
    
    print(f"\n📊 Dataset Statistics:")
    for filename, count in dataset_stats.items():
        print(f"  {filename}: {count:,} samples")
    
    print(f"\n✅ Combined Dataset:")
    print(f"  Total samples: {len(all_texts):,}")
    print(f"  Output file: {output_file}")
    print(f"  Estimated size: {len(all_texts) * 200 / 1024 / 1024:.1f} MB")
    
    return len(all_texts)

if __name__ == "__main__":
    print("🔄 Combining all datasets...")
    
    # Create Dataset directory if it doesn't exist
    Path("Dataset").mkdir(exist_ok=True)
    
    total_samples = combine_all_datasets()
    print(f"\n🎉 Ready for progressive training with {total_samples:,} samples!")
