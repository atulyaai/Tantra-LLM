import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import yaml
import glob
import random
from pathlib import Path
from tqdm import tqdm
import logging
from model_mamba import MambaDecoder
from tokenizers import Tokenizer
import safetensors.torch
import multiprocessing as mp

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServerOptimizedDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text).ids
        
        if len(tokens) < self.seq_len + 1:
            pad_token = self.tokenizer.get_vocab().get('[PAD]', 0)
            tokens = tokens + [pad_token] * (self.seq_len + 1 - len(tokens))
        
        start = random.randint(0, max(0, len(tokens) - self.seq_len - 1))
        tokens = tokens[start:start + self.seq_len + 1]
        
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y

def load_texts(data_file):
    texts = []
    with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'text' in data and data['text'].strip():
                    texts.append(data['text'].strip())
            except json.JSONDecodeError:
                continue
    return texts

def create_progressive_model(vocab_size, stage):
    """Create model that grows with training stages"""
    
    stage_configs = {
        1: {"d_model": 256, "n_layers": 4, "d_state": 16, "d_conv": 4},
        2: {"d_model": 512, "n_layers": 8, "d_state": 32, "d_conv": 4},
        3: {"d_model": 768, "n_layers": 16, "d_state": 64, "d_conv": 4},
        4: {"d_model": 1024, "n_layers": 24, "d_state": 128, "d_conv": 4}
    }
    
    config = stage_configs[stage]
    return MambaDecoder(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        d_state=config["d_state"],
        d_conv=config["d_conv"],
        dropout=0.1
    )

def transfer_weights(old_model, new_model):
    """Transfer compatible weights from old to new model"""
    
    old_state = old_model.state_dict()
    new_state = new_model.state_dict()
    
    transferred = 0
    for name, param in old_state.items():
        if name in new_state:
            old_shape = param.shape
            new_shape = new_state[name].shape
            
            if old_shape == new_shape:
                new_state[name] = param
                transferred += 1
            elif len(old_shape) == len(new_shape) and old_shape[0] == new_shape[0]:
                min_dim = min(old_shape[1], new_shape[1])
                new_state[name][:, :min_dim] = param[:, :min_dim]
                transferred += 1
    
    new_model.load_state_dict(new_state)
    logger.info(f"Transferred {transferred} compatible layers")
    return new_model

def train_stage(model, tokenizer, texts, stage, device):
    """Train one stage with server optimizations"""
    
    # Server-optimized configs (larger batch sizes, more workers)
    stage_configs = {
        1: {"seq_len": 128, "batch_size": 64, "lr": 0.003, "epochs": 2, "loss_threshold": 2.5, "acc_threshold": 0.50, "workers": 8},
        2: {"seq_len": 256, "batch_size": 32, "lr": 0.002, "epochs": 3, "loss_threshold": 2.0, "acc_threshold": 0.60, "workers": 6},
        3: {"seq_len": 512, "batch_size": 16, "lr": 0.001, "epochs": 4, "loss_threshold": 1.8, "acc_threshold": 0.65, "workers": 4},
        4: {"seq_len": 1024, "batch_size": 8, "lr": 0.0005, "epochs": 5, "loss_threshold": 1.5, "acc_threshold": 0.70, "workers": 2}
    }
    
    config = stage_configs[stage]
    logger.info(f"Stage {stage}: {config['seq_len']} seq, {config['batch_size']} batch, {config['lr']} lr, {config['workers']} workers")
    
    dataset = ServerOptimizedDataset(texts, tokenizer, config['seq_len'])
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['workers'],
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if config['workers'] > 0 else False
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(config['epochs']):
        logger.info(f"Stage {stage}, Epoch {epoch + 1}/{config['epochs']}")
        
        total_loss = 0
        total_tokens = 0
        correct_tokens = 0
        
        pbar = tqdm(dataloader, desc=f"Stage {stage} Epoch {epoch+1}", unit="batch")
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            logits = model(x)
            
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                pred = torch.argmax(logits, dim=-1)
                correct = (pred == y).sum().item()
                total_tokens += y.numel()
                correct_tokens += correct
                total_loss += loss.item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct_tokens/total_tokens:.3f}',
                    'ppl': f'{torch.exp(loss).item():.2f}'
                })
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_tokens / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        logger.info(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}, Accuracy: {accuracy:.3f}")
        
        # Check exit conditions
        if avg_loss < config['loss_threshold'] and accuracy > config['acc_threshold']:
            logger.info(f"Stage {stage} exit condition met! Loss: {avg_loss:.4f} < {config['loss_threshold']}, Acc: {accuracy:.3f} > {config['acc_threshold']}")
            break
    
    return avg_loss, perplexity, accuracy

def main():
    # Detect best device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        logger.info(f"Using CPU with {mp.cpu_count()} cores")
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file('Model/tokenizer.json')
    vocab_size = len(tokenizer.get_vocab())
    
    # Load data
    texts = load_texts('Dataset/combined_full_training.jsonl')
    logger.info(f"Loaded {len(texts):,} text samples")
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Progressive training stages
    stages = [1, 2, 3, 4]
    current_model = None
    
    for stage in stages:
        logger.info(f"\n🚀 Starting Stage {stage}")
        
        # Create model for this stage
        model = create_progressive_model(vocab_size, stage).to(device)
        logger.info(f"Stage {stage} model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Transfer weights from previous stage
        if current_model is not None:
            logger.info(f"Transferring weights from stage {stage-1} to stage {stage}")
            model = transfer_weights(current_model, model)
        
        # Train this stage
        loss, ppl, acc = train_stage(model, tokenizer, texts, stage, device)
        
        logger.info(f"Stage {stage} Results:")
        logger.info(f"  Loss: {loss:.4f}")
        logger.info(f"  Perplexity: {ppl:.2f}")
        logger.info(f"  Accuracy: {acc:.3f}")
        
        # Save stage weights
        stage_weights_path = f'Model/tantra_stage_{stage}.safetensors'
        safetensors.torch.save_file(model.state_dict(), stage_weights_path)
        logger.info(f"Saved stage {stage} weights: {stage_weights_path}")
        
        # Update current model
        current_model = model
        
        # Also save as main weights
        safetensors.torch.save_file(model.state_dict(), 'Model/tantra_weights.safetensors')
        logger.info(f"Updated main weights with stage {stage}")
    
    logger.info("🎉 Progressive training completed!")
    logger.info(f"Final model: {sum(p.numel() for p in current_model.parameters()):,} parameters")

if __name__ == "__main__":
    main()
