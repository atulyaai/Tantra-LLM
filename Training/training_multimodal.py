"""
Multi-Modal Mamba 3 Training Pipeline
Supports progressive training with dynamic vocabulary and compression
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import yaml
import random
from pathlib import Path
from tqdm import tqdm
import logging
import numpy as np
from typing import Dict, Any, List, Tuple
import wandb
from model_mamba3_multimodal import Mamba3MultiModal, Mamba3Config, DynamicVocabulary
import safetensors.torch
import multiprocessing as mp

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalDataset(Dataset):
    """Dataset for multi-modal training with dynamic vocabulary support"""
    
    def __init__(self, data_files: Dict[str, str], tokenizer, config: Mamba3Config, 
                 max_samples: int = None, modality_priority: List[str] = None):
        self.data_files = data_files
        self.tokenizer = tokenizer
        self.config = config
        self.max_samples = max_samples
        self.modality_priority = modality_priority or ["audio", "text", "vision"]
        
        # Load data
        self.samples = self._load_multimodal_data()
        
    def _load_multimodal_data(self) -> List[Dict[str, Any]]:
        """Load multi-modal data from files"""
        samples = []
        
        for modality, file_path in self.data_files.items():
            if not Path(file_path).exists():
                logger.warning(f"Data file {file_path} not found for modality {modality}")
                continue
                
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if modality not in samples:
                            samples.append({})
                        samples[-1][modality] = data
                    except json.JSONDecodeError:
                        continue
        
        # Filter samples that have the required modalities
        filtered_samples = []
        for sample in samples:
            if len(sample) >= 2:  # At least 2 modalities
                filtered_samples.append(sample)
        
        if self.max_samples:
            filtered_samples = filtered_samples[:self.max_samples]
        
        logger.info(f"Loaded {len(filtered_samples)} multi-modal samples")
        return filtered_samples
    
    def _process_audio(self, audio_data: Dict[str, Any]) -> torch.Tensor:
        """Process audio data to features"""
        # Simulate audio feature extraction
        # In practice, you would use actual audio processing
        if 'features' in audio_data:
            features = np.array(audio_data['features'])
        else:
            # Generate random features as placeholder
            features = np.random.randn(128, self.config.audio_dim)
        
        # Pad or truncate to fixed length
        target_length = 128
        if features.shape[0] > target_length:
            features = features[:target_length]
        else:
            padding = np.zeros((target_length - features.shape[0], self.config.audio_dim))
            features = np.vstack([features, padding])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _process_text(self, text_data: Dict[str, Any]) -> torch.Tensor:
        """Process text data with dynamic vocabulary"""
        text = text_data.get('text', '')
        if isinstance(text, dict):
            text = text.get('content', '')
        
        # Tokenize
        tokens = self.tokenizer.encode(text).ids
        
        # Pad or truncate
        target_length = 128
        if len(tokens) > target_length:
            tokens = tokens[:target_length]
        else:
            pad_token = self.tokenizer.get_vocab().get('[PAD]', 0)
            tokens = tokens + [pad_token] * (target_length - len(tokens))
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def _process_vision(self, vision_data: Dict[str, Any]) -> torch.Tensor:
        """Process vision data"""
        # Simulate image processing
        # In practice, you would load and preprocess actual images
        if 'image_path' in vision_data:
            # Load image from path
            # For now, generate random image
            image = torch.randn(3, 224, 224)
        else:
            # Generate random image as placeholder
            image = torch.randn(3, 224, 224)
        
        return image
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        inputs = {}
        targets = {}
        
        # Process each modality
        for modality in self.modality_priority:
            if modality in sample:
                if modality == "audio":
                    inputs[modality] = self._process_audio(sample[modality])
                    targets[modality] = inputs[modality]  # Autoencoder target
                
                elif modality == "text":
                    inputs[modality] = self._process_text(sample[modality])
                    # Create target by shifting tokens
                    text_tokens = inputs[modality]
                    targets[modality] = torch.cat([text_tokens[1:], text_tokens[:1]])
                
                elif modality == "vision":
                    inputs[modality] = self._process_vision(sample[modality])
                    targets[modality] = inputs[modality]  # Autoencoder target
        
        return inputs, targets


class CompressionTrainer:
    """Trainer with compression techniques"""
    
    def __init__(self, model: Mamba3MultiModal, config: Mamba3Config):
        self.model = model
        self.config = config
        self.teacher_model = None
        
    def setup_distillation(self, teacher_model: Mamba3MultiModal):
        """Setup knowledge distillation with teacher model"""
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        
    def distillation_loss(self, student_outputs: Dict[str, torch.Tensor], 
                         teacher_outputs: Dict[str, torch.Tensor],
                         alpha: float = 0.3) -> torch.Tensor:
        """Compute distillation loss"""
        total_loss = 0.0
        
        for modality in student_outputs:
            if modality in teacher_outputs:
                student_logits = student_outputs[modality]
                teacher_logits = teacher_outputs[modality]
                
                # Softmax with temperature
                temperature = 3.0
                student_soft = F.softmax(student_logits / temperature, dim=-1)
                teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
                
                # KL divergence loss
                kl_loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=-1),
                    teacher_soft,
                    reduction='batchmean'
                ) * (temperature ** 2)
                
                total_loss += alpha * kl_loss
        
        return total_loss
    
    def apply_quantization_aware_training(self, optimizer: optim.Optimizer):
        """Apply quantization-aware training"""
        # This would typically involve using frameworks like QAT
        # For now, we'll simulate it by applying quantization after each step
        pass
    
    def apply_pruning_gradual(self, epoch: int, total_epochs: int):
        """Apply gradual pruning during training"""
        if self.config.pruning_ratio > 0:
            current_ratio = self.config.pruning_ratio * (epoch / total_epochs)
            self.model.apply_pruning(current_ratio)


def train_multimodal_stage(model: Mamba3MultiModal, dataset: MultiModalDataset, 
                          config: Mamba3Config, stage: int, device: torch.device,
                          use_compression: bool = True) -> Tuple[float, float, float]:
    """Train one stage of the multi-modal model"""
    
    # Stage-specific configurations
    stage_configs = {
        1: {"seq_len": 128, "batch_size": 16, "lr": 0.003, "epochs": 3, "workers": 4},
        2: {"seq_len": 256, "batch_size": 8, "lr": 0.002, "epochs": 4, "workers": 3},
        3: {"seq_len": 512, "batch_size": 4, "lr": 0.001, "epochs": 5, "workers": 2},
        4: {"seq_len": 1024, "batch_size": 2, "lr": 0.0005, "epochs": 6, "workers": 1}
    }
    
    stage_config = stage_configs[stage]
    logger.info(f"Stage {stage}: {stage_config}")
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=stage_config["batch_size"],
        shuffle=True,
        num_workers=stage_config["workers"],
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_multimodal_batch
    )
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=stage_config["lr"], weight_decay=0.01)
    
    # Setup compression trainer
    compression_trainer = CompressionTrainer(model, config)
    
    # Loss functions for each modality
    loss_functions = {
        "audio": nn.MSELoss(),
        "text": nn.CrossEntropyLoss(),
        "vision": nn.MSELoss()
    }
    
    model.train()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    
    for epoch in range(stage_config["epochs"]):
        logger.info(f"Stage {stage}, Epoch {epoch + 1}/{stage_config['epochs']}")
        
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_correct = 0
        
        pbar = tqdm(dataloader, desc=f"Stage {stage} Epoch {epoch+1}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = {k: v.to(device) for k, v in targets.items()}
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs, modality_priority=dataset.modality_priority)
            
            # Compute losses
            total_loss = 0.0
            for modality in outputs:
                if modality in targets and modality in loss_functions:
                    if modality == "text":
                        # Text generation loss
                        loss = loss_functions[modality](
                            outputs[modality].view(-1, outputs[modality].size(-1)),
                            targets[modality].view(-1)
                        )
                    else:
                        # Reconstruction loss for audio/vision
                        loss = loss_functions[modality](outputs[modality], targets[modality])
                    
                    total_loss += loss
            
            # Apply compression techniques
            if use_compression:
                # Gradual pruning
                compression_trainer.apply_pruning_gradual(epoch, stage_config["epochs"])
                
                # Quantization-aware training
                compression_trainer.apply_quantization_aware_training(optimizer)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            epoch_loss += total_loss.item()
            
            # Calculate accuracy for text
            if "text" in outputs and "text" in targets:
                with torch.no_grad():
                    pred = torch.argmax(outputs["text"], dim=-1)
                    correct = (pred == targets["text"]).sum().item()
                    epoch_correct += correct
                    epoch_tokens += targets["text"].numel()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'acc': f'{epoch_correct/max(epoch_tokens, 1):.3f}'
            })
            
            # Update dynamic vocabulary
            if "text" in inputs:
                model.update_vocabulary(inputs["text"].flatten().tolist())
        
        avg_loss = epoch_loss / len(dataloader)
        accuracy = epoch_correct / max(epoch_tokens, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        logger.info(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}, Accuracy: {accuracy:.3f}")
        
        # Apply compression at the end of each epoch
        if use_compression:
            model.compress_model()
    
    return avg_loss, perplexity, accuracy


def collate_multimodal_batch(batch):
    """Collate function for multi-modal batches"""
    inputs = {}
    targets = {}
    
    # Collect all modalities present in the batch
    all_modalities = set()
    for sample_inputs, sample_targets in batch:
        all_modalities.update(sample_inputs.keys())
        all_modalities.update(sample_targets.keys())
    
    # Stack tensors for each modality
    for modality in all_modalities:
        modality_inputs = []
        modality_targets = []
        
        for sample_inputs, sample_targets in batch:
            if modality in sample_inputs:
                modality_inputs.append(sample_inputs[modality])
            if modality in sample_targets:
                modality_targets.append(sample_targets[modality])
        
        if modality_inputs:
            inputs[modality] = torch.stack(modality_inputs)
        if modality_targets:
            targets[modality] = torch.stack(modality_targets)
    
    return inputs, targets


def create_progressive_multimodal_model(vocab_size: int, stage: int, config: Mamba3Config) -> Mamba3MultiModal:
    """Create progressive multi-modal model"""
    
    # Update config for this stage
    stage_configs = {
        1: {"d_model": 256, "n_layers": 4, "num_experts": 4},
        2: {"d_model": 512, "n_layers": 8, "num_experts": 6},
        3: {"d_model": 768, "n_layers": 12, "num_experts": 8},
        4: {"d_model": 1024, "n_layers": 16, "num_experts": 12}
    }
    
    stage_config = stage_configs[stage]
    config.d_model = stage_config["d_model"]
    config.n_layers = stage_config["n_layers"]
    config.num_experts = stage_config["num_experts"]
    config.initial_vocab_size = vocab_size
    
    return Mamba3MultiModal(config)


def main():
    """Main training function"""
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info(f"Using CPU with {mp.cpu_count()} cores")
    
    # Load configuration
    config = Mamba3Config(
        d_model=768,
        n_layers=12,
        num_experts=8,
        quantization_bits=8,
        pruning_ratio=0.1,
        distillation_alpha=0.3
    )
    
    # Load tokenizer
    tokenizer_path = 'Model/tokenizer.json'
    if not Path(tokenizer_path).exists():
        logger.warning("Tokenizer not found. Creating basic tokenizer...")
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace
        
        tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=config.initial_vocab_size, 
                           special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[CLS]", "[MASK]"])
        
        # Train on sample data
        sample_texts = ["Hello world", "This is a test", "Machine learning is interesting"]
        tokenizer.train_from_iterator(sample_texts, trainer)
        tokenizer.save(tokenizer_path)
    else:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(tokenizer_path)
    
    vocab_size = len(tokenizer.get_vocab())
    
    # Prepare data files (create sample data if not exists)
    data_files = {
        "audio": "Dataset/audio_data.jsonl",
        "text": "Dataset/combined_full_training.jsonl",
        "vision": "Dataset/vision_data.jsonl"
    }
    
    # Create sample data if files don't exist
    Path("Dataset").mkdir(exist_ok=True)
    for modality, file_path in data_files.items():
        if not Path(file_path).exists():
            logger.info(f"Creating sample data for {modality}")
            create_sample_data(file_path, modality)
    
    # Progressive training stages
    stages = [1, 2, 3, 4]
    current_model = None
    
    for stage in stages:
        logger.info(f"\n🚀 Starting Multi-Modal Stage {stage}")
        
        # Create model for this stage
        model = create_progressive_multimodal_model(vocab_size, stage, config).to(device)
        logger.info(f"Stage {stage} model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Transfer weights from previous stage
        if current_model is not None:
            logger.info(f"Transferring weights from stage {stage-1} to stage {stage}")
            model = transfer_weights_multimodal(current_model, model)
        
        # Create dataset
        dataset = MultiModalDataset(data_files, tokenizer, config, max_samples=10000)
        
        # Train this stage
        loss, ppl, acc = train_multimodal_stage(model, dataset, config, stage, device)
        
        logger.info(f"Stage {stage} Results:")
        logger.info(f"  Loss: {loss:.4f}")
        logger.info(f"  Perplexity: {ppl:.2f}")
        logger.info(f"  Accuracy: {acc:.3f}")
        
        # Save stage weights
        stage_weights_path = f'Model/tantra_multimodal_stage_{stage}.safetensors'
        safetensors.torch.save_file(model.state_dict(), stage_weights_path)
        logger.info(f"Saved stage {stage} weights: {stage_weights_path}")
        
        # Update current model
        current_model = model
        
        # Save main weights
        safetensors.torch.save_file(model.state_dict(), 'Model/tantra_multimodal_weights.safetensors')
        logger.info(f"Updated main multi-modal weights with stage {stage}")
    
    logger.info("🎉 Multi-Modal Progressive Training Completed!")
    logger.info(f"Final model: {sum(p.numel() for p in current_model.parameters()):,} parameters")


def create_sample_data(file_path: str, modality: str):
    """Create sample data for testing"""
    sample_data = []
    
    if modality == "audio":
        for i in range(1000):
            sample_data.append({
                "features": np.random.randn(128, 128).tolist(),
                "duration": random.uniform(1.0, 10.0),
                "sample_rate": 16000
            })
    
    elif modality == "text":
        texts = [
            "Hello, how are you today?",
            "What is the capital of France?",
            "Explain the concept of machine learning.",
            "Write a short story about a robot.",
            "What are the benefits of renewable energy?"
        ]
        for i in range(1000):
            sample_data.append({
                "text": random.choice(texts),
                "category": random.choice(["question", "instruction", "conversation"])
            })
    
    elif modality == "vision":
        for i in range(1000):
            sample_data.append({
                "image_path": f"sample_image_{i}.jpg",
                "width": 224,
                "height": 224,
                "channels": 3
            })
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")


def transfer_weights_multimodal(old_model: Mamba3MultiModal, new_model: Mamba3MultiModal) -> Mamba3MultiModal:
    """Transfer weights between multi-modal models"""
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


if __name__ == "__main__":
    main()