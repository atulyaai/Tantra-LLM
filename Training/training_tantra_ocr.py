"""
Training Script for Tantra OCR LLM
Optimized for OCR language format, image processing, and enhanced memory
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
import time
from tantra_ocr_llm import TantraOCRLLM, TantraOCRConfig, OCRWeightEncoder
import safetensors.torch
import multiprocessing as mp
from PIL import Image
import cv2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TantraOCRDataset(Dataset):
    """Dataset for Tantra OCR LLM training with OCR-optimized data processing"""
    
    def __init__(self, data_files: Dict[str, str], tokenizer, config: TantraOCRConfig, 
                 max_samples: int = None, modality_priority: List[str] = None):
        self.data_files = data_files
        self.tokenizer = tokenizer
        self.config = config
        self.max_samples = max_samples
        self.modality_priority = modality_priority or ["audio", "text", "vision"]
        
        # Load data
        self.samples = self._load_multimodal_data()
        
        # OCR weight encoder for data augmentation
        self.ocr_encoder = OCRWeightEncoder(config)
    
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
        if 'features' in audio_data:
            features = np.array(audio_data['features'])
        else:
            # Generate random features as placeholder
            features = np.random.randn(256, self.config.audio_dim)
        
        # Pad or truncate to fixed length
        target_length = 256
        if features.shape[0] > target_length:
            features = features[:target_length]
        else:
            padding = np.zeros((target_length - features.shape[0], self.config.audio_dim))
            features = np.vstack([features, padding])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _process_text(self, text_data: Dict[str, Any]) -> torch.Tensor:
        """Process text data with OCR-optimized tokenization"""
        text = text_data.get('text', '')
        if isinstance(text, dict):
            text = text.get('content', '')
        
        # Tokenize with OCR-friendly preprocessing
        text = self._preprocess_text_for_ocr(text)
        tokens = self.tokenizer.encode(text).ids
        
        # Pad or truncate
        target_length = 256
        if len(tokens) > target_length:
            tokens = tokens[:target_length]
        else:
            pad_token = self.tokenizer.get_vocab().get('[PAD]', 0)
            tokens = tokens + [pad_token] * (target_length - len(tokens))
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def _preprocess_text_for_ocr(self, text: str) -> str:
        """Preprocess text to be more OCR-friendly"""
        # Convert to uppercase for better OCR recognition
        text = text.upper()
        
        # Replace common OCR-confusing characters
        replacements = {
            '0': 'O',  # Zero to O
            '1': 'I',  # One to I
            '5': 'S',  # Five to S
            '8': 'B',  # Eight to B
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _process_vision(self, vision_data: Dict[str, Any]) -> torch.Tensor:
        """Process vision data with OCR optimization"""
        # Simulate image processing with OCR enhancement
        if 'image_path' in vision_data:
            # Load image from path
            # For now, generate random image
            image = torch.randn(3, self.config.image_size, self.config.image_size)
        else:
            # Generate random image as placeholder
            image = torch.randn(3, self.config.image_size, self.config.image_size)
        
        # Apply OCR-friendly preprocessing
        image = self._preprocess_image_for_ocr(image)
        
        return image
    
    def _preprocess_image_for_ocr(self, image: torch.Tensor) -> torch.Tensor:
        """Preprocess image to be more OCR-friendly"""
        # Convert to numpy for processing
        img_np = image.numpy().transpose(1, 2, 0)
        
        # Convert to grayscale for better OCR
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        
        # Apply contrast enhancement
        gray = cv2.equalizeHist(gray)
        
        # Apply threshold for better OCR
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert back to tensor
        processed = torch.tensor(thresh, dtype=torch.float32)
        
        # Normalize
        processed = processed / 255.0
        
        # Convert back to 3-channel
        processed = processed.unsqueeze(0).repeat(3, 1, 1)
        
        return processed
    
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


class TantraOCRTrainer:
    """Trainer for Tantra OCR LLM with OCR-specific optimizations"""
    
    def __init__(self, model: TantraOCRLLM, config: TantraOCRConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.ocr_encoder = OCRWeightEncoder(config)
        
        # OCR-specific loss functions
        self.ocr_loss_weight = 0.1
        self.memory_loss_weight = 0.05
        
    def compute_ocr_loss(self, outputs: Dict[str, torch.Tensor], 
                        targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute OCR-specific loss for better pattern recognition"""
        ocr_loss = 0.0
        
        for modality in outputs:
            if modality in targets:
                # Standard reconstruction loss
                if modality == "text":
                    loss = F.cross_entropy(
                        outputs[modality].view(-1, outputs[modality].size(-1)),
                        targets[modality].view(-1)
                    )
                else:
                    loss = F.mse_loss(outputs[modality], targets[modality])
                
                # Add OCR pattern loss
                ocr_pattern_loss = self._compute_ocr_pattern_loss(outputs[modality], targets[modality])
                ocr_loss += loss + self.ocr_loss_weight * ocr_pattern_loss
        
        return ocr_loss
    
    def _compute_ocr_pattern_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute OCR pattern recognition loss"""
        # Convert to OCR images
        output_img = self.ocr_encoder.encode_weights_to_ocr_image(output, "output")
        target_img = self.ocr_encoder.encode_weights_to_ocr_image(target, "target")
        
        # Compute visual similarity loss
        output_array = np.array(output_img)
        target_array = np.array(target_img)
        
        # Convert to grayscale
        if len(output_array.shape) == 3:
            output_gray = cv2.cvtColor(output_array, cv2.COLOR_RGB2GRAY)
        else:
            output_gray = output_array
        
        if len(target_array.shape) == 3:
            target_gray = cv2.cvtColor(target_array, cv2.COLOR_RGB2GRAY)
        else:
            target_gray = target_array
        
        # Compute histogram similarity
        output_hist = cv2.calcHist([output_gray], [0], None, [256], [0, 256])
        target_hist = cv2.calcHist([target_gray], [0], None, [256], [0, 256])
        
        # Normalize histograms
        output_hist = output_hist / output_hist.sum()
        target_hist = target_hist / target_hist.sum()
        
        # Compute correlation
        correlation = cv2.compareHist(output_hist, target_hist, cv2.HISTCMP_CORREL)
        
        # Convert to loss (1 - correlation)
        pattern_loss = 1.0 - correlation
        
        return torch.tensor(pattern_loss, dtype=torch.float32, device=self.device)
    
    def compute_memory_loss(self, model: TantraOCRLLM) -> torch.Tensor:
        """Compute memory consistency loss"""
        memory_loss = 0.0
        
        # Get memory statistics
        stats = model.get_memory_statistics()
        
        # Encourage memory diversity
        if stats["total_memories"] > 0:
            # Memory diversity loss
            memory_types = stats["memory_types"]
            if len(memory_types) > 1:
                # Encourage balanced memory types
                type_counts = list(memory_types.values())
                mean_count = sum(type_counts) / len(type_counts)
                variance = sum((count - mean_count) ** 2 for count in type_counts) / len(type_counts)
                memory_loss += variance
        
        return memory_loss


def train_tantra_ocr_stage(model: TantraOCRLLM, dataset: TantraOCRDataset, 
                          config: TantraOCRConfig, stage: int, device: torch.device) -> Tuple[float, float, float]:
    """Train one stage of the Tantra OCR LLM"""
    
    # Stage-specific configurations
    stage_configs = {
        1: {"seq_len": 128, "batch_size": 8, "lr": 0.001, "epochs": 5, "workers": 4},
        2: {"seq_len": 256, "batch_size": 4, "lr": 0.0005, "epochs": 7, "workers": 3},
        3: {"seq_len": 512, "batch_size": 2, "lr": 0.0002, "epochs": 10, "workers": 2},
        4: {"seq_len": 1024, "batch_size": 1, "lr": 0.0001, "epochs": 15, "workers": 1}
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
        collate_fn=collate_tantra_ocr_batch
    )
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=stage_config["lr"], weight_decay=0.01)
    
    # Setup trainer
    trainer = TantraOCRTrainer(model, config, device)
    
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
            outputs = model(inputs, use_memory=True)
            
            # Compute OCR loss
            ocr_loss = trainer.compute_ocr_loss(outputs, targets)
            
            # Compute memory loss
            memory_loss = trainer.compute_memory_loss(model)
            
            # Total loss
            total_loss = ocr_loss + trainer.memory_loss_weight * memory_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Store memory
            model.store_memory(inputs, f"stage_{stage}_epoch_{epoch}")
            
            # Store weights as OCR periodically
            if batch_idx % 100 == 0:
                for name, param in model.named_parameters():
                    if 'weight' in name and len(param.shape) > 1:
                        model.store_weights_as_ocr(f"{name}_stage_{stage}", param.data)
            
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
                'acc': f'{epoch_correct/max(epoch_tokens, 1):.3f}',
                'mem': f'{model.get_memory_statistics()["total_memories"]}'
            })
        
        avg_loss = epoch_loss / len(dataloader)
        accuracy = epoch_correct / max(epoch_tokens, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        logger.info(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}, Accuracy: {accuracy:.3f}")
        
        # Get memory statistics
        stats = model.get_memory_statistics()
        logger.info(f"Memory stats - Total: {stats['total_memories']}, Types: {stats['memory_types']}")
    
    return avg_loss, perplexity, accuracy


def collate_tantra_ocr_batch(batch):
    """Collate function for Tantra OCR batches"""
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


def create_tantra_ocr_training_pipeline(config_path: str = "Config/tantra_ocr.yaml"):
    """Create Tantra OCR training pipeline"""
    
    # Load configuration
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = TantraOCRConfig(**config_dict.get('model', {}))
    else:
        logger.warning(f"Config file {config_path} not found, using defaults")
        config = TantraOCRConfig()
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info(f"Using CPU with {mp.cpu_count()} cores")
    
    # Load tokenizer
    tokenizer_path = 'Model/tokenizer.json'
    if Path(tokenizer_path).exists():
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(tokenizer_path)
        logger.info("Loaded existing tokenizer")
    else:
        logger.warning("No tokenizer found, creating basic tokenizer...")
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace
        
        tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=50000,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[CLS]", "[MASK]"]
        )
        
        # Train on sample data
        sample_texts = [
            "Hello world, this is a sample text for OCR training.",
            "Machine learning and artificial intelligence are fascinating topics.",
            "Natural language processing helps computers understand human language.",
            "Deep learning models can process vast amounts of data efficiently.",
            "The future of AI holds many exciting possibilities and challenges."
        ] * 100
        
        tokenizer.train_from_iterator(sample_texts, trainer)
        tokenizer.save('Model/tokenizer.json')
        logger.info("Created basic tokenizer")
    
    # Prepare data files
    data_files = {
        "audio": "Dataset/audio_data.jsonl",
        "text": "Dataset/combined_full_training.jsonl",
        "vision": "Dataset/vision_data.jsonl"
    }
    
    # Create sample data if files don't exist
    for modality, file_path in data_files.items():
        if not Path(file_path).exists():
            logger.info(f"Creating sample data for {modality}")
            create_sample_data(file_path, modality)
    
    return config, device, tokenizer, data_files


def create_sample_data(file_path: str, modality: str):
    """Create sample data for testing"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    sample_data = []
    
    if modality == "audio":
        for i in range(1000):
            sample_data.append({
                "features": np.random.randn(256, 256).tolist(),
                "duration": random.uniform(1.0, 10.0),
                "sample_rate": 16000,
                "transcript": f"Sample audio {i}"
            })
    
    elif modality == "text":
        texts = [
            "Hello, how are you today?",
            "What is the capital of France?",
            "Explain the concept of machine learning.",
            "Write a short story about a robot.",
            "What are the benefits of renewable energy?",
            "How does artificial intelligence work?",
            "Describe the process of photosynthesis.",
            "What is the meaning of life?",
            "Explain quantum computing in simple terms.",
            "How do neural networks learn?"
        ]
        for i in range(1000):
            sample_data.append({
                "text": random.choice(texts),
                "category": random.choice(["question", "instruction", "conversation", "explanation"])
            })
    
    elif modality == "vision":
        for i in range(1000):
            sample_data.append({
                "image_features": np.random.randn(256, 1024).tolist(),
                "width": 512,
                "height": 512,
                "channels": 3,
                "description": f"Sample image {i}"
            })
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")
    
    logger.info(f"Created {len(sample_data)} samples for {modality}")


def main():
    """Main training function"""
    logger.info("🚀 Starting Tantra OCR LLM Training Pipeline")
    
    # Create training pipeline
    config, device, tokenizer, data_files = create_tantra_ocr_training_pipeline()
    
    # Progressive training stages
    stages = [1, 2, 3, 4]
    current_model = None
    
    for stage in stages:
        logger.info(f"\n🎯 Starting Tantra OCR Stage {stage}")
        
        # Create model for this stage
        model = TantraOCRLLM(config).to(device)
        logger.info(f"Stage {stage} model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Transfer weights from previous stage
        if current_model is not None:
            logger.info(f"Transferring weights from stage {stage-1} to stage {stage}")
            # Simple weight transfer - in practice, you'd want more sophisticated transfer
            current_state = current_model.state_dict()
            new_state = model.state_dict()
            
            transferred = 0
            for name, param in current_state.items():
                if name in new_state and param.shape == new_state[name].shape:
                    new_state[name] = param
                    transferred += 1
            
            model.load_state_dict(new_state)
            logger.info(f"Transferred {transferred} compatible layers")
        
        # Create dataset
        dataset = TantraOCRDataset(data_files, tokenizer, config, max_samples=5000)
        
        # Train this stage
        loss, ppl, acc = train_tantra_ocr_stage(model, dataset, config, stage, device)
        
        logger.info(f"Stage {stage} Results:")
        logger.info(f"  Loss: {loss:.4f}")
        logger.info(f"  Perplexity: {ppl:.2f}")
        logger.info(f"  Accuracy: {acc:.3f}")
        
        # Save stage weights
        stage_weights_path = f'Model/tantra_ocr_stage_{stage}.safetensors'
        safetensors.torch.save_file(model.state_dict(), stage_weights_path)
        logger.info(f"Saved stage {stage} weights: {stage_weights_path}")
        
        # Update current model
        current_model = model
        
        # Save main weights
        safetensors.torch.save_file(model.state_dict(), 'Model/tantra_ocr_weights.safetensors')
        logger.info(f"Updated main Tantra OCR weights with stage {stage}")
    
    logger.info("🎉 Tantra OCR LLM Training Completed!")
    logger.info(f"Final model: {sum(p.numel() for p in current_model.parameters()):,} parameters")
    
    # Get final memory statistics
    stats = current_model.get_memory_statistics()
    logger.info(f"Final memory statistics: {stats}")


if __name__ == "__main__":
    main()