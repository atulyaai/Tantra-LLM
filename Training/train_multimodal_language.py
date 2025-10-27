"""
Training Script for Multi-Modal Language Model
Supports text, audio, vision with OCR weight storage, reasoning, and domain knowledge
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
from multimodal_language_model import MultiModalLanguageModel, MultiModalLanguageConfig
import safetensors.torch
import multiprocessing as mp
from PIL import Image
import cv2
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalLanguageDataset(Dataset):
    """Dataset for Multi-Modal Language Model training"""
    
    def __init__(self, data_files: Dict[str, str], tokenizer, config: MultiModalLanguageConfig, 
                 max_samples: int = None):
        self.data_files = data_files
        self.tokenizer = tokenizer
        self.config = config
        self.max_samples = max_samples
        
        # Load data
        self.samples = self._load_multimodal_data()
        
        # Domain knowledge categories
        self.domain_categories = [
            "science", "technology", "medicine", "history", "geography", 
            "literature", "mathematics", "philosophy", "art", "sports"
        ]
    
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
            if len(sample) >= 1:  # At least 1 modality
                filtered_samples.append(sample)
        
        if self.max_samples:
            filtered_samples = filtered_samples[:self.max_samples]
        
        logger.info(f"Loaded {len(filtered_samples)} multi-modal samples")
        return filtered_samples
    
    def _process_text(self, text_data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process text data with domain knowledge integration"""
        text = text_data.get('text', '')
        if isinstance(text, dict):
            text = text.get('content', '')
        
        # Tokenize
        tokens = self.tokenizer.encode(text).ids
        
        # Pad or truncate
        target_length = 256
        if len(tokens) > target_length:
            tokens = tokens[:target_length]
        else:
            pad_token = self.tokenizer.get_vocab().get('[PAD]', 0)
            tokens = tokens + [pad_token] * (target_length - len(tokens))
        
        # Create target (shifted tokens)
        target_tokens = tokens[1:] + [pad_token]
        
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(target_tokens, dtype=torch.long)
    
    def _process_audio(self, audio_data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process audio data"""
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
        
        # Create target (reconstruction)
        target = features.copy()
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    
    def _process_vision(self, vision_data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process vision data"""
        # Simulate image processing
        if 'image_path' in vision_data:
            # Load image from path
            # For now, generate random image
            image = torch.randn(3, 224, 224)
        else:
            # Generate random image as placeholder
            image = torch.randn(3, 224, 224)
        
        # Create target (reconstruction)
        target = image.clone()
        
        return image, target
    
    def _extract_domain_knowledge(self, text: str) -> str:
        """Extract domain knowledge from text"""
        text_lower = text.lower()
        
        for category in self.domain_categories:
            if category in text_lower:
                return category
        
        # Default to general knowledge
        return "general"
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        inputs = {}
        targets = {}
        domain = "general"
        
        # Process each modality
        if "text" in sample:
            text_input, text_target = self._process_text(sample["text"])
            inputs["text"] = text_input
            targets["text"] = text_target
            domain = self._extract_domain_knowledge(sample["text"].get('text', ''))
        
        if "audio" in sample:
            audio_input, audio_target = self._process_audio(sample["audio"])
            inputs["audio"] = audio_input
            targets["audio"] = audio_target
        
        if "vision" in sample:
            vision_input, vision_target = self._process_vision(sample["vision"])
            inputs["vision"] = vision_input
            targets["vision"] = vision_target
        
        return inputs, targets, domain


class MultiModalLanguageTrainer:
    """Trainer for Multi-Modal Language Model"""
    
    def __init__(self, model: MultiModalLanguageModel, config: MultiModalLanguageConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        
        # Loss functions
        self.text_criterion = nn.CrossEntropyLoss()
        self.audio_criterion = nn.MSELoss()
        self.vision_criterion = nn.MSELoss()
        
        # Training metrics
        self.training_history = []
        self.best_loss = float('inf')
    
    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer, 
                   epoch: int) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        total_text_loss = 0.0
        total_audio_loss = 0.0
        total_vision_loss = 0.0
        
        num_batches = len(dataloader)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, (inputs, targets, domains) in enumerate(pbar):
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model.forward(inputs, use_reasoning=True)
            
            # Compute losses
            total_batch_loss = 0.0
            text_loss = 0.0
            audio_loss = 0.0
            vision_loss = 0.0
            
            if "text" in outputs and "text" in targets:
                text_loss = self.text_criterion(
                    outputs["text"].view(-1, outputs["text"].size(-1)),
                    targets["text"].view(-1)
                )
                total_batch_loss += text_loss
                total_text_loss += text_loss.item()
            
            if "audio" in outputs and "audio" in targets:
                audio_loss = self.audio_criterion(outputs["audio"], targets["audio"])
                total_batch_loss += audio_loss
                total_audio_loss += audio_loss.item()
            
            if "vision" in outputs and "vision" in targets:
                vision_loss = self.vision_criterion(outputs["vision"], targets["vision"])
                total_batch_loss += vision_loss
                total_vision_loss += vision_loss.item()
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            total_loss += total_batch_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_batch_loss.item():.4f}',
                'text': f'{text_loss.item():.4f}' if text_loss > 0 else '0.0000',
                'audio': f'{audio_loss.item():.4f}' if audio_loss > 0 else '0.0000',
                'vision': f'{vision_loss.item():.4f}' if vision_loss > 0 else '0.0000'
            })
            
            # Store weights as OCR periodically
            if batch_idx % 100 == 0:
                self.model.store_weights_as_ocr()
        
        # Calculate average losses
        avg_loss = total_loss / num_batches
        avg_text_loss = total_text_loss / num_batches
        avg_audio_loss = total_audio_loss / num_batches
        avg_vision_loss = total_vision_loss / num_batches
        
        # Update training history
        epoch_metrics = {
            "epoch": epoch,
            "total_loss": avg_loss,
            "text_loss": avg_text_loss,
            "audio_loss": avg_audio_loss,
            "vision_loss": avg_vision_loss,
            "timestamp": time.time()
        }
        self.training_history.append(epoch_metrics)
        
        # Update best loss
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
        
        return epoch_metrics
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0.0
        total_text_loss = 0.0
        total_audio_loss = 0.0
        total_vision_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets, domains in dataloader:
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # Forward pass
                outputs = self.model.forward(inputs, use_reasoning=True)
                
                # Compute losses
                if "text" in outputs and "text" in targets:
                    text_loss = self.text_criterion(
                        outputs["text"].view(-1, outputs["text"].size(-1)),
                        targets["text"].view(-1)
                    )
                    total_text_loss += text_loss.item()
                    total_loss += text_loss.item()
                
                if "audio" in outputs and "audio" in targets:
                    audio_loss = self.audio_criterion(outputs["audio"], targets["audio"])
                    total_audio_loss += audio_loss.item()
                    total_loss += audio_loss.item()
                
                if "vision" in outputs and "vision" in targets:
                    vision_loss = self.vision_criterion(outputs["vision"], targets["vision"])
                    total_vision_loss += vision_loss.item()
                    total_loss += vision_loss.item()
        
        num_batches = len(dataloader)
        return {
            "total_loss": total_loss / num_batches,
            "text_loss": total_text_loss / num_batches,
            "audio_loss": total_audio_loss / num_batches,
            "vision_loss": total_vision_loss / num_batches
        }


def collate_multimodal_batch(batch):
    """Collate function for multi-modal batches"""
    inputs = {}
    targets = {}
    domains = []
    
    # Collect all modalities present in the batch
    all_modalities = set()
    for sample_inputs, sample_targets, domain in batch:
        all_modalities.update(sample_inputs.keys())
        all_modalities.update(sample_targets.keys())
        domains.append(domain)
    
    # Stack tensors for each modality
    for modality in all_modalities:
        modality_inputs = []
        modality_targets = []
        
        for sample_inputs, sample_targets, _ in batch:
            if modality in sample_inputs:
                modality_inputs.append(sample_inputs[modality])
            if modality in sample_targets:
                modality_targets.append(sample_targets[modality])
        
        if modality_inputs:
            inputs[modality] = torch.stack(modality_inputs)
        if modality_targets:
            targets[modality] = torch.stack(modality_targets)
    
    return inputs, targets, domains


def create_sample_data(file_path: str, modality: str):
    """Create sample data for testing"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    sample_data = []
    
    if modality == "text":
        texts = [
            "Hello, how are you today?",
            "What is artificial intelligence?",
            "Explain the concept of machine learning.",
            "How does quantum computing work?",
            "What are the benefits of renewable energy?",
            "Describe the process of photosynthesis.",
            "What is the meaning of life?",
            "Explain neural networks in simple terms.",
            "How do computers process information?",
            "What is the history of the internet?",
            "Describe the structure of DNA.",
            "How does the human brain work?",
            "What is climate change?",
            "Explain the theory of relativity.",
            "How do vaccines work?",
            "What is the future of technology?",
            "Describe the water cycle.",
            "How do solar panels work?",
            "What is the role of mathematics in science?",
            "Explain the concept of evolution."
        ]
        
        for i in range(1000):
            text = random.choice(texts)
            category = random.choice(["question", "instruction", "conversation", "explanation"])
            domain = random.choice(["science", "technology", "medicine", "history", "geography"])
            
            sample_data.append({
                "text": text,
                "category": category,
                "domain": domain,
                "metadata": {
                    "length": len(text),
                    "complexity": random.choice(["simple", "medium", "complex"])
                }
            })
    
    elif modality == "audio":
        for i in range(1000):
            sample_data.append({
                "features": np.random.randn(256, 256).tolist(),
                "duration": random.uniform(1.0, 10.0),
                "sample_rate": 16000,
                "transcript": f"Sample audio {i}",
                "language": random.choice(["en", "es", "fr", "de"]),
                "speaker_id": f"speaker_{random.randint(1, 10)}"
            })
    
    elif modality == "vision":
        for i in range(1000):
            sample_data.append({
                "image_path": f"sample_image_{i}.jpg",
                "width": 224,
                "height": 224,
                "channels": 3,
                "description": f"Sample image {i}",
                "category": random.choice(["object", "scene", "person", "text", "diagram"]),
                "quality": random.choice(["high", "medium", "low"])
            })
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")
    
    logger.info(f"Created {len(sample_data)} samples for {modality}")


def main():
    """Main training function"""
    logger.info("🚀 Starting Multi-Modal Language Model Training")
    
    # Create configuration
    config = MultiModalLanguageConfig(
        d_model=1024,
        n_layers=24,
        vocab_size=50000,
        ocr_enabled=True,
        memory_capacity=50000,
        domain_knowledge_size=10000
    )
    
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
            vocab_size=config.vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[CLS]", "[MASK]"]
        )
        
        # Train on sample data
        sample_texts = [
            "Hello world, this is a sample text for multi-modal training.",
            "Machine learning and artificial intelligence are fascinating topics.",
            "Natural language processing helps computers understand human language.",
            "Deep learning models can process vast amounts of data efficiently.",
            "The future of AI holds many exciting possibilities and challenges.",
            "Multi-modal learning combines different types of data for better understanding.",
            "OCR technology can convert images to text for processing.",
            "Domain knowledge helps models provide more accurate responses.",
            "Reasoning capabilities enable models to think logically and solve problems.",
            "Training on diverse data improves model generalization and performance."
        ] * 100
        
        tokenizer.train_from_iterator(sample_texts, trainer)
        tokenizer.save('Model/tokenizer.json')
        logger.info("Created basic tokenizer")
    
    # Prepare data files
    data_files = {
        "text": "Dataset/text_data.jsonl",
        "audio": "Dataset/audio_data.jsonl",
        "vision": "Dataset/vision_data.jsonl"
    }
    
    # Create sample data if files don't exist
    for modality, file_path in data_files.items():
        if not Path(file_path).exists():
            logger.info(f"Creating sample data for {modality}")
            create_sample_data(file_path, modality)
    
    # Create model
    model = MultiModalLanguageModel(config).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Add domain knowledge
    logger.info("Adding domain knowledge...")
    model.add_domain_knowledge("technology", "artificial_intelligence", 
                              "Artificial Intelligence is the simulation of human intelligence in machines.")
    model.add_domain_knowledge("science", "physics", 
                              "Physics is the study of matter, energy, and their interactions.")
    model.add_domain_knowledge("medicine", "anatomy", 
                              "Anatomy is the study of the structure of living organisms.")
    
    # Create dataset
    dataset = MultiModalLanguageDataset(data_files, tokenizer, config, max_samples=10000)
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_multimodal_batch
    )
    
    # Create trainer
    trainer = MultiModalLanguageTrainer(model, config, device)
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        logger.info(f"\n🎯 Starting Epoch {epoch + 1}/{num_epochs}")
        
        # Train epoch
        metrics = trainer.train_epoch(dataloader, optimizer, epoch + 1)
        
        logger.info(f"Epoch {epoch + 1} Results:")
        logger.info(f"  Total Loss: {metrics['total_loss']:.4f}")
        logger.info(f"  Text Loss: {metrics['text_loss']:.4f}")
        logger.info(f"  Audio Loss: {metrics['audio_loss']:.4f}")
        logger.info(f"  Vision Loss: {metrics['vision_loss']:.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % 2 == 0:
            checkpoint_path = f'Model/multimodal_language_epoch_{epoch + 1}.safetensors'
            safetensors.torch.save_file(model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_model_path = 'Model/multimodal_language_final.safetensors'
    safetensors.torch.save_file(model.state_dict(), final_model_path)
    logger.info(f"Saved final model: {final_model_path}")
    
    # Store all weights as OCR
    logger.info("Storing weights as OCR images...")
    model.store_weights_as_ocr()
    
    # Test model capabilities
    logger.info("\n🧪 Testing Model Capabilities")
    
    # Test response generation
    test_inputs = {
        "text": torch.randint(0, config.vocab_size, (1, 128)).to(device)
    }
    
    response = model.generate_response(test_inputs, "What is artificial intelligence?")
    logger.info(f"Response to AI question: {response}")
    
    greeting = model.generate_response(test_inputs)
    logger.info(f"Greeting: {greeting}")
    
    # Test conversation history
    history = model.get_conversation_history()
    logger.info(f"Conversation history: {len(history)} entries")
    
    logger.info("🎉 Multi-Modal Language Model Training Completed!")


if __name__ == "__main__":
    main()