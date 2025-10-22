"""
Main Multi-Modal Training Script
Integrates all components for end-to-end multi-modal Mamba 3 training
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
import argparse
import time
from model_mamba3_multimodal import Mamba3MultiModal, Mamba3Config, DynamicVocabulary
import safetensors.torch
import multiprocessing as mp

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalTrainingPipeline:
    """Complete multi-modal training pipeline"""
    
    def __init__(self, config_path: str = "Config/multimodal.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.device = self._setup_device()
        self.model = None
        self.tokenizer = None
        self.training_history = []
        
    def _load_config(self) -> Mamba3Config:
        """Load configuration from YAML file"""
        if Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Extract model config
            model_config = config_dict.get('model', {})
            return Mamba3Config(**model_config)
        else:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return Mamba3Config()
    
    def _setup_device(self) -> torch.device:
        """Setup compute device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = torch.device('cpu')
            logger.info(f"Using CPU with {mp.cpu_count()} cores")
        
        return device
    
    def _load_tokenizer(self):
        """Load or create tokenizer"""
        tokenizer_path = 'Model/tokenizer.json'
        if Path(tokenizer_path).exists():
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            logger.info("Loaded existing tokenizer")
        else:
            logger.warning("No tokenizer found, creating basic tokenizer...")
            self._create_basic_tokenizer()
    
    def _create_basic_tokenizer(self):
        """Create basic tokenizer for initial training"""
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace
        
        self.tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
        self.tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=self.config.initial_vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[CLS]", "[MASK]"]
        )
        
        # Train on sample data
        sample_texts = [
            "Hello world, this is a sample text for tokenizer training.",
            "Machine learning and artificial intelligence are fascinating topics.",
            "Natural language processing helps computers understand human language.",
            "Deep learning models can process vast amounts of data efficiently.",
            "The future of AI holds many exciting possibilities and challenges."
        ] * 100
        
        self.tokenizer.train_from_iterator(sample_texts, trainer)
        self.tokenizer.save('Model/tokenizer.json')
        logger.info("Created basic tokenizer")
    
    def _prepare_datasets(self) -> Dict[str, str]:
        """Prepare multi-modal datasets"""
        data_files = {
            "audio": "Dataset/audio_data.jsonl",
            "text": "Dataset/combined_full_training.jsonl",
            "vision": "Dataset/vision_data.jsonl"
        }
        
        # Create sample data if files don't exist
        for modality, file_path in data_files.items():
            if not Path(file_path).exists():
                logger.info(f"Creating sample data for {modality}")
                self._create_sample_data(file_path, modality)
        
        return data_files
    
    def _create_sample_data(self, file_path: str, modality: str):
        """Create sample data for testing"""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        sample_data = []
        
        if modality == "audio":
            for i in range(1000):
                sample_data.append({
                    "audio_features": np.random.randn(128, 128).tolist(),
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
                    "image_features": np.random.randn(196, 512).tolist(),
                    "width": 224,
                    "height": 224,
                    "channels": 3,
                    "description": f"Sample image {i}"
                })
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in sample_data:
                f.write(json.dumps(item) + "\n")
        
        logger.info(f"Created {len(sample_data)} samples for {modality}")
    
    def _create_model(self, stage: int) -> Mamba3MultiModal:
        """Create model for specific training stage"""
        stage_configs = {
            1: {"d_model": 256, "n_layers": 4, "num_experts": 4},
            2: {"d_model": 512, "n_layers": 8, "num_experts": 6},
            3: {"d_model": 768, "n_layers": 12, "num_experts": 8},
            4: {"d_model": 1024, "n_layers": 16, "num_experts": 12}
        }
        
        stage_config = stage_configs[stage]
        
        # Update config for this stage
        self.config.d_model = stage_config["d_model"]
        self.config.n_layers = stage_config["n_layers"]
        self.config.num_experts = stage_config["num_experts"]
        
        # Get vocabulary size
        vocab_size = len(self.tokenizer.get_vocab()) if self.tokenizer else self.config.initial_vocab_size
        self.config.initial_vocab_size = vocab_size
        
        model = Mamba3MultiModal(self.config).to(self.device)
        logger.info(f"Created stage {stage} model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        return model
    
    def _transfer_weights(self, old_model: Mamba3MultiModal, new_model: Mamba3MultiModal) -> Mamba3MultiModal:
        """Transfer weights between models"""
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
    
    def _train_stage(self, model: Mamba3MultiModal, stage: int, data_files: Dict[str, str]) -> Dict[str, float]:
        """Train one stage of the model"""
        from training_multimodal import MultiModalDataset, train_multimodal_stage
        
        # Stage-specific configurations
        stage_configs = {
            1: {"seq_len": 128, "batch_size": 16, "lr": 0.003, "epochs": 3, "workers": 4},
            2: {"seq_len": 256, "batch_size": 8, "lr": 0.002, "epochs": 4, "workers": 3},
            3: {"seq_len": 512, "batch_size": 4, "lr": 0.001, "epochs": 5, "workers": 2},
            4: {"seq_len": 1024, "batch_size": 2, "lr": 0.0005, "epochs": 6, "workers": 1}
        }
        
        stage_config = stage_configs[stage]
        logger.info(f"Stage {stage} configuration: {stage_config}")
        
        # Create dataset
        dataset = MultiModalDataset(
            data_files, 
            self.tokenizer, 
            self.config, 
            max_samples=10000,
            modality_priority=self.config.modality_priority
        )
        
        # Train stage
        loss, perplexity, accuracy = train_multimodal_stage(
            model, dataset, self.config, stage, self.device, use_compression=True
        )
        
        return {
            "loss": loss,
            "perplexity": perplexity,
            "accuracy": accuracy
        }
    
    def _save_model(self, model: Mamba3MultiModal, stage: int):
        """Save model weights"""
        # Save stage weights
        stage_weights_path = f'Model/tantra_multimodal_stage_{stage}.safetensors'
        safetensors.torch.save_file(model.state_dict(), stage_weights_path)
        logger.info(f"Saved stage {stage} weights: {stage_weights_path}")
        
        # Save main weights
        main_weights_path = 'Model/tantra_multimodal_weights.safetensors'
        safetensors.torch.save_file(model.state_dict(), main_weights_path)
        logger.info(f"Updated main weights: {main_weights_path}")
    
    def _log_training_progress(self, stage: int, metrics: Dict[str, float]):
        """Log training progress"""
        self.training_history.append({
            "stage": stage,
            "metrics": metrics,
            "timestamp": time.time()
        })
        
        logger.info(f"Stage {stage} completed:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    def train(self):
        """Main training loop"""
        logger.info("🚀 Starting Multi-Modal Mamba 3 Training Pipeline")
        
        # Initialize components
        self._load_tokenizer()
        data_files = self._prepare_datasets()
        
        # Progressive training stages
        stages = [1, 2, 3, 4]
        current_model = None
        
        for stage in stages:
            logger.info(f"\n🎯 Starting Stage {stage}")
            
            # Create model for this stage
            model = self._create_model(stage)
            
            # Transfer weights from previous stage
            if current_model is not None:
                logger.info(f"Transferring weights from stage {stage-1} to stage {stage}")
                model = self._transfer_weights(current_model, model)
            
            # Train this stage
            metrics = self._train_stage(model, stage, data_files)
            
            # Log progress
            self._log_training_progress(stage, metrics)
            
            # Save model
            self._save_model(model, stage)
            
            # Update current model
            current_model = model
        
        logger.info("🎉 Multi-Modal Training Completed!")
        logger.info(f"Final model: {sum(p.numel() for p in current_model.parameters()):,} parameters")
        
        # Save training history
        with open('logs/training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        return current_model
    
    def evaluate(self, model: Mamba3MultiModal):
        """Evaluate the trained model"""
        logger.info("📊 Starting Model Evaluation")
        
        from eval_multimodal import MultiModalEvaluator, create_test_data
        
        # Create evaluator
        evaluator = MultiModalEvaluator(model, self.device)
        
        # Create test data
        test_data = create_test_data(100)
        
        # Run evaluations
        audio_metrics = evaluator.evaluate_audio_quality(test_data["audio"])
        text_metrics = evaluator.evaluate_text_generation(test_data["text"], self.tokenizer)
        vision_metrics = evaluator.evaluate_vision_analysis(test_data["vision"])
        multimodal_metrics = evaluator.evaluate_multimodal_fusion(test_data["multimodal"], self.tokenizer)
        
        # Generate report
        report = evaluator.generate_evaluation_report()
        evaluator.plot_evaluation_results()
        
        logger.info("📊 Evaluation completed!")
        logger.info(f"Overall score: {report['summary']['average_score']:.3f}")
        
        return report


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Multi-Modal Mamba 3 Training")
    parser.add_argument("--config", type=str, default="Config/multimodal.yaml", help="Config file path")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], help="Train specific stage only")
    
    args = parser.parse_args()
    
    # Create training pipeline
    pipeline = MultiModalTrainingPipeline(args.config)
    
    if args.eval_only:
        # Load existing model for evaluation
        logger.info("Loading model for evaluation...")
        model = Mamba3MultiModal(pipeline.config).to(pipeline.device)
        
        weights_path = 'Model/tantra_multimodal_weights.safetensors'
        if Path(weights_path).exists():
            state_dict = safetensors.torch.load_file(weights_path)
            model.load_state_dict(state_dict)
            logger.info("Model loaded successfully")
        else:
            logger.error("No model weights found for evaluation")
            return
        
        # Run evaluation
        pipeline.evaluate(model)
    
    else:
        # Run training
        if args.stage:
            logger.info(f"Training stage {args.stage} only")
            # Implement single stage training
            pass
        else:
            # Run full training pipeline
            model = pipeline.train()
            
            # Run evaluation
            pipeline.evaluate(model)


if __name__ == "__main__":
    main()