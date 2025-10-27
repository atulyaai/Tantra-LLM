"""
OCR-Native LLM Training Pipeline
Trains the model to process everything in OCR format

Copyright (c) 2024 OCR-Native LLM Contributors
Licensed under the MIT License
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from tqdm import tqdm
import os
import json
from PIL import Image
import cv2

from ocr_native_llm import OCRNativeLLM, OCRNativeConfig

logger = logging.getLogger(__name__)


class OCRNativeDataset(Dataset):
    """Dataset for OCR-native training"""
    
    def __init__(self, config: OCRNativeConfig, data_path: str = None):
        self.config = config
        self.data = []
        
        if data_path and os.path.exists(data_path):
            self.load_from_file(data_path)
        else:
            self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample training data in OCR format"""
        # Sample conversations
        conversations = [
            {
                'text': "Hello, how are you today?",
                'response': "I'm doing well, thank you for asking!",
                'context': "greeting"
            },
            {
                'text': "What is artificial intelligence?",
                'response': "AI is the simulation of human intelligence in machines.",
                'context': "knowledge"
            },
            {
                'text': "Can you help me with math?",
                'response': "Of course! I'd be happy to help with math problems.",
                'context': "assistance"
            },
            {
                'text': "Tell me a story",
                'response': "Once upon a time, there was a brave little robot...",
                'context': "creative"
            },
            {
                'text': "What's the weather like?",
                'response': "I don't have access to real-time weather data.",
                'context': "information"
            }
        ]
        
        # Convert to OCR format
        for conv in conversations:
            ocr_data = self._convert_to_ocr_format(conv)
            self.data.append(ocr_data)
    
    def _convert_to_ocr_format(self, conversation: Dict[str, str]) -> Dict[str, Any]:
        """Convert conversation to OCR format"""
        # Create OCR-optimized text
        ocr_text = self._preprocess_for_ocr(conversation['text'])
        ocr_response = self._preprocess_for_ocr(conversation['response'])
        
        # Create OCR images
        ocr_input_img = self._text_to_ocr_image(ocr_text)
        ocr_target_img = self._text_to_ocr_image(ocr_response)
        
        return {
            'input_text': conversation['text'],
            'target_text': conversation['response'],
            'ocr_input': ocr_input_img,
            'ocr_target': ocr_target_img,
            'context': conversation['context']
        }
    
    def _preprocess_for_ocr(self, text: str) -> str:
        """Preprocess text for optimal OCR recognition"""
        # Convert to uppercase
        text = text.upper()
        
        # Replace problematic characters
        replacements = {
            '0': 'O', '1': 'I', '5': 'S', '8': 'B',
            ' ': '  ',  # Double spaces
            '?': ' ?', '!': ' !', '.': ' .',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _text_to_ocr_image(self, text: str) -> Image.Image:
        """Convert text to OCR-optimized image"""
        # Create image
        img = Image.new('RGB', (self.config.ocr_image_width, self.config.ocr_image_height), 
                       color='white')
        
        # Draw text
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 
                                self.config.ocr_font_size)
        draw.text((10, 10), text, fill='black', font=font)
        
        # Apply OCR optimization
        img_array = np.array(img)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Enhance contrast
        img_array = cv2.equalizeHist(img_array)
        _, img_array = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY)
        
        return Image.fromarray(img_array)
    
    def load_from_file(self, data_path: str):
        """Load data from file"""
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        for item in data:
            ocr_data = self._convert_to_ocr_format(item)
            self.data.append(ocr_data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class OCRNativeTrainer:
    """Trainer for OCR-native LLM"""
    
    def __init__(self, model: OCRNativeLLM, config: OCRNativeConfig):
        self.model = model
        self.config = config
        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # OCR-specific loss weights
        self.ocr_loss_weight = 0.3
        self.text_loss_weight = 0.7
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_ocr_loss = 0.0
        total_text_loss = 0.0
        
        for batch in tqdm(dataloader, desc="Training"):
            # Get batch data
            ocr_inputs = batch['ocr_input']
            ocr_targets = batch['ocr_target']
            input_texts = batch['input_text']
            target_texts = batch['target_text']
            
            # Prepare inputs
            inputs = {
                'text': input_texts,
                'ocr_input': ocr_inputs
            }
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Compute losses
            text_loss = self._compute_text_loss(outputs['text_logits'], target_texts)
            ocr_loss = self._compute_ocr_loss(outputs['ocr_output'], ocr_targets)
            
            # Combined loss
            total_loss_batch = (self.text_loss_weight * text_loss + 
                              self.ocr_loss_weight * ocr_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += total_loss_batch.item()
            total_ocr_loss += ocr_loss.item()
            total_text_loss += text_loss.item()
        
        return {
            'total_loss': total_loss / len(dataloader),
            'ocr_loss': total_ocr_loss / len(dataloader),
            'text_loss': total_text_loss / len(dataloader)
        }
    
    def _compute_text_loss(self, logits: torch.Tensor, targets: List[str]) -> torch.Tensor:
        """Compute text generation loss"""
        # Simplified text loss computation
        # In practice, would use proper tokenization
        batch_size = logits.shape[0]
        target_tokens = torch.randint(0, self.config.vocab_size, (batch_size, logits.shape[1]))
        
        return self.criterion(logits.view(-1, logits.size(-1)), target_tokens.view(-1))
    
    def _compute_ocr_loss(self, ocr_output: torch.Tensor, ocr_targets: List[Image.Image]) -> torch.Tensor:
        """Compute OCR reconstruction loss"""
        # Convert target images to tensors
        target_tensors = []
        for img in ocr_targets:
            img_array = np.array(img.convert('L')).flatten()
            target_tensor = torch.tensor(img_array, dtype=torch.float32)
            target_tensors.append(target_tensor)
        
        # Pad to same length
        max_len = max(len(t) for t in target_tensors)
        padded_targets = []
        for t in target_tensors:
            if len(t) < max_len:
                t = F.pad(t, (0, max_len - len(t)))
            padded_targets.append(t)
        
        target_tensor = torch.stack(padded_targets)
        
        # Compute MSE loss
        return F.mse_loss(ocr_output, target_tensor)
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Get batch data
                ocr_inputs = batch['ocr_input']
                input_texts = batch['input_text']
                target_texts = batch['target_text']
                
                # Prepare inputs
                inputs = {
                    'text': input_texts,
                    'ocr_input': ocr_inputs
                }
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss
                text_loss = self._compute_text_loss(outputs['text_logits'], target_texts)
                total_loss += text_loss.item()
        
        return {'eval_loss': total_loss / len(dataloader)}


def train_ocr_native_llm(config: OCRNativeConfig, 
                         data_path: str = None,
                         output_dir: str = "./ocr_native_checkpoints",
                         num_epochs: int = 10):
    """Main training function"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model
    model = OCRNativeLLM(config)
    
    # Create dataset
    dataset = OCRNativeDataset(config, data_path)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Create trainer
    trainer = OCRNativeTrainer(model, config)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_metrics = trainer.train_epoch(dataloader)
        print(f"Train Loss: {train_metrics['total_loss']:.4f}")
        print(f"  - Text Loss: {train_metrics['text_loss']:.4f}")
        print(f"  - OCR Loss: {train_metrics['ocr_loss']:.4f}")
        
        # Evaluate
        eval_metrics = trainer.evaluate(dataloader)
        print(f"Eval Loss: {eval_metrics['eval_loss']:.4f}")
        
        # Save checkpoint
        if eval_metrics['eval_loss'] < best_loss:
            best_loss = eval_metrics['eval_loss']
            checkpoint_path = os.path.join(output_dir, f"best_model_epoch_{epoch+1}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'loss': eval_metrics['eval_loss']
            }, checkpoint_path)
            print(f"New best model saved: {checkpoint_path}")
        
        # Store weights as OCR
        if epoch % 2 == 0:  # Every 2 epochs
            ocr_weights = model.store_weights_as_ocr()
            ocr_weights_path = os.path.join(output_dir, f"ocr_weights_epoch_{epoch+1}.json")
            
            # Save OCR weights metadata
            weights_metadata = {
                'epoch': epoch + 1,
                'layers': list(ocr_weights.keys()),
                'total_parameters': sum(p.numel() for p in model.parameters())
            }
            
            with open(ocr_weights_path, 'w') as f:
                json.dump(weights_metadata, f, indent=2)
            
            print(f"OCR weights stored: {ocr_weights_path}")
    
    print(f"\nTraining completed! Best loss: {best_loss:.4f}")
    return model


if __name__ == "__main__":
    # Configuration
    config = OCRNativeConfig(
        d_model=512,
        n_layers=8,
        n_heads=8,
        d_ff=2048,
        vocab_size=50000,
        max_seq_length=8192,
        learning_rate=1e-4,
        batch_size=4
    )
    
    # Train model
    model = train_ocr_native_llm(
        config=config,
        output_dir="./ocr_native_checkpoints",
        num_epochs=5
    )
    
    # Test model
    print("\nTesting OCR-native LLM...")
    
    # Example inputs
    inputs = {
        'text': "Hello, how are you?",
        'speech': np.random.randn(16000),
        'image': Image.new('RGB', (224, 224), color='white')
    }
    
    # Generate response
    response = model.generate_response(inputs, "Tell me about AI")
    print(f"Response: {response}")
    
    # Store weights as OCR
    ocr_weights = model.store_weights_as_ocr()
    print(f"Stored {len(ocr_weights)} weight layers as OCR images")
    
    # Add to memory
    memory_id = model.add_to_memory("AI is artificial intelligence", "knowledge", 0.9)
    print(f"Added to memory: {memory_id}")
    
    print("OCR-native LLM training completed!")