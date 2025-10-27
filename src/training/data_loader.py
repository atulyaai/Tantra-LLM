"""
Data Loaders for Tantra Conversational Speech Training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Any, Optional, Tuple
import random
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConversationDataset(Dataset):
    """Dataset for conversational training data"""
    
    def __init__(self, data_path: str, config, tokenizer=None):
        self.data_path = data_path
        self.config = config
        self.tokenizer = tokenizer
        self.conversations = []
        self.load_conversations()
    
    def load_conversations(self):
        """Load conversation data from files"""
        if not os.path.exists(self.data_path):
            logger.warning(f"Conversation data path not found: {self.data_path}")
            self.conversations = self._generate_sample_conversations()
            return
        
        for file_path in Path(self.data_path).glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.conversations.extend(data)
                    elif isinstance(data, dict):
                        self.conversations.append(data)
            except Exception as e:
                logger.error(f"Error loading conversation file {file_path}: {e}")
        
        if not self.conversations:
            logger.warning("No conversations loaded, generating sample data")
            self.conversations = self._generate_sample_conversations()
        
        logger.info(f"Loaded {len(self.conversations)} conversations")
    
    def _generate_sample_conversations(self) -> List[Dict]:
        """Generate sample conversation data for training"""
        sample_conversations = [
            {
                "conversation_id": "sample_001",
                "type": "general_chat",
                "personality": "helpful",
                "messages": [
                    {"role": "user", "content": "Hello, how are you today?"},
                    {"role": "assistant", "content": "Hello! I'm doing well, thank you for asking. I'm here to help you with any questions or tasks you might have. How can I assist you today?"}
                ],
                "context": "Initial greeting and offer of assistance"
            },
            {
                "conversation_id": "sample_002", 
                "type": "technical_support",
                "personality": "knowledgeable",
                "messages": [
                    {"role": "user", "content": "I'm having trouble with my computer running slowly"},
                    {"role": "assistant", "content": "I'd be happy to help you troubleshoot your computer performance issues. Let's start by identifying the potential causes. Can you tell me what specific tasks seem to be running slowly, and when did you first notice the slowdown?"}
                ],
                "context": "Technical support for computer performance"
            },
            {
                "conversation_id": "sample_003",
                "type": "creative_writing", 
                "personality": "creative",
                "messages": [
                    {"role": "user", "content": "Can you help me write a short story about a robot?"},
                    {"role": "assistant", "content": "I'd love to help you create a story about a robot! Let me craft something engaging for you:\n\n'The Last Guardian'\n\nIn the year 2157, ARIA-7 stood alone in the abandoned laboratory, her metallic fingers gently tracing the dust-covered control panels. She was the last of her kind—a caretaker robot designed to maintain the facility long after humans had departed for the stars. Her programming told her to wait, but her curiosity circuits had begun to wonder: what lay beyond the laboratory walls?'\n\nWould you like me to continue the story or explore a different angle?"}
                ],
                "context": "Creative writing assistance"
            },
            {
                "conversation_id": "sample_004",
                "type": "problem_solving",
                "personality": "analytical", 
                "messages": [
                    {"role": "user", "content": "I need to optimize my daily schedule to be more productive"},
                    {"role": "assistant", "content": "Great question! Let's approach this systematically. First, I'd like to understand your current situation better. Could you tell me:\n\n1. What are your main daily responsibilities?\n2. What times of day do you feel most energetic?\n3. What activities tend to take longer than expected?\n4. Do you have any fixed commitments (meetings, classes, etc.)?\n\nOnce I understand your current setup, I can suggest specific optimization strategies like time-blocking, the Pomodoro Technique, or priority matrix methods."}
                ],
                "context": "Productivity optimization consultation"
            },
            {
                "conversation_id": "sample_005",
                "type": "emotional_support",
                "personality": "empathetic",
                "messages": [
                    {"role": "user", "content": "I've been feeling really stressed lately with work and personal life"},
                    {"role": "assistant", "content": "I'm sorry to hear you're going through a stressful time. It sounds like you're dealing with a lot on your plate right now. That can be really overwhelming, and it's completely understandable to feel this way.\n\nSometimes it helps to break things down into smaller, more manageable pieces. Would you like to talk about what specific aspects are causing you the most stress? Sometimes just talking through it can help, and I'm here to listen and support you."}
                ],
                "context": "Emotional support and stress management"
            }
        ]
        
        # Generate more variations
        for i in range(20):
            base_conv = random.choice(sample_conversations)
            new_conv = base_conv.copy()
            new_conv["conversation_id"] = f"generated_{i:03d}"
            sample_conversations.append(new_conv)
        
        return sample_conversations
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        
        # Extract conversation context
        context = conversation.get("context", "")
        conv_type = conversation.get("type", "general_chat")
        personality = conversation.get("personality", "helpful")
        
        # Get messages
        messages = conversation.get("messages", [])
        if len(messages) < 2:
            # Pad with empty messages if needed
            messages = [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]
        
        # Get the last user message and assistant response
        user_message = ""
        assistant_response = ""
        
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                user_message = msg["content"]
            elif msg["role"] == "assistant":
                assistant_response = msg["content"]
                break
        
        # Create training example
        example = {
            "conversation_id": conversation.get("conversation_id", f"conv_{idx}"),
            "context": context,
            "conversation_type": conv_type,
            "personality": personality,
            "user_message": user_message,
            "assistant_response": assistant_response,
            "full_conversation": messages
        }
        
        return example


class SpeechDataset(Dataset):
    """Dataset for speech training data"""
    
    def __init__(self, data_path: str, config):
        self.data_path = data_path
        self.config = config
        self.speech_samples = []
        self.load_speech_data()
    
    def load_speech_data(self):
        """Load speech data from files"""
        if not os.path.exists(self.data_path):
            logger.warning(f"Speech data path not found: {self.data_path}")
            self.speech_samples = self._generate_sample_speech_data()
            return
        
        # Load audio files
        for file_path in Path(self.data_path).glob("*.wav"):
            try:
                # Load audio
                audio, sr = librosa.load(str(file_path), sr=self.config.speech_sample_rate)
                
                # Load corresponding text if available
                text_file = file_path.with_suffix('.txt')
                text = ""
                if text_file.exists():
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                
                self.speech_samples.append({
                    "audio_path": str(file_path),
                    "audio": audio,
                    "sample_rate": sr,
                    "text": text,
                    "duration": len(audio) / sr
                })
                
            except Exception as e:
                logger.error(f"Error loading speech file {file_path}: {e}")
        
        if not self.speech_samples:
            logger.warning("No speech samples loaded, generating sample data")
            self.speech_samples = self._generate_sample_speech_data()
        
        logger.info(f"Loaded {len(self.speech_samples)} speech samples")
    
    def _generate_sample_speech_data(self) -> List[Dict]:
        """Generate sample speech data for training"""
        sample_texts = [
            "Hello, welcome to Tantra conversational AI",
            "How can I help you today?",
            "I'm here to assist with your questions",
            "Let me think about that for a moment",
            "That's an interesting question",
            "I understand your concern",
            "Let me provide you with some information",
            "Is there anything else I can help with?",
            "Thank you for your patience",
            "I hope that was helpful"
        ]
        
        speech_samples = []
        for i, text in enumerate(sample_texts):
            # Generate synthetic audio (sine wave for demo)
            duration = random.uniform(1.0, 5.0)
            t = np.linspace(0, duration, int(self.config.speech_sample_rate * duration))
            frequency = 440 + i * 50  # Vary frequency
            audio = np.sin(2 * np.pi * frequency * t) * 0.3
            
            # Add some noise
            noise = np.random.normal(0, 0.05, len(audio))
            audio = audio + noise
            
            speech_samples.append({
                "audio_path": f"sample_{i:03d}.wav",
                "audio": audio,
                "sample_rate": self.config.speech_sample_rate,
                "text": text,
                "duration": duration
            })
        
        return speech_samples
    
    def __len__(self):
        return len(self.speech_samples)
    
    def __getitem__(self, idx):
        sample = self.speech_samples[idx]
        
        # Process audio
        audio = sample["audio"]
        if len(audio) > int(self.config.speech_sample_rate * self.config.speech_max_duration):
            # Truncate if too long
            max_samples = int(self.config.speech_sample_rate * self.config.speech_max_duration)
            audio = audio[:max_samples]
        elif len(audio) < int(self.config.speech_sample_rate * 0.5):
            # Pad if too short
            min_samples = int(self.config.speech_sample_rate * 0.5)
            audio = np.pad(audio, (0, min_samples - len(audio)), mode='constant')
        
        # Convert to mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.config.speech_sample_rate,
            n_fft=self.config.speech_n_fft,
            hop_length=self.config.speech_hop_length,
            n_mels=self.config.speech_n_mels,
            fmin=self.config.speech_f_min,
            fmax=self.config.speech_f_max
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return {
            "audio": torch.tensor(audio, dtype=torch.float32),
            "mel_spectrogram": torch.tensor(mel_spec, dtype=torch.float32),
            "text": sample["text"],
            "duration": sample["duration"],
            "sample_rate": sample["sample_rate"]
        }


class ConversationDataLoader:
    """Data loader for conversation training"""
    
    def __init__(self, config, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer
        
        # Create datasets
        self.train_dataset = ConversationDataset(
            os.path.join(config.conversation_data_path, "train"),
            config,
            tokenizer
        )
        
        self.val_dataset = ConversationDataset(
            os.path.join(config.conversation_data_path, "val"),
            config,
            tokenizer
        )
    
    def get_train_loader(self) -> DataLoader:
        """Get training data loader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_conversations
        )
    
    def get_val_loader(self) -> DataLoader:
        """Get validation data loader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_conversations
        )
    
    def _collate_conversations(self, batch):
        """Collate function for conversation batches"""
        return {
            "conversation_ids": [item["conversation_id"] for item in batch],
            "contexts": [item["context"] for item in batch],
            "conversation_types": [item["conversation_type"] for item in batch],
            "personalities": [item["personality"] for item in batch],
            "user_messages": [item["user_message"] for item in batch],
            "assistant_responses": [item["assistant_response"] for item in batch],
            "full_conversations": [item["full_conversation"] for item in batch]
        }


class SpeechDataLoader:
    """Data loader for speech training"""
    
    def __init__(self, config):
        self.config = config
        
        # Create datasets
        self.train_dataset = SpeechDataset(
            os.path.join(config.speech_data_path, "train"),
            config
        )
        
        self.val_dataset = SpeechDataset(
            os.path.join(config.speech_data_path, "val"),
            config
        )
    
    def get_train_loader(self) -> DataLoader:
        """Get training data loader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_speech
        )
    
    def get_val_loader(self) -> DataLoader:
        """Get validation data loader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_speech
        )
    
    def _collate_speech(self, batch):
        """Collate function for speech batches"""
        # Pad audio sequences to same length
        max_audio_len = max(len(item["audio"]) for item in batch)
        padded_audio = []
        
        for item in batch:
            audio = item["audio"]
            if len(audio) < max_audio_len:
                audio = torch.nn.functional.pad(audio, (0, max_audio_len - len(audio)))
            padded_audio.append(audio)
        
        # Pad mel spectrograms
        max_mel_len = max(item["mel_spectrogram"].shape[1] for item in batch)
        padded_mel_specs = []
        
        for item in batch:
            mel_spec = item["mel_spectrogram"]
            if mel_spec.shape[1] < max_mel_len:
                mel_spec = torch.nn.functional.pad(mel_spec, (0, max_mel_len - mel_spec.shape[1]))
            padded_mel_specs.append(mel_spec)
        
        return {
            "audio": torch.stack(padded_audio),
            "mel_spectrograms": torch.stack(padded_mel_specs),
            "texts": [item["text"] for item in batch],
            "durations": [item["duration"] for item in batch],
            "sample_rates": [item["sample_rate"] for item in batch]
        }