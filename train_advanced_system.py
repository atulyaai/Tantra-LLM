#!/usr/bin/env python3
"""
Advanced Tantra Training System
Complete training pipeline with web scraping, search, ROPE RAG, and LoRA
"""

import sys
import os
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.training_config import TrainingConfig
from src.training.conversational_trainer import ConversationalTrainer
from src.training.speech_trainer import SpeechTrainer
from src.training.lora_trainer import LoRATrainer, LoRAConfig
from src.web.web_scraper import WebScraper, ScrapingConfig
from src.web.web_search import WebSearch, SearchConfig
from src.rag.rope_rag import ROPERAG, RAGConfig
from src.versioning.model_versioning import ModelVersionManager
from src.versioning.version_config import VersionConfig
from src.core.tantra_llm import TantraLLM, TantraConfig
from src.utils.error_handler import logger


class AdvancedTantraTrainer:
    """Advanced training system with all capabilities"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self._initialize_components()
        
        # Initialize versioning
        self.version_config = VersionConfig()
        self.model_version_manager = ModelVersionManager(self.version_config)
        
        logger.info("AdvancedTantraTrainer initialized")
    
    def _initialize_components(self):
        """Initialize all training components"""
        # Base model
        tantra_config = TantraConfig()
        self.base_model = TantraLLM(tantra_config).to(self.device)
        
        # Web scraping
        scraping_config = ScrapingConfig(
            extract_text=True,
            extract_metadata=True,
            min_content_length=100,
            max_content_length=5000
        )
        self.web_scraper = WebScraper(scraping_config)
        
        # Web search
        search_config = SearchConfig(
            max_results=20,
            language="en"
        )
        self.web_search = WebSearch(search_config)
        
        # ROPE RAG
        rag_config = RAGConfig(
            model_name="tantra_rope_rag",
            retrieval_top_k=5,
            knowledge_base_path="knowledge_base"
        )
        self.rope_rag = ROPERAG(rag_config, self.base_model)
        
        # LoRA training
        lora_config = LoRAConfig(
            rank=16,
            alpha=32.0,
            learning_rate=1e-4,
            lora_learning_rate=1e-3
        )
        self.lora_trainer = LoRATrainer(self.base_model, lora_config)
        
        # Conversational trainer
        self.conv_trainer = ConversationalTrainer(self.config, self.base_model)
        
        # Speech trainer
        self.speech_trainer = SpeechTrainer(self.config, self.base_model)
    
    def collect_web_data(self, queries: List[str], max_pages_per_query: int = 5) -> List[Dict[str, Any]]:
        """Collect data from web using search and scraping"""
        logger.info(f"Collecting web data for {len(queries)} queries")
        
        all_web_data = []
        
        for query in queries:
            logger.info(f"Processing query: {query}")
            
            try:
                # Search for query
                search_results = self.web_search.search(query, max_results=max_pages_per_query)
                
                if not search_results:
                    logger.warning(f"No search results for query: {query}")
                    continue
                
                # Scrape content from search results
                urls = [result['url'] for result in search_results]
                scraped_content = self.web_scraper.scrape_urls(urls)
                
                # Add query context to scraped content
                for content in scraped_content:
                    if content.get('success', False):
                        content['query'] = query
                        content['search_rank'] = next(
                            (i for i, result in enumerate(search_results) 
                             if result['url'] == content['url']), 0
                        )
                
                all_web_data.extend(scraped_content)
                
                # Add delay between queries
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed to collect data for query '{query}': {e}")
                continue
        
        logger.info(f"Collected {len(all_web_data)} web documents")
        return all_web_data
    
    def build_knowledge_base(self, web_data: List[Dict[str, Any]], additional_data: List[Dict[str, Any]] = None):
        """Build knowledge base from web data and additional sources"""
        logger.info("Building knowledge base...")
        
        # Add web data to ROPE RAG
        self.rope_rag.add_web_content(web_data)
        
        # Add additional data if provided
        if additional_data:
            from src.rag.knowledge_base import Document
            documents = []
            
            for item in additional_data:
                doc = Document(
                    id=item.get('id', ''),
                    content=item.get('content', ''),
                    metadata=item.get('metadata', {})
                )
                documents.append(doc)
            
            self.rope_rag.add_documents(documents)
        
        logger.info(f"Knowledge base built with {len(self.rope_rag.knowledge_base.documents)} documents")
    
    def train_rope_rag(self, training_data: List[Dict[str, Any]]):
        """Train ROPE RAG system"""
        logger.info("Training ROPE RAG system...")
        
        try:
            self.rope_rag.train(training_data)
            logger.info("ROPE RAG training completed")
        except Exception as e:
            logger.error(f"ROPE RAG training failed: {e}")
            raise
    
    def train_lora(self, training_data: List[Dict[str, Any]]):
        """Train with LoRA fine-tuning"""
        logger.info("Training with LoRA...")
        
        try:
            # Prepare training data for LoRA
            lora_data = self._prepare_lora_data(training_data)
            
            # Create data loader
            from torch.utils.data import DataLoader, Dataset
            
            class LoRADataset(Dataset):
                def __init__(self, data):
                    self.data = data
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    return self.data[idx]
            
            dataset = LoRADataset(lora_data)
            train_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
            
            # Train LoRA
            self.lora_trainer.train(train_loader)
            
            logger.info("LoRA training completed")
            
        except Exception as e:
            logger.error(f"LoRA training failed: {e}")
            raise
    
    def train_conversational(self):
        """Train conversational capabilities"""
        logger.info("Training conversational capabilities...")
        
        try:
            self.conv_trainer.train()
            logger.info("Conversational training completed")
        except Exception as e:
            logger.error(f"Conversational training failed: {e}")
            raise
    
    def train_speech(self):
        """Train speech capabilities"""
        logger.info("Training speech capabilities...")
        
        try:
            self.speech_trainer.train()
            logger.info("Speech training completed")
        except Exception as e:
            logger.error(f"Speech training failed: {e}")
            raise
    
    def _prepare_lora_data(self, training_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare data for LoRA training"""
        lora_data = []
        
        for item in training_data:
            # Extract text content
            text = item.get('content', '')
            if not text:
                continue
            
            # Create training example
            example = {
                'input_ids': text,
                'labels': text,  # Self-supervised learning
                'metadata': item.get('metadata', {})
            }
            lora_data.append(example)
        
        return lora_data
    
    def evaluate_system(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate the complete system"""
        logger.info("Evaluating system performance...")
        
        evaluation_results = {
            'rope_rag': {},
            'conversational': {},
            'speech': {},
            'lora': {},
            'overall': {}
        }
        
        try:
            # Evaluate ROPE RAG
            if hasattr(self.rope_rag, 'evaluate'):
                rag_metrics = self.rope_rag.evaluate(test_data)
                evaluation_results['rope_rag'] = rag_metrics
            
            # Evaluate conversational
            if hasattr(self.conv_trainer, 'evaluate_conversation_quality'):
                conv_metrics = self.conv_trainer.evaluate_conversation_quality(test_data)
                evaluation_results['conversational'] = conv_metrics
            
            # Evaluate speech
            if hasattr(self.speech_trainer, 'evaluate_speech_quality'):
                speech_metrics = self.speech_trainer.evaluate_speech_quality(test_data)
                evaluation_results['speech'] = speech_metrics
            
            # Calculate overall metrics
            evaluation_results['overall'] = self._calculate_overall_metrics(evaluation_results)
            
            logger.info(f"System evaluation completed: {evaluation_results['overall']}")
            
        except Exception as e:
            logger.error(f"System evaluation failed: {e}")
            evaluation_results['error'] = str(e)
        
        return evaluation_results
    
    def _calculate_overall_metrics(self, component_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall system metrics"""
        overall = {
            'total_components': len(component_metrics),
            'successful_components': 0,
            'average_performance': 0.0
        }
        
        total_performance = 0.0
        successful_components = 0
        
        for component, metrics in component_metrics.items():
            if component == 'overall':
                continue
            
            if metrics and not metrics.get('error'):
                successful_components += 1
                
                # Calculate component performance (simplified)
                if 'accuracy' in metrics:
                    total_performance += metrics['accuracy']
                elif 'avg_relevance_score' in metrics:
                    total_performance += metrics['avg_relevance_score']
                elif 'avg_reconstruction_mse' in metrics:
                    # Lower is better for MSE
                    total_performance += max(0, 1 - metrics['avg_reconstruction_mse'])
        
        overall['successful_components'] = successful_components
        overall['average_performance'] = total_performance / max(successful_components, 1)
        
        return overall
    
    def save_models(self, save_path: str):
        """Save all trained models"""
        logger.info(f"Saving models to {save_path}")
        
        try:
            # Create save directory
            os.makedirs(save_path, exist_ok=True)
            
            # Save base model
            base_model_path = os.path.join(save_path, "base_model.pt")
            torch.save(self.base_model.state_dict(), base_model_path)
            
            # Save ROPE RAG
            rag_path = os.path.join(save_path, "rope_rag")
            self.rope_rag.save(rag_path)
            
            # Save LoRA checkpoint
            lora_path = os.path.join(save_path, "lora_checkpoint.pt")
            self.lora_trainer.save_checkpoint(lora_path)
            
            # Save conversational model
            conv_path = os.path.join(save_path, "conversational_model.pt")
            torch.save(self.conv_trainer.model.state_dict(), conv_path)
            
            # Save speech model
            speech_path = os.path.join(save_path, "speech_model.pt")
            torch.save(self.speech_trainer.model.state_dict(), speech_path)
            
            # Save training configuration
            config_path = os.path.join(save_path, "training_config.json")
            with open(config_path, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
            
            logger.info("All models saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            raise
    
    def run_complete_training(self, web_queries: List[str], training_data: List[Dict[str, Any]]):
        """Run complete training pipeline"""
        logger.info("Starting complete training pipeline...")
        
        start_time = time.time()
        
        try:
            # 1. Collect web data
            web_data = self.collect_web_data(web_queries)
            
            # 2. Build knowledge base
            self.build_knowledge_base(web_data, training_data)
            
            # 3. Train ROPE RAG
            self.train_rope_rag(training_data)
            
            # 4. Train LoRA
            self.train_lora(training_data)
            
            # 5. Train conversational
            self.train_conversational()
            
            # 6. Train speech
            self.train_speech()
            
            # 7. Evaluate system
            evaluation_results = self.evaluate_system(training_data[:10])  # Evaluate on subset
            
            # 8. Save models
            self.save_models("trained_models")
            
            # 9. Create version
            version_id = self.model_version_manager.create_version(
                model_path="trained_models/base_model.pt",
                version_type="complete_system",
                description="Complete Tantra system with web data, ROPE RAG, and LoRA",
                metadata={
                    'web_queries': web_queries,
                    'training_data_size': len(training_data),
                    'evaluation_results': evaluation_results,
                    'training_time': time.time() - start_time
                }
            )
            
            logger.info(f"Complete training pipeline finished in {time.time() - start_time:.2f} seconds")
            logger.info(f"Model version created: {version_id}")
            
            return {
                'success': True,
                'version_id': version_id,
                'training_time': time.time() - start_time,
                'evaluation_results': evaluation_results
            }
            
        except Exception as e:
            logger.error(f"Complete training pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'training_time': time.time() - start_time
            }


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Advanced Tantra Training System")
    parser.add_argument("--config", type=str, help="Path to training config file")
    parser.add_argument("--web-queries", type=str, nargs="+", 
                       help="Web search queries for data collection")
    parser.add_argument("--training-data", type=str, 
                       help="Path to training data file")
    parser.add_argument("--skip-web", action="store_true", 
                       help="Skip web data collection")
    parser.add_argument("--skip-rag", action="store_true", 
                       help="Skip ROPE RAG training")
    parser.add_argument("--skip-lora", action="store_true", 
                       help="Skip LoRA training")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('advanced_training.log'),
            logging.StreamHandler()
        ]
    )
    
    try:
        # Load configuration
        if args.config and os.path.exists(args.config):
            config = TrainingConfig.load_config(args.config)
        else:
            config = TrainingConfig()
        
        # Initialize trainer
        trainer = AdvancedTantraTrainer(config)
        
        # Prepare web queries
        web_queries = args.web_queries or [
            "artificial intelligence",
            "machine learning",
            "natural language processing",
            "conversational AI",
            "speech recognition",
            "transformer models",
            "deep learning",
            "neural networks"
        ]
        
        # Prepare training data
        training_data = []
        if args.training_data and os.path.exists(args.training_data):
            with open(args.training_data, 'r') as f:
                training_data = json.load(f)
        else:
            # Generate sample training data
            training_data = [
                {
                    'id': f'sample_{i}',
                    'content': f'This is sample training data {i} for Tantra system.',
                    'metadata': {'source': 'generated', 'type': 'sample'}
                }
                for i in range(100)
            ]
        
        # Run training
        if args.skip_web:
            web_queries = []
        
        results = trainer.run_complete_training(web_queries, training_data)
        
        if results['success']:
            print("\n" + "="*60)
            print("🎉 ADVANCED TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Version ID: {results['version_id']}")
            print(f"Training Time: {results['training_time']:.2f} seconds")
            print(f"Evaluation Results: {results['evaluation_results']['overall']}")
            print("\nModels saved to: trained_models/")
            print("="*60)
        else:
            print(f"\n❌ Training failed: {results['error']}")
            return 1
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())