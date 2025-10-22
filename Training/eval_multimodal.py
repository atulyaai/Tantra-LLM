"""
Multi-Modal Mamba 3 Evaluation Suite
Comprehensive evaluation for audio, text, and vision capabilities
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from model_mamba3_multimodal import Mamba3MultiModal, Mamba3Config
import safetensors.torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalEvaluator:
    """Comprehensive evaluator for multi-modal Mamba 3 model"""
    
    def __init__(self, model: Mamba3MultiModal, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Evaluation metrics storage
        self.metrics = {
            "audio": {},
            "text": {},
            "vision": {},
            "multimodal": {}
        }
    
    def evaluate_audio_quality(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate audio reconstruction quality"""
        logger.info("Evaluating audio quality...")
        
        mse_losses = []
        snr_values = []
        
        with torch.no_grad():
            for sample in tqdm(test_data, desc="Audio evaluation"):
                # Prepare input
                audio_input = torch.tensor(sample["audio_features"], dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Forward pass
                inputs = {"audio": audio_input}
                outputs = self.model(inputs, modality_priority=["audio"])
                
                # Calculate MSE loss
                mse_loss = F.mse_loss(outputs["audio"], audio_input)
                mse_losses.append(mse_loss.item())
                
                # Calculate SNR
                signal_power = torch.mean(audio_input ** 2)
                noise_power = torch.mean((outputs["audio"] - audio_input) ** 2)
                snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
                snr_values.append(snr.item())
        
        audio_metrics = {
            "mse_loss": np.mean(mse_losses),
            "snr_db": np.mean(snr_values),
            "reconstruction_quality": min(1.0, max(0.0, 1.0 - np.mean(mse_losses)))
        }
        
        self.metrics["audio"] = audio_metrics
        return audio_metrics
    
    def evaluate_text_generation(self, test_data: List[Dict[str, Any]], 
                                tokenizer) -> Dict[str, float]:
        """Evaluate text generation capabilities"""
        logger.info("Evaluating text generation...")
        
        perplexities = []
        accuracies = []
        bleu_scores = []
        
        with torch.no_grad():
            for sample in tqdm(test_data, desc="Text evaluation"):
                # Prepare input
                text_input = torch.tensor(sample["text_tokens"], dtype=torch.long).unsqueeze(0).to(self.device)
                
                # Forward pass
                inputs = {"text": text_input}
                outputs = self.model(inputs, modality_priority=["text"])
                
                # Calculate perplexity
                logits = outputs["text"]
                target = text_input
                
                # Cross-entropy loss
                ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
                perplexity = torch.exp(ce_loss).item()
                perplexities.append(perplexity)
                
                # Accuracy
                predictions = torch.argmax(logits, dim=-1)
                accuracy = (predictions == target).float().mean().item()
                accuracies.append(accuracy)
                
                # BLEU score (simplified)
                pred_tokens = predictions[0].cpu().tolist()
                target_tokens = target[0].cpu().tolist()
                bleu = self._calculate_bleu(pred_tokens, target_tokens)
                bleu_scores.append(bleu)
        
        text_metrics = {
            "perplexity": np.mean(perplexities),
            "accuracy": np.mean(accuracies),
            "bleu_score": np.mean(bleu_scores),
            "generation_quality": min(1.0, max(0.0, 1.0 - np.mean(perplexities) / 100.0))
        }
        
        self.metrics["text"] = text_metrics
        return text_metrics
    
    def evaluate_vision_analysis(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate vision analysis capabilities"""
        logger.info("Evaluating vision analysis...")
        
        mse_losses = []
        ssim_values = []
        
        with torch.no_grad():
            for sample in tqdm(test_data, desc="Vision evaluation"):
                # Prepare input
                vision_input = torch.tensor(sample["image_features"], dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Forward pass
                inputs = {"vision": vision_input}
                outputs = self.model(inputs, modality_priority=["vision"])
                
                # Calculate MSE loss
                mse_loss = F.mse_loss(outputs["vision"], vision_input)
                mse_losses.append(mse_loss.item())
                
                # Calculate SSIM (simplified)
                ssim = self._calculate_ssim(outputs["vision"], vision_input)
                ssim_values.append(ssim)
        
        vision_metrics = {
            "mse_loss": np.mean(mse_losses),
            "ssim": np.mean(ssim_values),
            "analysis_quality": min(1.0, max(0.0, np.mean(ssim_values)))
        }
        
        self.metrics["vision"] = vision_metrics
        return vision_metrics
    
    def evaluate_multimodal_fusion(self, test_data: List[Dict[str, Any]], 
                                  tokenizer) -> Dict[str, float]:
        """Evaluate multi-modal fusion capabilities"""
        logger.info("Evaluating multi-modal fusion...")
        
        fusion_scores = []
        cross_modal_consistency = []
        
        with torch.no_grad():
            for sample in tqdm(test_data, desc="Multi-modal evaluation"):
                # Prepare inputs
                inputs = {}
                if "audio_features" in sample:
                    inputs["audio"] = torch.tensor(sample["audio_features"], dtype=torch.float32).unsqueeze(0).to(self.device)
                if "text_tokens" in sample:
                    inputs["text"] = torch.tensor(sample["text_tokens"], dtype=torch.long).unsqueeze(0).to(self.device)
                if "image_features" in sample:
                    inputs["vision"] = torch.tensor(sample["image_features"], dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Forward pass with all modalities
                outputs = self.model(inputs, modality_priority=["audio", "text", "vision"])
                
                # Calculate fusion quality
                fusion_score = self._calculate_fusion_quality(outputs, inputs)
                fusion_scores.append(fusion_score)
                
                # Calculate cross-modal consistency
                consistency = self._calculate_cross_modal_consistency(outputs)
                cross_modal_consistency.append(consistency)
        
        multimodal_metrics = {
            "fusion_quality": np.mean(fusion_scores),
            "cross_modal_consistency": np.mean(cross_modal_consistency),
            "overall_multimodal_score": np.mean([np.mean(fusion_scores), np.mean(cross_modal_consistency)])
        }
        
        self.metrics["multimodal"] = multimodal_metrics
        return multimodal_metrics
    
    def evaluate_compression_efficiency(self) -> Dict[str, float]:
        """Evaluate compression efficiency"""
        logger.info("Evaluating compression efficiency...")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Count non-zero parameters (after pruning)
        non_zero_params = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                non_zero_params += (param != 0).sum().item()
        
        # Calculate compression ratio
        compression_ratio = non_zero_params / total_params if total_params > 0 else 1.0
        
        # Estimate memory usage
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        total_size = param_size + buffer_size
        
        compression_metrics = {
            "total_parameters": total_params,
            "non_zero_parameters": non_zero_params,
            "compression_ratio": compression_ratio,
            "model_size_mb": total_size / (1024 * 1024),
            "compression_efficiency": compression_ratio
        }
        
        return compression_metrics
    
    def evaluate_dynamic_vocabulary(self, test_texts: List[str]) -> Dict[str, float]:
        """Evaluate dynamic vocabulary capabilities"""
        logger.info("Evaluating dynamic vocabulary...")
        
        # Test vocabulary growth
        initial_vocab_size = self.model.dynamic_vocab.current_vocab_size
        
        # Process test texts
        for text in test_texts:
            # Simulate tokenization and vocabulary update
            tokens = text.split()  # Simple tokenization
            token_ids = [hash(token) % 100000 for token in tokens]  # Simulate token IDs
            self.model.dynamic_vocab.update_frequencies(token_ids)
        
        final_vocab_size = self.model.dynamic_vocab.current_vocab_size
        vocab_growth = final_vocab_size - initial_vocab_size
        
        vocabulary_metrics = {
            "initial_vocab_size": initial_vocab_size,
            "final_vocab_size": final_vocab_size,
            "vocab_growth": vocab_growth,
            "growth_rate": vocab_growth / initial_vocab_size if initial_vocab_size > 0 else 0
        }
        
        return vocabulary_metrics
    
    def _calculate_bleu(self, predictions: List[int], targets: List[int]) -> float:
        """Calculate simplified BLEU score"""
        # Simple n-gram overlap
        pred_ngrams = set(zip(predictions, predictions[1:]))
        target_ngrams = set(zip(targets, targets[1:]))
        
        if len(target_ngrams) == 0:
            return 0.0
        
        overlap = len(pred_ngrams.intersection(target_ngrams))
        precision = overlap / len(pred_ngrams) if len(pred_ngrams) > 0 else 0.0
        recall = overlap / len(target_ngrams)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def _calculate_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate simplified SSIM"""
        # Simplified SSIM calculation
        mu_pred = torch.mean(pred)
        mu_target = torch.mean(target)
        
        sigma_pred = torch.var(pred)
        sigma_target = torch.var(target)
        sigma_pred_target = torch.mean((pred - mu_pred) * (target - mu_target))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu_pred * mu_target + c1) * (2 * sigma_pred_target + c2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + c1) * (sigma_pred + sigma_target + c2))
        
        return ssim.item()
    
    def _calculate_fusion_quality(self, outputs: Dict[str, torch.Tensor], 
                                 inputs: Dict[str, torch.Tensor]) -> float:
        """Calculate multi-modal fusion quality"""
        # Simple fusion quality metric based on output consistency
        output_energies = []
        for modality, output in outputs.items():
            energy = torch.mean(output ** 2).item()
            output_energies.append(energy)
        
        # Check if outputs are balanced (not too different in energy)
        if len(output_energies) < 2:
            return 1.0
        
        energy_std = np.std(output_energies)
        energy_mean = np.mean(output_energies)
        
        # Lower std/mean ratio indicates better fusion
        fusion_quality = 1.0 / (1.0 + energy_std / (energy_mean + 1e-8))
        return fusion_quality
    
    def _calculate_cross_modal_consistency(self, outputs: Dict[str, torch.Tensor]) -> float:
        """Calculate cross-modal consistency"""
        if len(outputs) < 2:
            return 1.0
        
        # Calculate correlation between different modality outputs
        output_list = list(outputs.values())
        correlations = []
        
        for i in range(len(output_list)):
            for j in range(i + 1, len(output_list)):
                # Flatten and calculate correlation
                flat1 = output_list[i].flatten()
                flat2 = output_list[j].flatten()
                
                # Ensure same length
                min_len = min(len(flat1), len(flat2))
                flat1 = flat1[:min_len]
                flat2 = flat2[:min_len]
                
                correlation = torch.corrcoef(torch.stack([flat1, flat2]))[0, 1]
                if not torch.isnan(correlation):
                    correlations.append(correlation.item())
        
        return np.mean(correlations) if correlations else 0.0
    
    def generate_evaluation_report(self, output_path: str = "evaluation_report.json"):
        """Generate comprehensive evaluation report"""
        logger.info("Generating evaluation report...")
        
        # Calculate overall scores
        overall_scores = {}
        for modality, metrics in self.metrics.items():
            if metrics:
                overall_scores[modality] = np.mean(list(metrics.values()))
        
        # Create report
        report = {
            "model_info": {
                "total_parameters": sum(p.numel() for p in self.model.parameters()),
                "model_size_mb": sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
            },
            "modality_scores": self.metrics,
            "overall_scores": overall_scores,
            "summary": {
                "best_modality": max(overall_scores.items(), key=lambda x: x[1])[0] if overall_scores else "none",
                "average_score": np.mean(list(overall_scores.values())) if overall_scores else 0.0
            }
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {output_path}")
        return report
    
    def plot_evaluation_results(self, output_dir: str = "evaluation_plots"):
        """Generate evaluation plots"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Plot modality scores
        if self.metrics:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle("Multi-Modal Evaluation Results")
            
            # Audio metrics
            if "audio" in self.metrics and self.metrics["audio"]:
                ax = axes[0, 0]
                metrics = self.metrics["audio"]
                ax.bar(metrics.keys(), metrics.values())
                ax.set_title("Audio Quality Metrics")
                ax.set_ylabel("Score")
                plt.setp(ax.get_xticklabels(), rotation=45)
            
            # Text metrics
            if "text" in self.metrics and self.metrics["text"]:
                ax = axes[0, 1]
                metrics = self.metrics["text"]
                ax.bar(metrics.keys(), metrics.values())
                ax.set_title("Text Generation Metrics")
                ax.set_ylabel("Score")
                plt.setp(ax.get_xticklabels(), rotation=45)
            
            # Vision metrics
            if "vision" in self.metrics and self.metrics["vision"]:
                ax = axes[1, 0]
                metrics = self.metrics["vision"]
                ax.bar(metrics.keys(), metrics.values())
                ax.set_title("Vision Analysis Metrics")
                ax.set_ylabel("Score")
                plt.setp(ax.get_xticklabels(), rotation=45)
            
            # Multi-modal metrics
            if "multimodal" in self.metrics and self.metrics["multimodal"]:
                ax = axes[1, 1]
                metrics = self.metrics["multimodal"]
                ax.bar(metrics.keys(), metrics.values())
                ax.set_title("Multi-Modal Fusion Metrics")
                ax.set_ylabel("Score")
                plt.setp(ax.get_xticklabels(), rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/modality_metrics.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Evaluation plots saved to {output_dir}")


def create_test_data(num_samples: int = 100) -> Dict[str, List[Dict[str, Any]]]:
    """Create test data for evaluation"""
    test_data = {
        "audio": [],
        "text": [],
        "vision": [],
        "multimodal": []
    }
    
    # Audio test data
    for i in range(num_samples):
        test_data["audio"].append({
            "audio_features": np.random.randn(128, 128).tolist()
        })
    
    # Text test data
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing helps computers understand human language.",
        "Deep learning models can process vast amounts of data.",
        "The future of AI holds many exciting possibilities."
    ]
    
    for i in range(num_samples):
        text = sample_texts[i % len(sample_texts)]
        test_data["text"].append({
            "text_tokens": [hash(word) % 1000 for word in text.split()]
        })
    
    # Vision test data
    for i in range(num_samples):
        test_data["vision"].append({
            "image_features": np.random.randn(196, 512).tolist()  # 14x14 patches
        })
    
    # Multi-modal test data
    for i in range(num_samples):
        test_data["multimodal"].append({
            "audio_features": np.random.randn(128, 128).tolist(),
            "text_tokens": [hash(word) % 1000 for word in sample_texts[i % len(sample_texts)].split()],
            "image_features": np.random.randn(196, 512).tolist()
        })
    
    return test_data


def main():
    """Main evaluation function"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    config = Mamba3Config()
    model = Mamba3MultiModal(config).to(device)
    
    # Load weights if available
    weights_path = 'Model/tantra_multimodal_weights.safetensors'
    if Path(weights_path).exists():
        logger.info(f"Loading weights from {weights_path}")
        state_dict = safetensors.torch.load_file(weights_path)
        model.load_state_dict(state_dict)
    else:
        logger.warning("No weights found, using random initialization")
    
    # Create evaluator
    evaluator = MultiModalEvaluator(model, device)
    
    # Create test data
    test_data = create_test_data(100)
    
    # Run evaluations
    logger.info("Starting comprehensive evaluation...")
    
    # Audio evaluation
    audio_metrics = evaluator.evaluate_audio_quality(test_data["audio"])
    logger.info(f"Audio metrics: {audio_metrics}")
    
    # Text evaluation
    text_metrics = evaluator.evaluate_text_generation(test_data["text"], None)
    logger.info(f"Text metrics: {text_metrics}")
    
    # Vision evaluation
    vision_metrics = evaluator.evaluate_vision_analysis(test_data["vision"])
    logger.info(f"Vision metrics: {vision_metrics}")
    
    # Multi-modal evaluation
    multimodal_metrics = evaluator.evaluate_multimodal_fusion(test_data["multimodal"], None)
    logger.info(f"Multi-modal metrics: {multimodal_metrics}")
    
    # Compression evaluation
    compression_metrics = evaluator.evaluate_compression_efficiency()
    logger.info(f"Compression metrics: {compression_metrics}")
    
    # Dynamic vocabulary evaluation
    test_texts = ["This is a test sentence for vocabulary evaluation."] * 10
    vocab_metrics = evaluator.evaluate_dynamic_vocabulary(test_texts)
    logger.info(f"Vocabulary metrics: {vocab_metrics}")
    
    # Generate report
    report = evaluator.generate_evaluation_report()
    
    # Generate plots
    evaluator.plot_evaluation_results()
    
    logger.info("Evaluation completed!")
    logger.info(f"Overall average score: {report['summary']['average_score']:.3f}")


if __name__ == "__main__":
    main()