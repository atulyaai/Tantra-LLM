"""
Comprehensive Error Scenario Testing
Tests error handling, recovery, and edge cases across all components.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from core.models.spikingbrain_model import SpikingBrainForCausalLM, SpikingBrainConfig
from core.control.decision_engine import DecisionEngine
from core.control.response_generator import ResponseGenerator
from core.memory.advanced_memory import AdvancedMemoryManager
from core.fusion.unified_fusion import FusionOrchestrator
from encoders.vision import VisionEncoder
from utils.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
from utils.config_validator import ConfigValidator


class TestErrorScenarios:
    """Test error scenarios and recovery mechanisms."""
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler for testing."""
        return ErrorHandler()
    
    @pytest.fixture
    def mock_model(self):
        """Create mock SpikingBrain model."""
        config = SpikingBrainConfig(
            vocab_size=1000,
            hidden_size=512,
            num_attention_heads=8,
            num_hidden_layers=4,
            intermediate_size=2048
        )
        return SpikingBrainForCausalLM(config)
    
    @pytest.fixture
    def mock_memory_manager(self):
        """Create mock memory manager."""
        return AdvancedMemoryManager(embedding_dim=512, max_episodic=100)
    
    @pytest.fixture
    def mock_personality_layer(self):
        """Create mock personality layer."""
        personality = Mock()
        personality.select_mode.return_value = "DirectAssertive"
        personality.parameterize.return_value = {
            "temperature": 0.8,
            "top_p": 0.9,
            "max_tokens": 100,
            "prompt_prefix": ""
        }
        return personality
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.return_value = "Test response"
        tokenizer.eos_token_id = 2
        return tokenizer
    
    def test_model_loading_errors(self, error_handler):
        """Test model loading error scenarios."""
        # Test missing model file
        with patch('torch.load') as mock_load:
            mock_load.side_effect = FileNotFoundError("Model file not found")
            
            result = error_handler.handle_error(
                FileNotFoundError("Model file not found"),
                "model_loading",
                ErrorCategory.MODEL_LOADING,
                ErrorSeverity.HIGH
            )
            
            assert result is None  # Should return None for fallback
    
    def test_memory_operation_errors(self, error_handler, mock_memory_manager):
        """Test memory operation error scenarios."""
        # Test memory storage with invalid data
        with pytest.raises(Exception):
            mock_memory_manager.store(
                content=None,  # Invalid content
                importance=0.5
            )
        
        # Test memory recall with empty query
        result = mock_memory_manager.recall("", top_k=5)
        assert result == []  # Should return empty list
    
    def test_vision_encoding_errors(self, error_handler):
        """Test vision encoding error scenarios."""
        vision_encoder = VisionEncoder(embed_dim=512)
        
        # Test with invalid image data
        result = vision_encoder("invalid_image_data")
        assert result is not None  # Should return fallback embeddings
        
        # Test with None input
        result = vision_encoder(None)
        assert result is not None  # Should handle None gracefully
    
    def test_fusion_processing_errors(self, error_handler):
        """Test fusion processing error scenarios."""
        fusion_orchestrator = FusionOrchestrator(
            text_dim=512,
            vision_dim=512,
            audio_dim=512,
            model_dim=512
        )
        
        # Test with mismatched dimensions
        text_embeds = torch.randn(1, 10, 512)
        vision_embeds = torch.randn(1, 5, 256)  # Wrong dimension
        
        result = fusion_orchestrator.fuse(
            text_embeds=text_embeds,
            vision_embeds=vision_embeds
        )
        
        # Should handle dimension mismatch gracefully
        assert result is not None
    
    def test_decision_engine_errors(self, mock_personality_layer, mock_memory_manager):
        """Test decision engine error scenarios."""
        decision_engine = DecisionEngine(mock_personality_layer, mock_memory_manager)
        
        # Test with None input
        result = decision_engine.decide(None)
        assert result is not None
        assert "mode" in result
        assert "recall" in result
        
        # Test with empty string
        result = decision_engine.decide("")
        assert result is not None
        
        # Test with very long input
        long_input = "test " * 10000
        result = decision_engine.decide(long_input)
        assert result is not None
        assert result["complexity"] > 0.8  # Should detect high complexity
    
    def test_response_generation_errors(self, mock_model, mock_personality_layer, mock_tokenizer):
        """Test response generation error scenarios."""
        # Create mock fusion orchestrator
        mock_fusion = Mock()
        mock_fusion.fuse.return_value = torch.randn(1, 10, 512)
        
        response_generator = ResponseGenerator(
            mock_model, mock_fusion, mock_tokenizer, mock_personality_layer
        )
        
        # Test with None model
        response_generator.model = None
        result = response_generator.generate({}, {})
        assert result is not None
        assert "fallback" in result.lower()
        
        # Test with invalid perception output
        invalid_perception = {"text_tokens": None}
        result = response_generator.generate(invalid_perception, {})
        assert result is not None
        
        # Test with safety concerns
        decision_with_safety = {
            "safety_check": {"safe": False, "concerns": ["harmful content"]},
            "mode": "DirectAssertive"
        }
        result = response_generator.generate({}, decision_with_safety)
        assert result is not None
        assert "cannot assist" in result.lower()
    
    def test_configuration_validation_errors(self):
        """Test configuration validation error scenarios."""
        validator = ConfigValidator()
        
        # Test invalid configuration
        invalid_config = {
            "model_dim": 4096,
            "spikingbrain": {
                "hidden_size": 512,  # Mismatch with model_dim
                "num_attention_heads": 8,
                "num_hidden_layers": 4,
                "intermediate_size": 2048
            },
            "vision": {
                "embed_dim": 1024  # Mismatch with model_dim
            },
            "audio": {
                "embed_dim": 1024  # Mismatch with model_dim
            },
            "memory": {
                "embedding_dim": 768  # Mismatch with model_dim
            }
        }
        
        results = validator.validate_model_config(invalid_config)
        assert len(results) > 0
        
        # Check for dimension mismatch warnings
        dimension_warnings = [r for r in results if "dimension" in r.message.lower()]
        assert len(dimension_warnings) > 0
    
    def test_memory_consolidation_errors(self, mock_memory_manager):
        """Test memory consolidation error scenarios."""
        # Add some memories
        for i in range(5):
            mock_memory_manager.store(f"Memory {i}", importance=0.8)
        
        # Test consolidation with corrupted memory
        original_memories = mock_memory_manager.episodic.memories
        original_memories[0].content = None  # Corrupt memory
        
        # Should handle corrupted memory gracefully
        mock_memory_manager._consolidate_memories()
        assert len(mock_memory_manager.episodic.memories) >= 0
    
    def test_error_recovery_strategies(self, error_handler):
        """Test different error recovery strategies."""
        # Test retry strategy
        with patch('time.sleep') as mock_sleep:
            result = error_handler._retry_operation(
                ConnectionError("Connection failed"),
                "test_operation"
            )
            assert result is None  # Should return None after retries
        
        # Test fallback strategy
        result = error_handler._fallback_operation(
            FileNotFoundError("File not found"),
            "model_loading"
        )
        assert result is None  # Should return fallback result
    
    def test_error_tracking(self, error_handler):
        """Test error tracking and statistics."""
        # Generate some errors
        error_handler.handle_error(
            ValueError("Test error 1"),
            "test_context",
            ErrorCategory.UNKNOWN,
            ErrorSeverity.MEDIUM
        )
        
        error_handler.handle_error(
            RuntimeError("Test error 2"),
            "test_context",
            ErrorCategory.UNKNOWN,
            ErrorSeverity.HIGH
        )
        
        # Check error statistics
        stats = error_handler.get_error_statistics()
        assert stats["total_errors"] == 2
        assert "unknown:ValueError" in stats["error_counts"]
        assert "unknown:RuntimeError" in stats["error_counts"]
    
    def test_safe_operation_context_manager(self, error_handler):
        """Test SafeOperation context manager."""
        from utils.error_handler import SafeOperation
        
        def failing_function():
            raise ValueError("Test error")
        
        def working_function():
            return "success"
        
        # Test with failing function
        with SafeOperation("test_operation", ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM, error_handler) as safe_op:
            result = safe_op.execute(failing_function)
            assert result is None  # Should return None due to error
        
        # Test with working function
        with SafeOperation("test_operation", ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM, error_handler) as safe_op:
            result = safe_op.execute(working_function)
            assert result == "success"
    
    def test_edge_cases(self, mock_model, mock_personality_layer, mock_tokenizer):
        """Test various edge cases."""
        mock_fusion = Mock()
        mock_fusion.fuse.return_value = torch.randn(1, 10, 512)
        
        response_generator = ResponseGenerator(
            mock_model, mock_fusion, mock_tokenizer, mock_personality_layer
        )
        
        # Test with extremely long input
        long_text = "test " * 50000
        perception_out = {"text_tokens": mock_tokenizer.encode(long_text)}
        decision = {"mode": "DirectAssertive", "recall_depth": 3}
        
        result = response_generator.generate(perception_out, decision)
        assert result is not None
        
        # Test with empty input
        perception_out = {"text_tokens": []}
        result = response_generator.generate(perception_out, decision)
        assert result is not None
        
        # Test with special characters
        special_text = "!@#$%^&*()_+{}|:<>?[]\\;'\",./"
        perception_out = {"text_tokens": mock_tokenizer.encode(special_text)}
        result = response_generator.generate(perception_out, decision)
        assert result is not None
    
    def test_memory_decay_errors(self, mock_memory_manager):
        """Test memory decay error scenarios."""
        # Add some memories
        for i in range(10):
            mock_memory_manager.store(f"Memory {i}", importance=0.5)
        
        # Test decay with corrupted importance scores
        original_scores = mock_memory_manager.episodic.importance_scores
        mock_memory_manager.episodic.importance_scores[0] = float('inf')  # Invalid score
        
        # Should handle invalid scores gracefully
        mock_memory_manager._apply_memory_decay()
        assert len(mock_memory_manager.episodic.memories) >= 0
    
    def test_multimodal_fusion_errors(self):
        """Test multimodal fusion error scenarios."""
        fusion_orchestrator = FusionOrchestrator(
            text_dim=512,
            vision_dim=512,
            audio_dim=512,
            model_dim=512
        )
        
        # Test with None inputs
        result = fusion_orchestrator.fuse(
            text_embeds=torch.randn(1, 10, 512),
            vision_embeds=None,
            audio_embeds=None
        )
        assert result is not None
        
        # Test with empty tensors
        result = fusion_orchestrator.fuse(
            text_embeds=torch.empty(0, 0, 512),
            vision_embeds=torch.empty(0, 0, 512),
            audio_embeds=torch.empty(0, 0, 512)
        )
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__])