"""
Configuration Validation System
Validates configuration consistency and provides health checks.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ValidationSeverity(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a configuration validation."""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    suggested_fix: Optional[str] = None


class ConfigValidator:
    """Validates configuration consistency and provides health checks."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_results: List[ValidationResult] = []
    
    def validate_model_config(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate model configuration."""
        results = []
        
        # Check required fields
        required_fields = ["model_dim", "spikingbrain", "vision", "audio", "memory"]
        for field in required_fields:
            if field not in config:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Missing required field: {field}",
                    field=field,
                    suggested_fix=f"Add {field} to configuration"
                ))
        
        # Validate SpikingBrain config
        if "spikingbrain" in config:
            spb_config = config["spikingbrain"]
            spb_results = self._validate_spikingbrain_config(spb_config)
            results.extend(spb_results)
        
        # Validate vision config
        if "vision" in config:
            vision_results = self._validate_vision_config(config["vision"])
            results.extend(vision_results)
        
        # Validate audio config
        if "audio" in config:
            audio_results = self._validate_audio_config(config["audio"])
            results.extend(audio_results)
        
        # Validate memory config
        if "memory" in config:
            memory_results = self._validate_memory_config(config["memory"])
            results.extend(memory_results)
        
        # Check dimension consistency
        dimension_results = self._validate_dimension_consistency(config)
        results.extend(dimension_results)
        
        self.validation_results = results
        return results
    
    def _validate_spikingbrain_config(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate SpikingBrain configuration."""
        results = []
        
        # Check required fields
        required_fields = ["hidden_size", "num_attention_heads", "num_hidden_layers", "intermediate_size"]
        for field in required_fields:
            if field not in config:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Missing SpikingBrain field: {field}",
                    field=f"spikingbrain.{field}",
                    suggested_fix=f"Add {field} to spikingbrain configuration"
                ))
        
        # Validate hidden_size
        if "hidden_size" in config:
            hidden_size = config["hidden_size"]
            if not isinstance(hidden_size, int) or hidden_size <= 0:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid hidden_size: {hidden_size}",
                    field="spikingbrain.hidden_size",
                    suggested_fix="hidden_size must be a positive integer"
                ))
            elif hidden_size % 64 != 0:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"hidden_size {hidden_size} is not divisible by 64",
                    field="spikingbrain.hidden_size",
                    suggested_fix="Consider using a multiple of 64 for better performance"
                ))
        
        # Validate attention heads
        if "num_attention_heads" in config and "hidden_size" in config:
            num_heads = config["num_attention_heads"]
            hidden_size = config["hidden_size"]
            if hidden_size % num_heads != 0:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"hidden_size {hidden_size} not divisible by num_attention_heads {num_heads}",
                    field="spikingbrain.num_attention_heads",
                    suggested_fix="Ensure hidden_size is divisible by num_attention_heads"
                ))
        
        # Validate intermediate size
        if "intermediate_size" in config and "hidden_size" in config:
            intermediate_size = config["intermediate_size"]
            hidden_size = config["hidden_size"]
            if intermediate_size < hidden_size:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"intermediate_size {intermediate_size} is smaller than hidden_size {hidden_size}",
                    field="spikingbrain.intermediate_size",
                    suggested_fix="Consider making intermediate_size larger than hidden_size"
                ))
        
        return results
    
    def _validate_vision_config(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate vision configuration."""
        results = []
        
        # Check embed_dim
        if "embed_dim" in config:
            embed_dim = config["embed_dim"]
            if not isinstance(embed_dim, int) or embed_dim <= 0:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid vision embed_dim: {embed_dim}",
                    field="vision.embed_dim",
                    suggested_fix="embed_dim must be a positive integer"
                ))
        
        return results
    
    def _validate_audio_config(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate audio configuration."""
        results = []
        
        # Check embed_dim
        if "embed_dim" in config:
            embed_dim = config["embed_dim"]
            if not isinstance(embed_dim, int) or embed_dim <= 0:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid audio embed_dim: {embed_dim}",
                    field="audio.embed_dim",
                    suggested_fix="embed_dim must be a positive integer"
                ))
        
        return results
    
    def _validate_memory_config(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate memory configuration."""
        results = []
        
        # Check embedding_dim
        if "embedding_dim" in config:
            embedding_dim = config["embedding_dim"]
            if not isinstance(embedding_dim, int) or embedding_dim <= 0:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid memory embedding_dim: {embedding_dim}",
                    field="memory.embedding_dim",
                    suggested_fix="embedding_dim must be a positive integer"
                ))
        
        # Check max_episodic
        if "max_episodic" in config:
            max_episodic = config["max_episodic"]
            if not isinstance(max_episodic, int) or max_episodic <= 0:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid max_episodic: {max_episodic}",
                    field="memory.max_episodic",
                    suggested_fix="max_episodic must be a positive integer"
                ))
            elif max_episodic > 100000:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"max_episodic {max_episodic} is very large",
                    field="memory.max_episodic",
                    suggested_fix="Consider reducing max_episodic for better performance"
                ))
        
        return results
    
    def _validate_dimension_consistency(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate dimension consistency across components."""
        results = []
        
        # Get dimensions
        model_dim = config.get("model_dim")
        vision_dim = config.get("vision", {}).get("embed_dim")
        audio_dim = config.get("audio", {}).get("embed_dim")
        memory_dim = config.get("memory", {}).get("embedding_dim")
        spb_hidden = config.get("spikingbrain", {}).get("hidden_size")
        
        # Check model dimension consistency
        if model_dim and spb_hidden and model_dim != spb_hidden:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"model_dim {model_dim} != spikingbrain.hidden_size {spb_hidden}",
                field="model_dim",
                suggested_fix="Ensure model_dim matches spikingbrain.hidden_size"
            ))
        
        # Check vision dimension
        if vision_dim and model_dim and vision_dim != model_dim:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"vision.embed_dim {vision_dim} != model_dim {model_dim}",
                field="vision.embed_dim",
                suggested_fix="Consider aligning vision.embed_dim with model_dim for consistency"
            ))
        
        # Check audio dimension
        if audio_dim and model_dim and audio_dim != model_dim:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"audio.embed_dim {audio_dim} != model_dim {model_dim}",
                field="audio.embed_dim",
                suggested_fix="Consider aligning audio.embed_dim with model_dim for consistency"
            ))
        
        # Check memory dimension
        if memory_dim and model_dim and memory_dim != model_dim:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"memory.embedding_dim {memory_dim} != model_dim {model_dim}",
                field="memory.embedding_dim",
                suggested_fix="Consider aligning memory.embedding_dim with model_dim for consistency"
            ))
        
        return results
    
    def validate_personality_config(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate personality configuration."""
        results = []
        
        # Check required modes
        required_modes = ["DirectAssertive", "MentorBuilder", "CriticalChallenger", "CreativeExplorer"]
        if "tones" in config:
            available_modes = list(config["tones"].keys())
            for mode in required_modes:
                if mode not in available_modes:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        message=f"Missing personality mode: {mode}",
                        field=f"personality.tones.{mode}",
                        suggested_fix=f"Add {mode} configuration to personality tones"
                    ))
        
        return results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        if not self.validation_results:
            return {"status": "no_validation", "message": "No validation performed"}
        
        total_results = len(self.validation_results)
        error_count = sum(1 for r in self.validation_results if r.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for r in self.validation_results if r.severity == ValidationSeverity.WARNING)
        critical_count = sum(1 for r in self.validation_results if r.severity == ValidationSeverity.CRITICAL)
        
        is_valid = error_count == 0 and critical_count == 0
        
        return {
            "status": "valid" if is_valid else "invalid",
            "total_issues": total_results,
            "critical": critical_count,
            "errors": error_count,
            "warnings": warning_count,
            "is_valid": is_valid
        }
    
    def get_critical_issues(self) -> List[ValidationResult]:
        """Get critical validation issues."""
        return [r for r in self.validation_results if r.severity == ValidationSeverity.CRITICAL]
    
    def get_error_issues(self) -> List[ValidationResult]:
        """Get error validation issues."""
        return [r for r in self.validation_results if r.severity == ValidationSeverity.ERROR]
    
    def get_warning_issues(self) -> List[ValidationResult]:
        """Get warning validation issues."""
        return [r for r in self.validation_results if r.severity == ValidationSeverity.WARNING]
    
    def print_validation_report(self) -> None:
        """Print a detailed validation report."""
        summary = self.get_validation_summary()
        
        print("=" * 60)
        print("CONFIGURATION VALIDATION REPORT")
        print("=" * 60)
        print(f"Status: {summary['status'].upper()}")
        print(f"Total Issues: {summary['total_issues']}")
        print(f"Critical: {summary['critical']}")
        print(f"Errors: {summary['errors']}")
        print(f"Warnings: {summary['warnings']}")
        print()
        
        # Print critical issues
        critical_issues = self.get_critical_issues()
        if critical_issues:
            print("CRITICAL ISSUES:")
            print("-" * 20)
            for issue in critical_issues:
                print(f"❌ {issue.field}: {issue.message}")
                if issue.suggested_fix:
                    print(f"   💡 Fix: {issue.suggested_fix}")
                print()
        
        # Print error issues
        error_issues = self.get_error_issues()
        if error_issues:
            print("ERROR ISSUES:")
            print("-" * 15)
            for issue in error_issues:
                print(f"🔴 {issue.field}: {issue.message}")
                if issue.suggested_fix:
                    print(f"   💡 Fix: {issue.suggested_fix}")
                print()
        
        # Print warning issues
        warning_issues = self.get_warning_issues()
        if warning_issues:
            print("WARNING ISSUES:")
            print("-" * 16)
            for issue in warning_issues:
                print(f"🟡 {issue.field}: {issue.message}")
                if issue.suggested_fix:
                    print(f"   💡 Fix: {issue.suggested_fix}")
                print()
        
        print("=" * 60)


def validate_all_configs(model_config: Dict[str, Any], personality_config: Dict[str, Any]) -> Tuple[bool, List[ValidationResult]]:
    """Validate all configurations and return overall status."""
    validator = ConfigValidator()
    
    # Validate model config
    model_results = validator.validate_model_config(model_config)
    
    # Validate personality config
    personality_results = validator.validate_personality_config(personality_config)
    
    all_results = model_results + personality_results
    
    # Check if any critical or error issues exist
    has_critical_errors = any(
        r.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR] 
        for r in all_results
    )
    
    return not has_critical_errors, all_results