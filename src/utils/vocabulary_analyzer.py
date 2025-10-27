"""
Vocabulary Analyzer - Real-time token vocabulary analysis
Extracts and validates vocabulary details from tokenizer files
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class VocabularyAnalyzer:
    """Analyzes vocabulary from various tokenizer formats"""
    
    def __init__(self, model_dir: str = "Model"):
        self.model_dir = Path(model_dir)
        self.tokenizer_files = [
            "tokenizer.json",
            "tokenizer_vocab.json", 
            "vocab.json",
            "vocabulary.json"
        ]
    
    def analyze_vocabulary(self) -> Dict[str, Any]:
        """Comprehensive vocabulary analysis"""
        vocabulary_data = {
            'size': 0,
            'tokens': [],
            'token_types': {},
            'special_tokens': {},
            'encoding_stats': {},
            'file_source': None,
            'is_valid': False,
            'errors': []
        }
        
        # Try to load from different tokenizer files
        for tokenizer_file in self.tokenizer_files:
            file_path = self.model_dir / tokenizer_file
            if file_path.exists():
                try:
                    vocab_data = self._load_vocabulary_file(file_path)
                    if vocab_data:
                        vocabulary_data.update(vocab_data)
                        vocabulary_data['file_source'] = str(file_path)
                        vocabulary_data['is_valid'] = True
                        break
                except Exception as e:
                    vocabulary_data['errors'].append(f"Failed to load {tokenizer_file}: {e}")
                    logger.warning(f"Failed to load {tokenizer_file}: {e}")
        
        # If no vocabulary found, try to extract from model
        if not vocabulary_data['is_valid']:
            vocab_from_model = self._extract_vocabulary_from_model()
            if vocab_from_model:
                vocabulary_data.update(vocab_from_model)
                vocabulary_data['file_source'] = 'extracted_from_model'
                vocabulary_data['is_valid'] = True
        
        # Analyze vocabulary characteristics
        if vocabulary_data['is_valid']:
            vocabulary_data.update(self._analyze_vocabulary_characteristics(vocabulary_data))
        
        return vocabulary_data
    
    def _load_vocabulary_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load vocabulary from specific file format"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if file_path.name == "tokenizer.json":
            return self._parse_tokenizer_json(data)
        elif file_path.name == "tokenizer_vocab.json":
            return self._parse_vocab_json(data)
        else:
            return self._parse_generic_json(data)
    
    def _parse_tokenizer_json(self, data: Dict) -> Dict[str, Any]:
        """Parse HuggingFace tokenizer.json format"""
        vocabulary = {
            'size': 0,
            'tokens': [],
            'token_types': {},
            'special_tokens': {},
            'encoding_stats': {}
        }
        
        # Extract vocabulary
        if 'model' in data and data['model'] and 'vocab' in data['model']:
            vocab = data['model']['vocab']
            vocabulary['size'] = len(vocab)
            vocabulary['tokens'] = list(vocab.keys())
            
            # Analyze token types
            vocabulary['token_types'] = self._categorize_tokens(vocabulary['tokens'])
        
        # Extract special tokens
        if 'added_tokens' in data:
            special_tokens = {}
            for token_info in data['added_tokens']:
                token = token_info.get('content', '')
                token_type = token_info.get('special', False)
                if token_type:
                    special_tokens[token] = token_info
            vocabulary['special_tokens'] = special_tokens
        
        # Extract encoding stats
        if 'normalizer' in data:
            vocabulary['encoding_stats']['normalizer'] = data['normalizer'].get('type', 'unknown')
        
        if 'pre_tokenizer' in data:
            vocabulary['encoding_stats']['pre_tokenizer'] = data['pre_tokenizer'].get('type', 'unknown')
        
        return vocabulary
    
    def _parse_vocab_json(self, data: Dict) -> Dict[str, Any]:
        """Parse simple vocabulary JSON format"""
        vocabulary = {
            'size': 0,
            'tokens': [],
            'token_types': {},
            'special_tokens': {},
            'encoding_stats': {}
        }
        
        if isinstance(data, dict):
            vocabulary['size'] = len(data)
            vocabulary['tokens'] = list(data.keys())
            vocabulary['token_types'] = self._categorize_tokens(vocabulary['tokens'])
        elif isinstance(data, list):
            vocabulary['size'] = len(data)
            vocabulary['tokens'] = data
            vocabulary['token_types'] = self._categorize_tokens(vocabulary['tokens'])
        
        return vocabulary
    
    def _parse_generic_json(self, data: Any) -> Dict[str, Any]:
        """Parse generic JSON vocabulary format"""
        vocabulary = {
            'size': 0,
            'tokens': [],
            'token_types': {},
            'special_tokens': {},
            'encoding_stats': {}
        }
        
        if isinstance(data, dict):
            if 'vocab' in data:
                vocab = data['vocab']
                vocabulary['size'] = len(vocab)
                vocabulary['tokens'] = list(vocab.keys()) if isinstance(vocab, dict) else vocab
            elif 'tokens' in data:
                vocabulary['size'] = len(data['tokens'])
                vocabulary['tokens'] = data['tokens']
            else:
                vocabulary['size'] = len(data)
                vocabulary['tokens'] = list(data.keys()) if isinstance(data, dict) else data
        elif isinstance(data, list):
            vocabulary['size'] = len(data)
            vocabulary['tokens'] = data
        
        vocabulary['token_types'] = self._categorize_tokens(vocabulary['tokens'])
        return vocabulary
    
    def _extract_vocabulary_from_model(self) -> Optional[Dict[str, Any]]:
        """Extract vocabulary from model weights"""
        try:
            import torch
            
            model_path = self.model_dir / "weights" / "Tantra_v1.0.pt"
            if not model_path.exists():
                return None
            
            model_data = torch.load(model_path, map_location='cpu')
            
            # Look for embedding layer
            if 'embed.weight' in model_data:
                embed_weight = model_data['embed.weight']
                vocab_size = embed_weight.shape[0]
                
                # Generate synthetic tokens
                tokens = [f"token_{i}" for i in range(vocab_size)]
                
                return {
                    'size': vocab_size,
                    'tokens': tokens,
                    'token_types': self._categorize_tokens(tokens),
                    'special_tokens': {},
                    'encoding_stats': {'source': 'extracted_from_model'}
                }
            
        except Exception as e:
            logger.warning(f"Failed to extract vocabulary from model: {e}")
        
        return None
    
    def _categorize_tokens(self, tokens: List[str]) -> Dict[str, int]:
        """Categorize tokens by type"""
        categories = {
            'special': 0,
            'punctuation': 0,
            'numbers': 0,
            'letters': 0,
            'mixed': 0,
            'unicode': 0,
            'whitespace': 0,
            'other': 0
        }
        
        for token in tokens:
            if self._is_special_token(token):
                categories['special'] += 1
            elif self._is_punctuation(token):
                categories['punctuation'] += 1
            elif self._is_number(token):
                categories['numbers'] += 1
            elif self._is_letters_only(token):
                categories['letters'] += 1
            elif self._is_mixed(token):
                categories['mixed'] += 1
            elif self._is_unicode(token):
                categories['unicode'] += 1
            elif self._is_whitespace(token):
                categories['whitespace'] += 1
            else:
                categories['other'] += 1
        
        return categories
    
    def _is_special_token(self, token: str) -> bool:
        """Check if token is special (starts with special characters)"""
        special_prefixes = ['<', '[', '##', '▁', 'Ġ']
        return any(token.startswith(prefix) for prefix in special_prefixes)
    
    def _is_punctuation(self, token: str) -> bool:
        """Check if token is punctuation"""
        import string
        return all(c in string.punctuation for c in token)
    
    def _is_number(self, token: str) -> bool:
        """Check if token is a number"""
        try:
            float(token)
            return True
        except ValueError:
            return False
    
    def _is_letters_only(self, token: str) -> bool:
        """Check if token contains only letters"""
        return token.isalpha()
    
    def _is_mixed(self, token: str) -> bool:
        """Check if token is mixed alphanumeric"""
        return any(c.isalpha() for c in token) and any(c.isdigit() for c in token)
    
    def _is_unicode(self, token: str) -> bool:
        """Check if token contains unicode characters"""
        return any(ord(c) > 127 for c in token)
    
    def _is_whitespace(self, token: str) -> bool:
        """Check if token is whitespace"""
        return token.isspace()
    
    def _analyze_vocabulary_characteristics(self, vocab_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze vocabulary characteristics"""
        tokens = vocab_data['tokens']
        
        # Length statistics
        token_lengths = [len(token) for token in tokens]
        
        # Character frequency
        char_freq = {}
        for token in tokens:
            for char in token:
                char_freq[char] = char_freq.get(char, 0) + 1
        
        # Most common characters
        most_common_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Language detection (simple heuristic)
        language_hints = self._detect_language_hints(tokens)
        
        return {
            'length_stats': {
                'min_length': min(token_lengths) if token_lengths else 0,
                'max_length': max(token_lengths) if token_lengths else 0,
                'avg_length': sum(token_lengths) / len(token_lengths) if token_lengths else 0,
                'median_length': sorted(token_lengths)[len(token_lengths)//2] if token_lengths else 0
            },
            'character_frequency': dict(most_common_chars),
            'language_hints': language_hints,
            'coverage_analysis': self._analyze_coverage(tokens)
        }
    
    def _detect_language_hints(self, tokens: List[str]) -> Dict[str, Any]:
        """Detect language hints from tokens"""
        hints = {
            'english_indicators': 0,
            'unicode_indicators': 0,
            'special_formatting': 0
        }
        
        for token in tokens:
            # English indicators
            if any(c.isalpha() and ord(c) < 128 for c in token):
                hints['english_indicators'] += 1
            
            # Unicode indicators
            if any(ord(c) > 127 for c in token):
                hints['unicode_indicators'] += 1
            
            # Special formatting
            if any(c in ['<', '>', '[', ']', '##'] for c in token):
                hints['special_formatting'] += 1
        
        return hints
    
    def _analyze_coverage(self, tokens: List[str]) -> Dict[str, Any]:
        """Analyze vocabulary coverage"""
        total_chars = sum(len(token) for token in tokens)
        unique_chars = len(set(''.join(tokens)))
        
        return {
            'total_characters': total_chars,
            'unique_characters': unique_chars,
            'character_diversity': unique_chars / total_chars if total_chars > 0 else 0,
            'token_diversity': len(set(tokens)) / len(tokens) if tokens else 0
        }
    
    def generate_vocabulary_report(self, vocab_data: Dict[str, Any]) -> str:
        """Generate vocabulary analysis report"""
        report = f"""
# Vocabulary Analysis Report

## Basic Information
- **Vocabulary Size**: {vocab_data['size']:,}
- **Source File**: {vocab_data['file_source'] or 'Not found'}
- **Status**: {'✅ Valid' if vocab_data['is_valid'] else '❌ Invalid'}

## Token Types Distribution
"""
        
        if vocab_data['token_types']:
            for token_type, count in vocab_data['token_types'].items():
                percentage = (count / vocab_data['size']) * 100 if vocab_data['size'] > 0 else 0
                report += f"- **{token_type.title()}**: {count:,} ({percentage:.1f}%)\n"
        
        if vocab_data.get('length_stats'):
            stats = vocab_data['length_stats']
            report += f"""
## Length Statistics
- **Minimum Length**: {stats['min_length']}
- **Maximum Length**: {stats['max_length']}
- **Average Length**: {stats['avg_length']:.2f}
- **Median Length**: {stats['median_length']}
"""
        
        if vocab_data.get('character_frequency'):
            report += "\n## Most Common Characters\n"
            for char, freq in list(vocab_data['character_frequency'].items())[:10]:
                report += f"- **'{char}'**: {freq:,} occurrences\n"
        
        if vocab_data.get('language_hints'):
            hints = vocab_data['language_hints']
            report += f"""
## Language Analysis
- **English Indicators**: {hints['english_indicators']:,}
- **Unicode Indicators**: {hints['unicode_indicators']:,}
- **Special Formatting**: {hints['special_formatting']:,}
"""
        
        if vocab_data.get('coverage_analysis'):
            coverage = vocab_data['coverage_analysis']
            report += f"""
## Coverage Analysis
- **Total Characters**: {coverage['total_characters']:,}
- **Unique Characters**: {coverage['unique_characters']:,}
- **Character Diversity**: {coverage['character_diversity']:.3f}
- **Token Diversity**: {coverage['token_diversity']:.3f}
"""
        
        if vocab_data['errors']:
            report += "\n## Errors\n"
            for error in vocab_data['errors']:
                report += f"- {error}\n"
        
        # Sample tokens
        if vocab_data['tokens']:
            sample_size = min(20, len(vocab_data['tokens']))
            report += f"\n## Sample Tokens (First {sample_size})\n"
            for i, token in enumerate(vocab_data['tokens'][:sample_size]):
                report += f"{i+1:2d}. `{token}`\n"
        
        return report

def main():
    """Main function for testing"""
    analyzer = VocabularyAnalyzer()
    
    print("Analyzing vocabulary...")
    vocab_data = analyzer.analyze_vocabulary()
    
    # Generate report
    report = analyzer.generate_vocabulary_report(vocab_data)
    print(report)
    
    # Save report
    with open("vocabulary_analysis_report.md", "w") as f:
        f.write(report)
    
    print("Vocabulary analysis complete! Report saved to vocabulary_analysis_report.md")

if __name__ == "__main__":
    main()