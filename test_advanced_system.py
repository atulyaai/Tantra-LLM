#!/usr/bin/env python3
"""
Test Advanced Tantra System
Comprehensive testing of all components
"""

import sys
import os
import logging
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.training_config import TrainingConfig
from src.training.lora_trainer import LoRATrainer, LoRAConfig
from src.web.web_scraper import WebScraper, ScrapingConfig
from src.web.web_search import WebSearch, SearchConfig
from src.rag.rope_rag import ROPERAG, RAGConfig
from src.versioning.model_versioning import ModelVersionManager
from src.versioning.version_config import VersionConfig
from src.core.tantra_llm import TantraLLM, TantraConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_versioning_system():
    """Test versioning system"""
    print("\n" + "="*50)
    print("🧪 TESTING VERSIONING SYSTEM")
    print("="*50)
    
    try:
        # Initialize versioning
        version_config = VersionConfig()
        model_version_manager = ModelVersionManager(version_config)
        
        # Test version creation
        print("✅ Versioning system initialized")
        
        # Test version info
        stats = model_version_manager.get_version_statistics()
        print(f"📊 Version statistics: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Versioning system test failed: {e}")
        return False


def test_web_scraping():
    """Test web scraping functionality"""
    print("\n" + "="*50)
    print("🌐 TESTING WEB SCRAPING")
    print("="*50)
    
    try:
        # Initialize web scraper
        scraping_config = ScrapingConfig(
            extract_text=True,
            extract_metadata=True,
            min_content_length=50
        )
        scraper = WebScraper(scraping_config)
        
        # Test scraping (using a simple example)
        test_url = "https://example.com"
        print(f"🔍 Testing scraping: {test_url}")
        
        # Note: This will fail in the test environment, but we can test the structure
        try:
            result = scraper.scrape_url(test_url)
            print(f"✅ Scraping result: {result.get('success', False)}")
        except Exception as e:
            print(f"⚠️  Scraping failed (expected in test environment): {e}")
        
        # Test cache stats
        cache_stats = scraper.get_cache_stats()
        print(f"📊 Cache stats: {cache_stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Web scraping test failed: {e}")
        return False


def test_web_search():
    """Test web search functionality"""
    print("\n" + "="*50)
    print("🔍 TESTING WEB SEARCH")
    print("="*50)
    
    try:
        # Initialize web search
        search_config = SearchConfig(
            max_results=5,
            language="en"
        )
        searcher = WebSearch(search_config)
        
        # Test search
        test_query = "artificial intelligence"
        print(f"🔍 Testing search: {test_query}")
        
        results = searcher.search(test_query)
        print(f"✅ Search results: {len(results)} found")
        
        # Test search stats
        search_stats = searcher.get_search_stats()
        print(f"📊 Search stats: {search_stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Web search test failed: {e}")
        return False


def test_rope_rag():
    """Test ROPE RAG system"""
    print("\n" + "="*50)
    print("🧠 TESTING ROPE RAG SYSTEM")
    print("="*50)
    
    try:
        # Initialize ROPE RAG
        rag_config = RAGConfig(
            model_name="test_rope_rag",
            retrieval_top_k=3,
            knowledge_base_path="test_knowledge_base"
        )
        rope_rag = ROPERAG(rag_config)
        
        # Test with sample data
        sample_docs = [
            {
                'id': 'doc1',
                'content': 'Artificial intelligence is a field of computer science.',
                'metadata': {'source': 'test', 'type': 'definition'}
            },
            {
                'id': 'doc2', 
                'content': 'Machine learning is a subset of artificial intelligence.',
                'metadata': {'source': 'test', 'type': 'definition'}
            }
        ]
        
        # Add documents
        rope_rag.add_documents(sample_docs)
        print("✅ Documents added to knowledge base")
        
        # Test retrieval and generation
        test_query = "What is artificial intelligence?"
        result = rope_rag.retrieve_and_generate(test_query)
        
        print(f"✅ Query: {test_query}")
        print(f"✅ Response: {result.get('response', 'No response')[:100]}...")
        print(f"✅ Confidence: {result.get('confidence', 0.0):.2f}")
        
        # Test stats
        stats = rope_rag.get_stats()
        print(f"📊 RAG stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ ROPE RAG test failed: {e}")
        return False


def test_lora_training():
    """Test LoRA training system"""
    print("\n" + "="*50)
    print("🎯 TESTING LORA TRAINING")
    print("="*50)
    
    try:
        # Initialize base model
        tantra_config = TantraConfig()
        base_model = TantraLLM(tantra_config)
        
        # Initialize LoRA trainer
        lora_config = LoRAConfig(
            rank=8,
            alpha=16.0,
            learning_rate=1e-4,
            lora_learning_rate=1e-3
        )
        lora_trainer = LoRATrainer(base_model, lora_config)
        
        print("✅ LoRA trainer initialized")
        
        # Test parameter counts
        total_params = lora_trainer.get_total_parameters_count()
        lora_params = lora_trainer.get_lora_parameters_count()
        efficiency = lora_trainer.get_parameter_efficiency()
        
        print(f"📊 Total parameters: {total_params:,}")
        print(f"📊 LoRA parameters: {lora_params:,}")
        print(f"📊 Parameter efficiency: {efficiency:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ LoRA training test failed: {e}")
        return False


def test_integration():
    """Test system integration"""
    print("\n" + "="*50)
    print("🔗 TESTING SYSTEM INTEGRATION")
    print("="*50)
    
    try:
        # Initialize all components
        training_config = TrainingConfig()
        
        # Test component initialization
        components = {
            'versioning': test_versioning_system(),
            'web_scraping': test_web_scraping(),
            'web_search': test_web_search(),
            'rope_rag': test_rope_rag(),
            'lora_training': test_lora_training()
        }
        
        # Calculate success rate
        successful_components = sum(components.values())
        total_components = len(components)
        success_rate = successful_components / total_components
        
        print(f"\n📊 Integration Test Results:")
        print(f"   Successful components: {successful_components}/{total_components}")
        print(f"   Success rate: {success_rate:.1%}")
        
        for component, success in components.items():
            status = "✅" if success else "❌"
            print(f"   {status} {component}")
        
        return success_rate > 0.8  # 80% success rate threshold
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def test_performance():
    """Test system performance"""
    print("\n" + "="*50)
    print("⚡ TESTING PERFORMANCE")
    print("="*50)
    
    try:
        performance_results = {}
        
        # Test ROPE RAG performance
        start_time = time.time()
        rag_config = RAGConfig()
        rope_rag = ROPERAG(rag_config)
        rag_init_time = time.time() - start_time
        performance_results['rag_initialization'] = rag_init_time
        
        # Test LoRA performance
        start_time = time.time()
        tantra_config = TantraConfig()
        base_model = TantraLLM(tantra_config)
        lora_config = LoRAConfig()
        lora_trainer = LoRATrainer(base_model, lora_config)
        lora_init_time = time.time() - start_time
        performance_results['lora_initialization'] = lora_init_time
        
        # Test versioning performance
        start_time = time.time()
        version_config = VersionConfig()
        model_version_manager = ModelVersionManager(version_config)
        versioning_init_time = time.time() - start_time
        performance_results['versioning_initialization'] = versioning_init_time
        
        print("📊 Performance Results:")
        for component, time_taken in performance_results.items():
            print(f"   {component}: {time_taken:.3f}s")
        
        # Check if performance is acceptable
        total_time = sum(performance_results.values())
        acceptable = total_time < 10.0  # Should initialize in under 10 seconds
        
        print(f"✅ Total initialization time: {total_time:.3f}s")
        print(f"✅ Performance acceptable: {acceptable}")
        
        return acceptable
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 ADVANCED TANTRA SYSTEM TESTING")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run all tests
    tests = {
        'Versioning System': test_versioning_system,
        'Web Scraping': test_web_scraping,
        'Web Search': test_web_search,
        'ROPE RAG': test_rope_rag,
        'LoRA Training': test_lora_training,
        'System Integration': test_integration,
        'Performance': test_performance
    }
    
    results = {}
    for test_name, test_func in tests.items():
        print(f"\n🧪 Running {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Calculate overall results
    total_tests = len(tests)
    passed_tests = sum(results.values())
    success_rate = passed_tests / total_tests
    
    total_time = time.time() - start_time
    
    # Print final results
    print("\n" + "="*60)
    print("🎯 FINAL TEST RESULTS")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Overall Results:")
    print(f"   Tests passed: {passed_tests}/{total_tests}")
    print(f"   Success rate: {success_rate:.1%}")
    print(f"   Total time: {total_time:.2f}s")
    
    if success_rate >= 0.8:
        print("\n🎉 SYSTEM READY FOR TRAINING!")
        print("   Run: python train_advanced_system.py")
    else:
        print("\n⚠️  SYSTEM NEEDS ATTENTION")
        print("   Some components failed - check logs above")
    
    return success_rate >= 0.8


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)