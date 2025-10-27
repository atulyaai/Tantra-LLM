#!/usr/bin/env python3
"""
Basic System Test
Test core functionality without heavy dependencies
"""

import sys
import os
import logging
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_file_structure():
    """Test that all required files exist"""
    print("\n" + "="*50)
    print("📁 TESTING FILE STRUCTURE")
    print("="*50)
    
    required_files = [
        "src/versioning/__init__.py",
        "src/versioning/version_config.py", 
        "src/versioning/model_versioning.py",
        "src/web/__init__.py",
        "src/web/web_scraper.py",
        "src/web/web_search.py",
        "src/rag/__init__.py",
        "src/rag/rope_rag.py",
        "src/rag/knowledge_base.py",
        "src/rag/retriever.py",
        "src/rag/generator.py",
        "src/rag/evaluator.py",
        "src/training/lora_trainer.py",
        "train_advanced_system.py",
        "test_advanced_system.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    else:
        print("✅ All required files exist")
        return True


def test_versioning_config():
    """Test versioning configuration"""
    print("\n" + "="*50)
    print("⚙️ TESTING VERSIONING CONFIG")
    print("="*50)
    
    try:
        from src.versioning.version_config import VersionConfig, VersionInfo
        
        # Test VersionConfig
        config = VersionConfig()
        print(f"✅ VersionConfig created: {config.get_version_string()}")
        
        # Test version increment
        new_version = config.increment_version("patch")
        print(f"✅ Version incremented: {new_version}")
        
        # Test VersionInfo
        version_info = VersionInfo(
            version="1.0.0",
            timestamp=time.time(),
            version_type="model",
            description="Test version"
        )
        print(f"✅ VersionInfo created: {version_info.version}")
        
        return True
        
    except Exception as e:
        print(f"❌ Versioning config test failed: {e}")
        return False


def test_web_scraping_config():
    """Test web scraping configuration"""
    print("\n" + "="*50)
    print("🌐 TESTING WEB SCRAPING CONFIG")
    print("="*50)
    
    try:
        from src.web.web_scraper import ScrapingConfig
        
        # Test ScrapingConfig
        config = ScrapingConfig(
            extract_text=True,
            extract_metadata=True,
            min_content_length=100
        )
        print(f"✅ ScrapingConfig created")
        print(f"   - Extract text: {config.extract_text}")
        print(f"   - Extract metadata: {config.extract_metadata}")
        print(f"   - Min content length: {config.min_content_length}")
        
        return True
        
    except Exception as e:
        print(f"❌ Web scraping config test failed: {e}")
        return False


def test_web_search_config():
    """Test web search configuration"""
    print("\n" + "="*50)
    print("🔍 TESTING WEB SEARCH CONFIG")
    print("="*50)
    
    try:
        from src.web.web_search import SearchConfig
        
        # Test SearchConfig
        config = SearchConfig(
            max_results=10,
            language="en"
        )
        print(f"✅ SearchConfig created")
        print(f"   - Max results: {config.max_results}")
        print(f"   - Language: {config.language}")
        
        return True
        
    except Exception as e:
        print(f"❌ Web search config test failed: {e}")
        return False


def test_rag_config():
    """Test RAG configuration"""
    print("\n" + "="*50)
    print("🧠 TESTING RAG CONFIG")
    print("="*50)
    
    try:
        from src.rag.rope_rag import RAGConfig
        
        # Test RAGConfig
        config = RAGConfig(
            model_name="test_rag",
            retrieval_top_k=5,
            embedding_dim=768
        )
        print(f"✅ RAGConfig created")
        print(f"   - Model name: {config.model_name}")
        print(f"   - Retrieval top k: {config.retrieval_top_k}")
        print(f"   - Embedding dim: {config.embedding_dim}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG config test failed: {e}")
        return False


def test_lora_config():
    """Test LoRA configuration"""
    print("\n" + "="*50)
    print("🎯 TESTING LORA CONFIG")
    print("="*50)
    
    try:
        from src.training.lora_trainer import LoRAConfig
        
        # Test LoRAConfig
        config = LoRAConfig(
            rank=16,
            alpha=32.0,
            learning_rate=1e-4
        )
        print(f"✅ LoRAConfig created")
        print(f"   - Rank: {config.rank}")
        print(f"   - Alpha: {config.alpha}")
        print(f"   - Learning rate: {config.learning_rate}")
        
        return True
        
    except Exception as e:
        print(f"❌ LoRA config test failed: {e}")
        return False


def test_knowledge_base():
    """Test knowledge base functionality"""
    print("\n" + "="*50)
    print("📚 TESTING KNOWLEDGE BASE")
    print("="*50)
    
    try:
        from src.rag.knowledge_base import Document, KnowledgeBase
        from src.rag.rope_rag import RAGConfig
        
        # Test Document
        doc = Document(
            id="test_doc",
            content="This is a test document for the knowledge base.",
            metadata={"source": "test", "type": "sample"}
        )
        print(f"✅ Document created: {doc.id}")
        
        # Test KnowledgeBase (without heavy dependencies)
        rag_config = RAGConfig(embedding_dim=128)  # Smaller for testing
        kb = KnowledgeBase(rag_config)
        print(f"✅ KnowledgeBase created")
        
        # Test adding document
        kb.add_documents([doc])
        print(f"✅ Document added to knowledge base")
        
        # Test stats
        stats = kb.get_stats()
        print(f"✅ Knowledge base stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Knowledge base test failed: {e}")
        return False


def test_requirements():
    """Test requirements file"""
    print("\n" + "="*50)
    print("📦 TESTING REQUIREMENTS")
    print("="*50)
    
    try:
        with open("requirements.txt", "r") as f:
            requirements = f.read()
        
        # Check for key dependencies
        key_deps = [
            "torch",
            "transformers", 
            "beautifulsoup4",
            "sentence-transformers",
            "PyGithub",
            "tensorboard"
        ]
        
        missing_deps = []
        for dep in key_deps:
            if dep not in requirements:
                missing_deps.append(dep)
        
        if missing_deps:
            print(f"❌ Missing dependencies: {missing_deps}")
            return False
        else:
            print("✅ All key dependencies found in requirements.txt")
            return True
            
    except Exception as e:
        print(f"❌ Requirements test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🧪 BASIC SYSTEM TESTING")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run all tests
    tests = {
        'File Structure': test_file_structure,
        'Versioning Config': test_versioning_config,
        'Web Scraping Config': test_web_scraping_config,
        'Web Search Config': test_web_search_config,
        'RAG Config': test_rag_config,
        'LoRA Config': test_lora_config,
        'Knowledge Base': test_knowledge_base,
        'Requirements': test_requirements
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
        print("\n🎉 SYSTEM STRUCTURE IS READY!")
        print("   Install dependencies: pip install -r requirements.txt")
        print("   Run full test: python3 test_advanced_system.py")
        print("   Start training: python3 train_advanced_system.py")
    else:
        print("\n⚠️  SYSTEM NEEDS ATTENTION")
        print("   Some components failed - check logs above")
    
    return success_rate >= 0.8


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)