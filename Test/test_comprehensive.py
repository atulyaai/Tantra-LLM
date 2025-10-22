#!/usr/bin/env python3
"""
Comprehensive test suite for Tantra LLM
Tests all major components and catches common issues
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add workspace to Python path
sys.path.insert(0, '/workspace')

def test_imports():
    """Test that all modules can be imported without errors"""
    print("Testing imports...")
    
    try:
        from Training.agent import Agent, ToolSpec
        print("✓ Agent imports successful")
    except Exception as e:
        print(f"✗ Agent import failed: {e}")
        return False
    
    try:
        from Training.memory import Memory
        print("✓ Memory imports successful")
    except Exception as e:
        print(f"✗ Memory import failed: {e}")
        return False
    
    try:
        from Training.tools_basic import TOOL_SPECS
        print("✓ Tools imports successful")
    except Exception as e:
        print(f"✗ Tools import failed: {e}")
        return False
    
    try:
        from Training.model_mamba import MambaDecoder, MambaBlock
        print("✓ Mamba model imports successful")
    except Exception as e:
        print(f"✗ Mamba model import failed: {e}")
        return False
    
    try:
        from Training.mamba_runtime import MambaRuntime
        print("✓ Mamba runtime imports successful")
    except Exception as e:
        print(f"✗ Mamba runtime import failed: {e}")
        return False
    
    try:
        from Training.model_runtime import TextRuntime
        print("✓ Text runtime imports successful")
    except Exception as e:
        print(f"✗ Text runtime import failed: {e}")
        return False
    
    return True

def test_config_files():
    """Test that all config files are valid YAML"""
    print("\nTesting config files...")
    
    config_files = [
        "Config/agent.yaml",
        "Config/data_sources.yaml", 
        "Config/pretrain.yaml",
        "Config/realtime.yaml",
        "Config/serve.yaml"
    ]
    
    try:
        import yaml
    except ImportError:
        print("✗ PyYAML not available")
        return False
    
    for config_file in config_files:
        if not Path(config_file).exists():
            print(f"✗ Config file missing: {config_file}")
            return False
        
        try:
            with open(config_file, 'r') as f:
                yaml.safe_load(f)
            print(f"✓ {config_file} is valid YAML")
        except Exception as e:
            print(f"✗ {config_file} is invalid: {e}")
            return False
    
    return True

def test_directories():
    """Test that required directories exist"""
    print("\nTesting directories...")
    
    required_dirs = [
        "Dataset",
        "Model", 
        "logs",
        "Config",
        "Training",
        "Test"
    ]
    
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            print(f"✗ Directory missing: {dir_name}")
            return False
        print(f"✓ Directory exists: {dir_name}")
    
    return True

def test_model_creation():
    """Test that models can be created without errors"""
    print("\nTesting model creation...")
    
    try:
        from Training.model_mamba import MambaDecoder
        
        # Test creating a small model
        model = MambaDecoder(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            d_state=16,
            d_conv=4,
            dropout=0.1
        )
        print("✓ MambaDecoder creation successful")
        
        # Test forward pass
        import torch
        x = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            output = model(x)
        print("✓ MambaDecoder forward pass successful")
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False
    
    return True

def test_agent_creation():
    """Test that agent can be created"""
    print("\nTesting agent creation...")
    
    try:
        from Training.agent import Agent, ToolSpec
        from Training.memory import Memory
        from Training.tools_basic import TOOL_SPECS
        
        # Create mock LLM function
        def mock_llm(prompt, gen):
            return "This is a test response."
        
        # Create agent
        config = {
            "persona": {"name": "Test", "system_instructions": "You are a test assistant."},
            "tools": {"enabled": ["calc"]},
            "memory": {"short": {"max_turns": 5}}
        }
        
        memory = Memory(config)
        tools = {name: ToolSpec(name, spec["schema"], spec["func"], spec.get("destructive", False)) 
                for name, spec in TOOL_SPECS.items()}
        
        agent = Agent(config, tools, memory, mock_llm)
        print("✓ Agent creation successful")
        
        # Test agent run
        traces, response = agent.run("What is 2+2?")
        print("✓ Agent run successful")
        
    except Exception as e:
        print(f"✗ Agent creation failed: {e}")
        return False
    
    return True

def test_file_operations():
    """Test file operations work correctly"""
    print("\nTesting file operations...")
    
    try:
        from Training.tools_basic import tool_files
        
        # Test reading a file
        result = tool_files({"action": "read", "path": "README.md"})
        if "error" in result:
            print(f"✗ File read failed: {result['error']}")
            return False
        print("✓ File read successful")
        
        # Test writing a file
        test_content = "This is a test file."
        result = tool_files({"action": "write", "path": "test_file.txt", "content": test_content})
        if "error" in result:
            print(f"✗ File write failed: {result['error']}")
            return False
        print("✓ File write successful")
        
        # Clean up
        if Path("test_file.txt").exists():
            Path("test_file.txt").unlink()
        
    except Exception as e:
        print(f"✗ File operations failed: {e}")
        return False
    
    return True

def test_calculator():
    """Test calculator tool"""
    print("\nTesting calculator...")
    
    try:
        from Training.tools_basic import tool_calc
        
        # Test simple calculation
        result = tool_calc({"expr": "2 + 2"})
        if "error" in result:
            print(f"✗ Calculator failed: {result['error']}")
            return False
        
        if result.get("result") != 4:
            print(f"✗ Calculator gave wrong result: {result}")
            return False
        
        print("✓ Calculator works correctly")
        
    except Exception as e:
        print(f"✗ Calculator test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🧪 Running comprehensive test suite for Tantra LLM\n")
    
    tests = [
        test_imports,
        test_config_files,
        test_directories,
        test_model_creation,
        test_agent_creation,
        test_file_operations,
        test_calculator
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"✗ {test.__name__} failed")
        except Exception as e:
            print(f"✗ {test.__name__} crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The codebase is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)