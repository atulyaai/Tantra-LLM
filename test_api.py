#!/usr/bin/env python3
"""
Simple API test script
"""

import sys
from pathlib import Path

# Add workspace to Python path
sys.path.insert(0, '/workspace')

def test_api_imports():
    """Test that API can be imported and initialized"""
    print("Testing API imports...")
    
    try:
        from Training.serve_api import app, build_agent
        print("✓ API imports successful")
        
        # Test agent creation
        agent = build_agent()
        print("✓ Agent creation successful")
        
        # Test basic agent functionality
        traces, response = agent.run("Hello, how are you?")
        print(f"✓ Agent response: {response[:50]}...")
        
        return True
    except Exception as e:
        print(f"✗ API test failed: {e}")
        return False

def test_api_endpoints():
    """Test API endpoint definitions"""
    print("\nTesting API endpoints...")
    
    try:
        from Training.serve_api import app
        
        # Check if endpoints are defined
        routes = [route.path for route in app.routes]
        expected_routes = ["/healthz", "/infer", "/chat/stream", "/agent/stream", "/memory/flush"]
        
        for route in expected_routes:
            if route in routes:
                print(f"✓ Endpoint {route} exists")
            else:
                print(f"✗ Endpoint {route} missing")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Endpoint test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Tantra LLM API\n")
    
    success = True
    success &= test_api_imports()
    success &= test_api_endpoints()
    
    if success:
        print("\n🎉 API tests passed! The API is ready to use.")
        print("\nTo start the server, run:")
        print("  cd /workspace && python3 Training/serve_api.py")
        print("\nThen visit: http://localhost:8000/docs for API documentation")
    else:
        print("\n❌ Some API tests failed.")
        sys.exit(1)