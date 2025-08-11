#!/usr/bin/env python3
"""
Test the restructured import structure
"""

import os
import sys

# Ensure project root is on sys.path even when running as `python tests/test_structure.py`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def test_imports():
    """Test that the imports work correctly"""
    print("Testing import structure...")
    
    try:
        # Test importing the core module
        from rag_core import rag_pipeline
        print("✅ Successfully imported rag_pipeline from rag_core")
        
        # Test importing the app module
        from app import app
        print("✅ Successfully imported FastAPI app from app")
        
        print("\n🎉 All imports successful! The restructuring is complete.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_imports() 