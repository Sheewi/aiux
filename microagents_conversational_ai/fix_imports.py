#!/usr/bin/env python3
"""
Fix all base_agent import issues throughout the codebase.
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Fix imports in a single file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Fix base_agent imports
        patterns = [
            (r'from base_agent import (.+)', r'from generated_agents.base_agent import \1'),
            (r'from \.base_agent import (.+)', r'from generated_agents.base_agent import \1'),
            (r'import base_agent', r'import generated_agents.base_agent as base_agent'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        # Fix registry imports
        registry_patterns = [
            (r'from registry import register\b', r'from registry import register_agent'),
            (r'@register\(\)', r'@register_agent()'),
        ]
        
        for pattern, replacement in registry_patterns:
            content = re.sub(pattern, replacement, content)
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            return True, f"Fixed imports in {file_path}"
        else:
            return False, f"No changes needed in {file_path}"
    
    except Exception as e:
        return False, f"Error processing {file_path}: {e}"

def fix_all_imports():
    """Fix imports in all Python files."""
    base_dir = Path(__file__).parent
    python_files = list(base_dir.glob("**/*.py"))
    
    fixed_count = 0
    total_count = 0
    
    print("🔧 Fixing import issues...")
    print("=" * 50)
    
    for file_path in python_files:
        # Skip this script itself and __pycache__ directories
        if file_path.name == "fix_imports.py" or "__pycache__" in str(file_path):
            continue
        
        total_count += 1
        changed, message = fix_imports_in_file(file_path)
        
        if changed:
            fixed_count += 1
            print(f"✅ {message}")
        else:
            print(f"⏭️  {message}")
    
    print("=" * 50)
    print(f"📊 Summary: Fixed {fixed_count}/{total_count} files")
    
    # Test the imports after fixing
    print("\n🧪 Testing imports...")
    test_imports()

def test_imports():
    """Test that the imports work correctly."""
    try:
        # Test basic imports
        from generated_agents.base_agent import MicroAgent, HybridAgent, BaseInput, BaseOutput
        print("✅ Basic base_agent imports work")
        
        # Test registry imports
        from registry import agent_registry, register_agent
        print("✅ Registry imports work")
        
        # Test multimodal router
        from multimodal_context_router import MultimodalContextRouter
        print("✅ Multimodal router imports work")
        
        print("🎉 All critical imports working!")
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    fix_all_imports()
