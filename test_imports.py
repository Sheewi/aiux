#!/usr/bin/env python3
"""
Test script to verify all required imports are working
Run after installation to validate the environment
"""

import sys
import importlib
from typing import List, Tuple

def test_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """Test if a module can be imported successfully"""
    try:
        if package_name:
            # For packages with different import names
            importlib.import_module(module_name)
            return True, f"✅ {package_name} ({module_name}) imported successfully"
        else:
            importlib.import_module(module_name)
            return True, f"✅ {module_name} imported successfully"
    except ImportError as e:
        package = package_name or module_name
        return False, f"❌ {package} failed: {str(e)}"
    except Exception as e:
        package = package_name or module_name
        return False, f"⚠️ {package} error: {str(e)}"

def main():
    """Test all critical imports"""
    print("🧪 Testing Critical Package Imports")
    print("=" * 50)
    
    # Built-in modules (should always work)
    builtin_modules = [
        ("sqlite3", None),
        ("socket", None),
        ("configparser", None),
    ]
    
    # Core dependencies with pinned versions
    core_modules = [
        ("click", "click==8.1.7"),
        ("redis", "redis==5.0.1"),
        ("apscheduler", "APScheduler==3.10.1"),
        ("paramiko", "paramiko==3.4.0"),
        ("fabric", "fabric==3.1.0"),
    ]
    
    # Essential packages
    essential_modules = [
        ("requests", None),
        ("fastapi", None),
        ("pydantic", None),
        ("pandas", None),
        ("numpy", None),
    ]
    
    # Optional but important
    optional_modules = [
        ("google.cloud.aiplatform", "google-cloud-aiplatform"),
        ("kubernetes", None),
        ("docker", None),
        ("cryptography", None),
    ]
    
    all_passed = True
    failed_modules = []
    
    # Test built-in modules
    print("\n📦 Testing Built-in Modules:")
    for module, package in builtin_modules:
        success, message = test_import(module, package)
        print(f"  {message}")
        if not success:
            all_passed = False
            failed_modules.append(package or module)
    
    # Test core dependencies
    print("\n🔧 Testing Core Dependencies:")
    for module, package in core_modules:
        success, message = test_import(module, package)
        print(f"  {message}")
        if not success:
            all_passed = False
            failed_modules.append(package or module)
    
    # Test essential packages
    print("\n⚡ Testing Essential Packages:")
    for module, package in essential_modules:
        success, message = test_import(module, package)
        print(f"  {message}")
        if not success:
            all_passed = False
            failed_modules.append(package or module)
    
    # Test optional packages
    print("\n🔍 Testing Optional Packages:")
    optional_failed = 0
    for module, package in optional_modules:
        success, message = test_import(module, package)
        print(f"  {message}")
        if not success:
            optional_failed += 1
    
    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All critical imports working!")
        if optional_failed > 0:
            print(f"⚠️ {optional_failed} optional packages failed (this is usually OK)")
    else:
        print("❌ Some critical imports failed!")
        print(f"Failed modules: {', '.join(failed_modules)}")
        
        print("\n🔧 Troubleshooting suggestions:")
        print("1. Run: pip install --upgrade --force-reinstall <failed_package>")
        print("2. Try: ./install_requirements_alt.sh")
        print("3. Check: pip list | grep <package_name>")
        
        sys.exit(1)
    
    # Version information
    print("\n📋 Python Environment Info:")
    print(f"  Python version: {sys.version}")
    print(f"  Python executable: {sys.executable}")
    
    # Try to get versions of key packages
    try:
        import click
        print(f"  Click version: {click.__version__}")
    except:
        pass
    
    try:
        import redis
        print(f"  Redis version: {redis.__version__}")
    except:
        pass
    
    print("\n✨ Environment validation complete!")

if __name__ == "__main__":
    main()
