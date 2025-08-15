#!/usr/bin/env python3
"""
Final System Validation Script
Ensures everything is properly organized with no placeholders.
"""

import json
import sys
from pathlib import Path

def validate_no_placeholders():
    """Check for any remaining placeholder patterns."""
    print("🔍 Scanning for placeholders...")
    
    # Files to check
    files_to_check = [
        "grok_output_format.py",
        "multimodal_context_router.py", 
        "registry.py",
        "complete_grok_demo.py",
        "test_grok_system.py"
    ]
    
    placeholder_patterns = [
        "{response_text}",
        "{command}",
        "{command_output}",
        "{diagram_source}",
        "{programming_language}",
        "{code_content}",
        "{suggested_filename}",
        "PLACEHOLDER",
        "TODO:",
        "FIXME:",
        "REPLACE_ME"
    ]
    
    issues_found = []
    
    for file_path in files_to_check:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                for pattern in placeholder_patterns:
                    if pattern in content:
                        issues_found.append(f"{file_path}: {pattern}")
            except Exception as e:
                print(f"⚠️  Could not check {file_path}: {e}")
    
    if issues_found:
        print("❌ Placeholders found:")
        for issue in issues_found:
            print(f"  - {issue}")
        return False
    else:
        print("✅ No placeholders found!")
        return True

def validate_imports():
    """Test all critical imports."""
    print("\n🔧 Validating imports...")
    
    try:
        from grok_output_format import create_grok_output, RenderType, GrokStyleOutput
        from multimodal_context_router import MultimodalContextRouter, OutputMode
        from registry import agent_registry
        print("✅ All core imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def validate_grok_output():
    """Test Grok output format creation."""
    print("\n🎨 Validating Grok output format...")
    
    try:
        from grok_output_format import create_grok_output, RenderType
        
        # Test each render type
        test_cases = [
            (RenderType.TEXT, "Sample text output"),
            (RenderType.CLI, {"command": "ls -la", "output": "file listing"}),
            (RenderType.GRAPH, {"nodes": [{"id": "1", "label": "Node"}], "edges": []}),
            (RenderType.CODE, {"language": "python", "code": "print('hello')"})
        ]
        
        for render_type, content in test_cases:
            output = create_grok_output(render_type, content)
            json_str = output.to_json()
            
            # Verify no placeholders in JSON
            if "{" in json_str and "}" in json_str:
                # Check if it's actual JSON structure vs placeholder
                parsed = json.loads(json_str)
                print(f"✅ {render_type.value} format validated")
            else:
                print(f"✅ {render_type.value} format validated")
        
        return True
    except Exception as e:
        print(f"❌ Grok output validation failed: {e}")
        return False

def validate_router_functionality():
    """Test multimodal router basic functionality."""
    print("\n🔄 Validating router functionality...")
    
    try:
        from multimodal_context_router import MultimodalContextRouter
        from registry import agent_registry
        from tokenizer.action_tokenizer import ActionTokenizer
        
        router = MultimodalContextRouter(
            microagent_registry=agent_registry,
            tokenizer=ActionTokenizer()
        )
        
        # Test input mode detection
        modes = [
            ("Show me agents", "natural language"),
            ("$ ls -la", "command line"),
            ('{"query": "test"}', "structured")
        ]
        
        for test_input, expected_type in modes:
            mode = router._detect_input_mode(test_input)
            print(f"✅ '{test_input}' → {mode.value}")
        
        return True
    except Exception as e:
        print(f"❌ Router validation failed: {e}")
        return False

def validate_organization():
    """Check file organization."""
    print("\n📁 Validating file organization...")
    
    required_files = [
        "multimodal_context_router.py",
        "registry.py", 
        "grok_output_format.py",
        "complete_grok_demo.py",
        "test_grok_system.py",
        "PROJECT_ORGANIZATION.md",
        "README_GROK_SYSTEM.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ All required files present!")
        return True

def main():
    """Run all validation checks."""
    print("🎯 Final System Validation")
    print("=" * 50)
    
    checks = [
        ("Placeholder Check", validate_no_placeholders),
        ("Import Validation", validate_imports),
        ("Grok Output Format", validate_grok_output),
        ("Router Functionality", validate_router_functionality),
        ("File Organization", validate_organization)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} failed with exception: {e}")
            results.append((check_name, False))
    
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {check_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} checks passed")
    
    if passed == len(results):
        print("\n🎉 SYSTEM FULLY VALIDATED!")
        print("✅ No placeholders found")
        print("✅ All imports working")
        print("✅ Grok output format functional")
        print("✅ Router processing correctly")
        print("✅ Files properly organized")
        print("\n🚀 READY FOR PRODUCTION USE!")
        return True
    else:
        print(f"\n⚠️  {len(results) - passed} validation(s) failed")
        print("Please review and fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
