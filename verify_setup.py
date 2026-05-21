#!/usr/bin/env python3
"""
Setup Verification Script
Checks if all requirements are met to run the pipeline
"""

import sys
import subprocess
import json
from pathlib import Path

class SetupVerifier:
    """Verify project setup"""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []
    
    def print_header(self):
        """Print welcome header"""
        print("=" * 80)
        print("🔍 NLP PIPELINE - SETUP VERIFICATION")
        print("=" * 80)
        print()
    
    def check(self, name: str, condition: bool, error_msg: str = "") -> bool:
        """
        Perform a check
        
        Args:
            name: Check name
            condition: True if check passes
            error_msg: Error message if check fails
        
        Returns:
            True if check passes
        """
        if condition:
            print(f"✓ {name}")
            self.checks_passed += 1
            return True
        else:
            print(f"✗ {name}")
            if error_msg:
                print(f"  → {error_msg}")
            self.checks_failed += 1
            return False
    
    def warning(self, msg: str):
        """Add warning"""
        print(f"⚠ {msg}")
        self.warnings.append(msg)
    
    def check_python_version(self):
        """Check Python version"""
        print("\n1️⃣  PYTHON SETUP")
        print("-" * 80)
        
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        self.check(
            f"Python version ({version_str})",
            version.major >= 3 and version.minor >= 8,
            f"Requires Python 3.8+, found {version_str}"
        )
    
    def check_virtual_environment(self):
        """Check if in virtual environment"""
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        
        self.check(
            "Virtual environment",
            in_venv,
            "Recommended: Run from a virtual environment (python -m venv venv)"
        )
        
        if not in_venv:
            self.warning("Not using virtual environment - setup in venv recommended")
    
    def check_dependencies(self):
        """Check installed Python packages"""
        print("\n2️⃣  PYTHON DEPENDENCIES")
        print("-" * 80)
        
        required = [
            'torch',
            'transformers',
            'requests',
            'pandas',
            'sklearn',
            'pyyaml',
            'rouge_score',
            'bert_score',
        ]
        
        installed = {}
        try:
            import pkg_resources
            for package in pkg_resources.working_set:
                installed[package.key] = package.version
        except:
            pass
        
        for package in required:
            # Handle sklearn special case
            check_name = 'scikit-learn' if package == 'sklearn' else package
            found = package in installed or check_name in installed
            
            version = installed.get(package) or installed.get(check_name, "unknown")
            
            if found:
                self.check(f"{package} ({version})", True)
            else:
                self.check(
                    f"{package}",
                    False,
                    f"Install with: pip install -r requirements.txt"
                )
    
    def check_ollama(self):
        """Check Ollama installation and connection"""
        print("\n3️⃣  OLLAMA SETUP")
        print("-" * 80)
        
        # Check if ollama command exists
        try:
            result = subprocess.run(['ollama', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            ollama_version = result.stdout.strip() if result.returncode == 0 else "unknown"
            self.check(f"Ollama installed ({ollama_version})", True)
        except FileNotFoundError:
            self.check(
                "Ollama installed",
                False,
                "Download from https://ollama.ai and install"
            )
            return
        
        # Check if Ollama server is running
        try:
            import requests
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            self.check("Ollama server running", response.status_code == 200)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_count = len(models)
                if model_count > 0:
                    print(f"  Available models: {model_count}")
                    for model in models[:5]:
                        print(f"    - {model['name']}")
                    if model_count > 5:
                        print(f"    ... and {model_count - 5} more")
                else:
                    self.warning("No models downloaded. Run: ollama pull llama2")
        except Exception as e:
            self.check(
                "Ollama server running",
                False,
                "Start with: ollama serve (in another terminal)"
            )
    
    def check_project_structure(self):
        """Check project file structure"""
        print("\n4️⃣  PROJECT STRUCTURE")
        print("-" * 80)
        
        required_files = [
            'main.py',
            'pipeline.py',
            'config.py',
            'config.yaml',
            'requirements.txt',
            'README.md',
            'tema_za_master_rad.txt',
        ]
        
        for file in required_files:
            path = Path(file)
            self.check(f"{file}", path.exists())
        
        # Check for optional directories
        print("\nOptional directories:")
        for dir in ['results', 'data']:
            path = Path(dir)
            if path.exists():
                print(f"✓ {dir}/ exists")
            else:
                print(f"  {dir}/ (will be created automatically)")
    
    def check_configuration(self):
        """Check configuration file"""
        print("\n5️⃣  CONFIGURATION")
        print("-" * 80)
        
        try:
            import yaml
            with open('config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            self.check("config.yaml valid", True)
            
            # Show some config values
            if 'model' in config:
                print(f"  Model: {config['model'].get('name', 'unknown')}")
            if 'pipeline' in config:
                print(f"  Technique: {config['pipeline'].get('prompting_technique', 'unknown')}")
        
        except Exception as e:
            self.check("config.yaml valid", False, str(e))
    
    def print_summary(self):
        """Print verification summary"""
        print("\n" + "=" * 80)
        print("📊 SUMMARY")
        print("=" * 80)
        
        total = self.checks_passed + self.checks_failed
        
        print(f"\nChecks passed: {self.checks_passed}/{total}")
        
        if self.checks_failed == 0:
            print("\n✅ All checks passed! You're ready to run the pipeline:")
            print("\n   python main.py")
        else:
            print(f"\n❌ {self.checks_failed} check(s) failed. See above for details.")
            print("\n💡 For help, see:")
            print("   - QUICKSTART.md - Quick setup guide")
            print("   - README.md - Complete documentation")
            print("   - Troubleshooting section in README.md")
        
        if self.warnings:
            print(f"\n⚠  {len(self.warnings)} warning(s):")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        print()
    
    def run(self):
        """Run all checks"""
        self.print_header()
        
        self.check_python_version()
        self.check_virtual_environment()
        self.check_dependencies()
        self.check_ollama()
        self.check_project_structure()
        self.check_configuration()
        
        self.print_summary()
        
        return self.checks_failed == 0

def main():
    """Main function"""
    verifier = SetupVerifier()
    success = verifier.run()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
