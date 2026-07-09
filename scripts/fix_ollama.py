#!/usr/bin/env python3
"""
Ollama Troubleshooting Script for Windows
Helps diagnose and fix Ollama installation issues
"""

import os
import subprocess
import sys
from pathlib import Path

def print_header():
    """Print troubleshooting header"""
    print("=" * 60)
    print("🔧 OLLAMA TROUBLESHOOTING - WINDOWS")
    print("=" * 60)
    print()

def check_ollama_installation():
    """Check if Ollama is installed"""
    print("1️⃣  CHECKING OLLAMA INSTALLATION")
    print("-" * 40)

    # Common installation paths
    possible_paths = [
        r"C:\Program Files\Ollama\ollama.exe",
        r"C:\Program Files (x86)\Ollama\ollama.exe",
        r"C:\Users\{}\AppData\Local\Programs\Ollama\ollama.exe".format(os.environ.get('USERNAME', 'User')),
        r"C:\Users\{}\AppData\Local\Ollama\ollama.exe".format(os.environ.get('USERNAME', 'User')),
        r"C:\Ollama\ollama.exe",
    ]

    found_paths = []
    for path in possible_paths:
        if Path(path).exists():
            found_paths.append(path)
            print(f"✅ Found Ollama at: {path}")

    if not found_paths:
        print("❌ Ollama executable not found in common locations")
        print("\n🔍 SEARCHING for ollama.exe...")
        # Search in Program Files
        try:
            result = subprocess.run(['where', 'ollama'], capture_output=True, text=True, shell=True)
            if result.returncode == 0:
                paths = result.stdout.strip().split('\n')
                for path in paths:
                    if path.strip():
                        found_paths.append(path.strip())
                        print(f"✅ Found via 'where': {path.strip()}")
        except:
            pass

    if not found_paths:
        print("❌ Ollama not found. Please reinstall from https://ollama.ai/download/windows")
        return None

    return found_paths[0] if found_paths else None

def check_path_environment():
    """Check if Ollama is in PATH"""
    print("\n2️⃣  CHECKING PATH ENVIRONMENT")
    print("-" * 40)

    path_dirs = os.environ.get('PATH', '').split(';')
    ollama_in_path = any('ollama' in dir.lower() for dir in path_dirs)

    if ollama_in_path:
        print("✅ Ollama appears to be in PATH")
    else:
        print("⚠️  Ollama not found in PATH environment variable")

    # Show current PATH entries related to Ollama
    ollama_paths = [dir for dir in path_dirs if 'ollama' in dir.lower()]
    if ollama_paths:
        print("   Current Ollama PATH entries:")
        for path in ollama_paths:
            print(f"   - {path}")

    return ollama_in_path

def test_ollama_command(ollama_path=None):
    """Test if ollama command works"""
    print("\n3️⃣  TESTING OLLAMA COMMAND")
    print("-" * 40)

    # Try direct command first
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Ollama command works! Version: {version}")
            return True
        else:
            print(f"❌ Ollama command failed: {result.stderr}")
    except FileNotFoundError:
        print("❌ 'ollama' command not found")
    except Exception as e:
        print(f"❌ Error testing ollama command: {e}")

    # Try with full path if we found it
    if ollama_path:
        try:
            result = subprocess.run([ollama_path, '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"✅ Ollama works with full path! Version: {version}")
                print(f"   Full path: {ollama_path}")
                return ollama_path
            else:
                print(f"❌ Ollama failed even with full path: {result.stderr}")
        except Exception as e:
            print(f"❌ Error testing with full path: {e}")

    return False

def fix_path_issue(ollama_path):
    """Try to fix PATH issue"""
    print("\n4️⃣  FIXING PATH ISSUE")
    print("-" * 40)

    if not ollama_path:
        print("❌ No Ollama path found to add to PATH")
        return False

    ollama_dir = str(Path(ollama_path).parent)

    print(f"📁 Ollama directory: {ollama_dir}")

    # Check if directory exists
    if not Path(ollama_dir).exists():
        print(f"❌ Directory doesn't exist: {ollama_dir}")
        return False

    # Try to add to PATH for current session
    current_path = os.environ.get('PATH', '')
    if ollama_dir not in current_path:
        new_path = current_path + ';' + ollama_dir
        os.environ['PATH'] = new_path
        print(f"✅ Added to PATH for current session: {ollama_dir}")

        # Test if it works now
        try:
            result = subprocess.run(['ollama', '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("✅ PATH fix successful! Ollama command now works.")
                return True
        except:
            pass

    print("⚠️  PATH fix didn't work. You may need to:")
    print("   1. Restart your computer")
    print("   2. Or manually add to system PATH:")
    print(f"      - Go to System Properties > Environment Variables")
    print(f"      - Add '{ollama_dir}' to PATH variable")
    print("   3. Restart Command Prompt/PowerShell")

    return False

def provide_solutions():
    """Provide comprehensive solutions"""
    print("\n" * 2)
    print("=" * 60)
    print("🛠️  COMPREHENSIVE SOLUTIONS")
    print("=" * 60)

    print("\n📋 STEP-BY-STEP FIX:")
    print("1. Close all Command Prompt and PowerShell windows")
    print("2. Restart your computer completely")
    print("3. Open a new Command Prompt as Administrator")
    print("4. Try: ollama --version")

    print("\n🔧 MANUAL PATH FIX:")
    print("If restart doesn't work:")
    print("1. Press Win + R, type 'sysdm.cpl', press Enter")
    print("2. Go to 'Advanced' tab > 'Environment Variables'")
    print("3. Under 'System variables', find 'Path', click 'Edit'")
    print("4. Click 'New' and add: C:\\Program Files\\Ollama\\")
    print("5. Click OK, restart computer")

    print("\n🔄 ALTERNATIVE START METHODS:")
    print("If PATH still doesn't work, use full path:")
    print("1. Find ollama.exe location (usually C:\\Program Files\\Ollama\\)")
    print("2. Start server: \"C:\\Program Files\\Ollama\\ollama.exe\" serve")
    print("3. Download models: \"C:\\Program Files\\Ollama\\ollama.exe\" pull llama2")

    print("\n📞 STILL HAVING ISSUES?")
    print("- Check Windows Event Viewer for installation errors")
    print("- Try reinstalling Ollama as Administrator")
    print("- Check antivirus/firewall blocking Ollama")
    print("- Visit: https://github.com/jmorganca/ollama/issues")

def main():
    """Main troubleshooting function"""
    print_header()

    # Step 1: Check installation
    ollama_path = check_ollama_installation()

    # Step 2: Check PATH
    path_ok = check_path_environment()

    # Step 3: Test command
    command_works = test_ollama_command(ollama_path)

    # Step 4: Try to fix if needed
    if not command_works and ollama_path:
        fix_path_issue(ollama_path)

    # Step 5: Final test
    print("\n5️⃣  FINAL VERIFICATION")
    print("-" * 40)
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("🎉 SUCCESS! Ollama is working correctly")
            print(f"   Version: {result.stdout.strip()}")
            print("\n🚀 You can now run:")
            print("   ollama serve    (start server)")
            print("   ollama pull llama2  (download model)")
            print("   streamlit run app.py  (launch UI)")
        else:
            print("❌ Still not working. See solutions below.")
            provide_solutions()
    except:
        print("❌ Ollama command still not working.")
        provide_solutions()

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
