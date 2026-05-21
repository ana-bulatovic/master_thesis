#!/usr/bin/env python3
"""
Quick launcher for the NLP Pipeline Web UI
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit web interface"""
    print("🚀 Launching NLP Pipeline Web UI...")
    print("=" * 50)

    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )

    if not in_venv:
        print("⚠️  Warning: Not running in virtual environment")
        print("   Consider activating your venv first:")
        print("   Windows: venv\\Scripts\\activate")
        print("   macOS/Linux: source venv/bin/activate")
        print()

    # Check if app.py exists
    if not os.path.exists('app.py'):
        print("❌ Error: app.py not found in current directory")
        print("   Make sure you're in the project root directory")
        return

    # Launch Streamlit
    try:
        print("🌐 Starting Streamlit server...")
        print("   Open http://localhost:8501 in your browser")
        print("   Press Ctrl+C to stop the server")
        print()

        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'],
                      check=True)

    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching Streamlit: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Streamlit is installed: pip install streamlit")
        print("2. Check if Ollama is running: ollama serve")
        print("3. Try running directly: streamlit run app.py")

    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")

if __name__ == "__main__":
    main()
