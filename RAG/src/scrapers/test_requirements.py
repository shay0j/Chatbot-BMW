# test_real_versions.py
import importlib.metadata
import sys

packages = [
    'playwright',
    'beautifulsoup4', 
    'requests',
    'aiofiles',
    'cohere',
    'chromadb',
    'langchain',
    'pandas',
    'pydantic',
    'fastapi',
    'uvicorn',
    'python-dotenv'
]

print("🔍 ACTUAL INSTALLED VERSIONS:")
print("=" * 40)

for package in packages:
    try:
        version = importlib.metadata.version(package)
        print(f"✅ {package}: {version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"❌ {package}: NOT FOUND")
    except Exception as e:
        print(f"⚠️  {package}: Error - {e}")

print(f"\n🐍 Python: {sys.version}")