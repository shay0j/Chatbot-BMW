from firecrawl import FirecrawlApp
import inspect

# Sprawdź dostępne metody
app = FirecrawlApp(api_key="dummy_key")

print("🔍 Dostępne metody w FirecrawlApp:")
methods = [m for m in dir(app) if not m.startswith('_')]
for method in methods:
    print(f"  - {method}")

print("\n🔍 Sprawdźmy typ obiektu:")
print(f"  Typ: {type(app)}")
print(f"  Moduł: {app.__module__}")

# Sprawdź źródło
try:
    print(f"\n🔍 Źródło klasy:")
    print(inspect.getsource(app.__class__))
except:
    pass

# Sprawdź dostępne atrybuty
print("\n🔍 Sprawdźmy dostępne atrybuty poprzez __dict__:")
if hasattr(app, '__dict__'):
    for key in app.__dict__:
        print(f"  {key}: {type(app.__dict__[key])}")