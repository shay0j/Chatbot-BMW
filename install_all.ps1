# install_all_final.ps1
Write-Host "=== INSTALACJA WSZYSTKIEGO ===" -ForegroundColor Cyan

$groups = @(
    # Grupa 1: Core FastAPI
    @("fastapi==0.110.0", "uvicorn[standard]==0.29.0", "starlette==0.36.3", "python-multipart==0.0.9"),
    
    # Grupa 2: Logging & Config
    @("loguru==0.7.2", "python-dotenv==1.0.1", "pydantic==2.7.0", "pydantic-settings==2.2.1"),
    
    # Grupa 3: HTTP & Web
    @("httpx==0.27.0", "requests==2.31.0", "aiohttp==3.9.5", "aiofiles==23.2.1", "websockets==13.0"),
    
    # Grupa 4: Data processing
    @("pandas==2.2.2", "numpy==1.26.4", "orjson==3.10.0", "msgpack==1.0.8", "python-dateutil==2.9.0.post0", "pytz==2024.1"),
    
    # Grupa 5: AI Core - POPRAWIONE tokenizers na 0.15.2!
    @("tokenizers==0.15.2", "transformers==4.39.0", "torch==2.2.1", "sentence-transformers==2.7.0"),
    
    # Grupa 6: AI LangChain + chromadb JEST TU!
    @("openai==1.25.1", "tiktoken==0.7.0", "chromadb==0.4.24", "langchain==0.1.20"),
    
    # Grupa 7: langchain-community osobno
    @("langchain-community"),
    
    # Grupa 8: Vector DB
    @("faiss-cpu==1.13.2", "hnswlib==0.7.0", "annoy==1.17.3"),
    
    # Grupa 9: Web Scraping
    @("beautifulsoup4==4.12.3", "lxml==5.2.1", "selenium==4.19.0"),
    
    # Grupa 10: Databases (poprawione sqlalchemy z 2.2.29 na 2.0.29)
    @("sqlalchemy==2.0.29", "redis==5.0.6", "pymongo==4.6.3", "duckdb==0.10.0"),
    
    # Grupa 11: Development
    @("pytest==7.4.0", "pytest-asyncio==0.23.5", "black==24.3.0", "ipython==8.23.0")
)

foreach ($group in $groups) {
    Write-Host "`nInstalling: $($group -join ', ')" -ForegroundColor Yellow
    
    # Sprawdź czy to langchain-community (instalujemy bez wersji)
    if ($group[0] -eq "langchain-community") {
        pip install langchain-community
    } else {
        # Dla pozostałych grup używamy $group
        pip install $group
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "UWAGA: Błąd w grupie, kontynuuję..." -ForegroundColor Red
    }
}

# Playwright osobno
Write-Host "`n=== PLAYWRIGHT INSTALL ===" -ForegroundColor Cyan
pip install playwright==1.46.0
playwright install

# firecrawl-py
Write-Host "`n=== FIRECRAWL ===" -ForegroundColor Cyan
pip install firecrawl-py==4.12.0

Write-Host "`n=== SPRAWDZENIE CHROMADB ===" -ForegroundColor Green
pip show chromadb

Write-Host "`n=== SPRAWDZENIE WSZYSTKIEGO ===" -ForegroundColor Green
$check_packages = @("fastapi", "uvicorn", "loguru", "openai", "langchain", "torch", "transformers", "chromadb", "pydantic")
foreach ($pkg in $check_packages) {
    $version = pip show $pkg 2>$null | Select-String -Pattern "Version:" | ForEach-Object { $_.ToString().Split(":")[1].Trim() }
    if ($version) {
        Write-Host "✓ $pkg == $version" -ForegroundColor Green
    } else {
        Write-Host "✗ $pkg - BRAK" -ForegroundColor Red
    }
}

Write-Host "`n=== TEST URUCHOMIENIA ===" -ForegroundColor Cyan
Write-Host "Uruchom: uvicorn app.main:app --reload" -ForegroundColor Yellow