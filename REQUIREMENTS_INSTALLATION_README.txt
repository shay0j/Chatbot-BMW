# Nadaj uprawnienia
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Uruchom skrypt
.\install_all.ps1

# Test uruchomienia
uvicorn app.main:app --reload