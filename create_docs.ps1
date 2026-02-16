# Skrypt do utworzenia dokumentów - uruchom w PowerShell
Write-Host "Tworzenie dokumentów BMW..." -ForegroundColor Green

# 1. Utwórz katalog jeśli nie istnieje
$knowledgeBasePath = ".\data\knowledge_base"
if (-not (Test-Path $knowledgeBasePath)) {
    New-Item -ItemType Directory -Force -Path $knowledgeBasePath
    Write-Host "Utworzono katalog: $knowledgeBasePath" -ForegroundColor Yellow
} else {
    Write-Host "Katalog już istnieje: $knowledgeBasePath" -ForegroundColor Yellow
}

# 2. Utwórz plik z modelami BMW
$modelFile = ".\data\knowledge_base\modele_bmw.txt"
@"
Modele BMW dostępne w Polsce:

Seria 1 (F40):
- Typ: Hatchback premium
- Silniki: 118i, 120i, 118d, 120d
- Cena: od 150 000 PLN

Seria 3 (G20):
- Typ: Sedan średniej klasy
- Silniki: 318i, 320i, 330i, 318d, 320d, 330d, 330e
- Cena: od 200 000 PLN

Seria 5 (G30):
- Typ: Sedan biznesowy
- Silniki: 520i, 530i, 540i, 520d, 530d, 540d, 530e
- Cena: od 300 000 PLN

Seria 7 (G70):
- Typ: Flagowy sedan luksusowy
- Silniki: 740i, 760i, 740d
- Cena: od 500 000 PLN

SUV-y:
- X1 (U11): mały SUV, od 180 000 PLN
- X3 (G01): średni SUV, od 250 000 PLN  
- X5 (G05): duży SUV, od 350 000 PLN
- X7 (G07): pełnowymiarowy SUV, od 450 000 PLN

Sportowe:
- Z4 (G29): roadster, od 250 000 PLN
- M2 (G87): coupé sportowe, od 350 000 PLN
- M3 (G80): sedan sportowy, od 400 000 PLN
- M4 (G82): coupé sportowe, od 420 000 PLN

Elektryczne:
- i4: sedan elektryczny, od 250 000 PLN
- iX: SUV elektryczny, od 400 000 PLN
- i7: sedan elektryczny, od 600 000 PLN
"@ | Out-File -FilePath $modelFile -Encoding UTF8
Write-Host "Utworzono: modele_bmw.txt" -ForegroundColor Green

# 3. Utwórz plik z silnikami
$engineFile = ".\data\knowledge_base\silniki_bmw.txt"
@"
Silniki BMW dostępne w Polsce:

BENZY NOWE:
- B38: 1.5L 3-cylindrowy, 136-140 KM
- B48: 2.0L 4-cylindrowy, 156-245 KM
- B58: 3.0L 6-cylindrowy, 340-387 KM
- S58: 3.0L 6-cylindrowy M, 480-510 KM

DIESLE:
- B37: 1.5L 3-cylindrowy, 116 KM
- B47: 2.0L 4-cylindrowy, 150-190 KM
- B57: 3.0L 6-cylindrowy, 286-340 KM

HYBRYDY:
- 330e: 2.0L + silnik elektryczny, 292 KM
- 530e: 2.0L + silnik elektryczny, 299 KM
- 745e: 3.0L + silnik elektryczny, 394 KM

ELEKTRYCZNE:
- eDrive35: 286 KM, zasięg 400 km
- eDrive40: 340 KM, zasięg 500 km
- M60: 619 KM, zasięg 450 km

MOCY TYPOWE:
- 118i: 136 KM
- 320i: 184 KM  
- 330i: 245 KM
- 520d: 190 KM
- 530d: 286 KM
- M3: 510 KM
"@ | Out-File -FilePath $engineFile -Encoding UTF8
Write-Host "Utworzono: silniki_bmw.txt" -ForegroundColor Green

# 4. Utwórz plik z wyposażeniem
$equipmentFile = ".\data\knowledge_base\wyposazenie_bmw.txt"
@"
Pakiety wyposażenia BMW:

ADVANTAGE (podstawowy):
- Klimatyzacja automatyczna
- System nawigacji Professional
- Światła LED
- Skórzana tapicerka Sensatec
- Asystent pasa ruchu

LUXURY LINE (elegancki):
- Wszystko z Advantage PLUS:
- Chromowane akcenty
- Skóra Vernasca
- Fotele z pamięcią
- Światła przeciwmgielne LED
- Dziewięciocalowe felgi

M SPORT (sportowy):
- Wszystko z Advantage PLUS:
- Obniżone zawieszenie
- Sportowe fotele
- Kierownica M
- Aluminiowe wstawki
- Dyfuzor M Sport

PAKIETY DODATKOWE:
- Pakiet Driving Assistant: asystent pasa, kontrola martwego pola
- Pakiet Comfort: podgrzewane fotele, bezkluczykowy dostęp
- Pakiet Premium: system audio Harman Kardon, hud
- Pakiet M Pro: hamulce M, dyfuzor węglowy

OPCJE INDYWIDUALNE:
- Skóra Merino
- Drewniane wykończenia
- Dach panoramiczny
- System audio Bowers & Wilkins
- Pakiet zimowy
"@ | Out-File -FilePath $equipmentFile -Encoding UTF8
Write-Host "Utworzono: wyposazenie_bmw.txt" -ForegroundColor Green

# 5. Pokaż podsumowanie
Write-Host "`nPODSUMOWANIE:" -ForegroundColor Cyan
Write-Host "=" * 50
Get-ChildItem ".\data\knowledge_base" | Format-Table Name, @{Name="Size(KB)";Expression={[math]::Round($_.Length/1KB,2)}} -AutoSize
Write-Host "`nPliki gotowe! Możesz teraz uruchomić: python simple_load.py" -ForegroundColor Cyan