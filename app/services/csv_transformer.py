"""
CSV Transformer — konwertuje surowe dane CSV na teksty czytelne dla człowieka.
Rozwiązuje problem złych osadzeń wektorowych (embeddings) CSV → FAISS.

PROBLEM: Surowe wiersze CSV ("Type: SUV\nPower_hp: 248") nie pasują semantycznie
do naturalnych zapytań ("opowiedz mi o X3").

ROZWIĄZANIE: Przekształcamy dane w czytelne po polsku akapity, które lepiej
osadzają się w przestrzeni wektorowej i dają trafniejsze wyniki wyszukiwania.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path


# ============================================
# MAPOWANIE NAZW (angielskie kolumny → polski)
# ============================================

POWERTRAIN_MAP = {
    "Petrol": "benzynowy",
    "Diesel": "diesel",
    "Plug-in Hybrid": "hybryda plug-in",
    "Electric": "elektryczny",
    "Mild Hybrid": "mild hybrid",
}

ENGINE_CONFIG_MAP = {
    "Inline-3": "3-cylindrowy rzędowy",
    "Inline-4": "4-cylindrowy rzędowy",
    "Inline-6": "6-cylindrowy rzędowy",
    "V8": "V8",
    "V12": "V12",
    "Electric Motor": "silnik elektryczny",
}

BODY_STYLE_MAP = {
    "SUV": "SUV",
    "Sedan": "sedan",
    "Wagon": "kombi (Touring)",
    "Coupe": "coupé",
    "Convertible": "kabriolet",
    "Hatchback": "hatchback",
    "Roadster": "roadster",
    "Gran Coupe": "Gran Coupé",
    "Sports Activity Vehicle": "SAV",
}

DRIVE_TYPE_MAP = {
    "AWD": "napęd na wszystkie koła (xDrive)",
    "RWD": "napęd na tylną oś",
    "FWD": "napęd na przednią oś",
}


def _safe_val(row: pd.Series, col: str) -> Optional[str]:
    """Bezpiecznie pobiera wartość z wiersza CSV"""
    val = row.get(col)
    if pd.isna(val) or str(val).strip() == "":
        return None
    return str(val).strip()


def _safe_num(row: pd.Series, col: str) -> Optional[str]:
    """Pobiera wartość liczbową, usuwa .0 jeśli to int"""
    val = _safe_val(row, col)
    if val is None:
        return None
    try:
        num = float(val)
        if num == int(num):
            return str(int(num))
        return val
    except (ValueError, TypeError):
        return val


# ============================================
# TRANSFORMACJA POJEDYNCZEGO WIERSZA CSV
# ============================================

def transform_csv_row_to_text(row: pd.Series) -> str:
    """
    Przekształca jeden wiersz CSV w czytelny po polsku opis modelu BMW.
    
    Zamiast: "Type: SUV\\nModel: X3\\nPower_hp: 248\\nTorque_Nm: 350"
    Tworzy:  "BMW X3 to SUV z napędem AWD. Silnik benzynowy 248 KM, 350 Nm..."
    """
    parts = []
    
    model = _safe_val(row, "Model") or _safe_val(row, "Series") or "nieznany"
    body = _safe_val(row, "BodyStyle")
    body_pl = BODY_STYLE_MAP.get(body, body) if body else None
    
    # Linia 1: Model + typ nadwozia
    header = f"BMW {model}"
    if body_pl:
        header += f" to {body_pl}"
    parts.append(header + ".")
    
    # Linia 2: Napęd
    powertrain = _safe_val(row, "Powertrain")
    powertrain_pl = POWERTRAIN_MAP.get(powertrain, powertrain) if powertrain else None
    engine_config = _safe_val(row, "EngineConfig")
    engine_pl = ENGINE_CONFIG_MAP.get(engine_config, engine_config) if engine_config else None
    displacement = _safe_val(row, "Displacement_L")
    
    engine_parts = []
    if powertrain_pl:
        engine_parts.append(f"Silnik {powertrain_pl}")
    if engine_pl:
        engine_parts.append(f"({engine_pl}")
        if displacement:
            engine_parts.append(f"{displacement}L)")
        else:
            engine_parts.append(")")
    if engine_parts:
        parts.append(" ".join(engine_parts) + ".")
    
    # Linia 3: Moc i moment
    power = _safe_num(row, "Power_hp")
    torque = _safe_num(row, "Torque_Nm")
    perf_parts = []
    if power:
        perf_parts.append(f"Moc: {power} KM")
    if torque:
        perf_parts.append(f"moment obrotowy: {torque} Nm")
    if perf_parts:
        parts.append(", ".join(perf_parts) + ".")
    
    # Linia 4: Skrzynia i napęd
    transmission = _safe_val(row, "Transmission")
    gears = _safe_num(row, "Gears")
    drive = _safe_val(row, "DriveType")
    drive_pl = DRIVE_TYPE_MAP.get(drive, drive) if drive else None
    
    trans_parts = []
    if transmission and gears:
        trans_parts.append(f"Skrzynia {transmission.lower()} {gears}-biegowa")
    elif transmission:
        trans_parts.append(f"Skrzynia {transmission.lower()}")
    if drive_pl:
        trans_parts.append(drive_pl)
    if trans_parts:
        parts.append(", ".join(trans_parts) + ".")
    
    # Linia 5: Osiągi
    accel = _safe_val(row, "Accel_0_100_kmh_s")
    top_speed = _safe_num(row, "TopSpeed_kmh")
    perf = []
    if accel:
        perf.append(f"Przyspieszenie 0-100 km/h: {accel}s")
    if top_speed:
        perf.append(f"prędkość maksymalna: {top_speed} km/h")
    if perf:
        parts.append(", ".join(perf) + ".")
    
    # Linia 6: Wymiary
    length = _safe_num(row, "Length_mm")
    width = _safe_num(row, "Width_mm")
    height = _safe_num(row, "Height_mm")
    wheelbase = _safe_num(row, "Wheelbase_mm")
    if length and width and height:
        dim_text = f"Wymiary: {length} x {width} x {height} mm"
        if wheelbase:
            dim_text += f", rozstaw osi: {wheelbase} mm"
        parts.append(dim_text + ".")
    
    # Linia 7: Praktyczne dane
    seats = _safe_num(row, "Seats")
    cargo = _safe_num(row, "Cargo_L")
    weight = _safe_num(row, "CurbWeight_kg")
    pract = []
    if seats:
        pract.append(f"{seats} miejsc")
    if cargo:
        pract.append(f"bagażnik: {cargo} litrów")
    if weight:
        pract.append(f"masa: {weight} kg")
    if pract:
        parts.append(", ".join(pract).capitalize() + ".")
    
    # Linia 8: Bateria (dla EV/PHEV)
    battery = _safe_val(row, "Battery_kWh")
    if battery:
        parts.append(f"Bateria: {battery} kWh.")
    
    # Linia 9: Hamulce i zawieszenie
    front_susp = _safe_val(row, "FrontSuspension")
    front_brake = _safe_val(row, "FrontBrakes")
    if front_susp or front_brake:
        tech = []
        if front_susp:
            tech.append(f"zawieszenie: {front_susp}")
        if front_brake:
            tech.append(f"hamulce: {front_brake}")
        parts.append("Dane techniczne: " + ", ".join(tech) + ".")
    
    # Linia 10: Cena
    price = _safe_val(row, "PLN")
    if price:
        parts.append(f"Cena: {price}.")
    
    return "\n".join(parts)


# ============================================
# GRUPOWANIE WARIANTÓW MODELI
# ============================================

def group_and_build_model_documents(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Grupuje wiersze CSV według modelu BMW i tworzy jeden bogaty dokument na model.
    
    Zamiast 15 oddzielnych chunków dla X3 (3 silniki × 5 cen),
    tworzy 1 kompleksowy dokument ze wszystkimi wariantami.
    
    Returns:
        Lista słowników z kluczami: 'text', 'metadata'
    """
    documents = []
    
    # Grupuj po modelu
    if "Model" not in df.columns:
        print("[WARNING] CSV nie ma kolumny 'Model' - indeksowanie wiersz po wierszu")
        for idx, row in df.iterrows():
            text = transform_csv_row_to_text(row)
            documents.append({
                "text": text,
                "metadata": {
                    "type": "csv",
                    "category": "model_specs",
                    "filename": "BMW_models.csv",
                    "row_id": idx,
                }
            })
        return documents
    
    grouped = df.groupby("Model")
    
    for model_name, group in grouped:
        parts = [f"# BMW {model_name}\n"]
        
        # Zbierz unikalne warianty napędu
        seen_powertrains = set()
        variant_texts = []
        prices = set()
        
        for _, row in group.iterrows():
            powertrain = _safe_val(row, "Powertrain") or "unknown"
            power = _safe_num(row, "Power_hp") or "?"
            variant_key = f"{powertrain}_{power}"
            
            # Dodaj cenę
            price = _safe_val(row, "PLN")
            if price:
                prices.add(price)
            
            # Unikaj duplikatów tego samego wariantu napędu
            if variant_key not in seen_powertrains:
                seen_powertrains.add(variant_key)
                variant_texts.append(transform_csv_row_to_text(row))
        
        if len(variant_texts) == 1:
            parts.append(variant_texts[0])
        else:
            parts.append(f"BMW {model_name} jest dostępny w {len(variant_texts)} wariantach napędowych:\n")
            for i, vt in enumerate(variant_texts, 1):
                parts.append(f"--- Wariant {i} ---")
                parts.append(vt)
                parts.append("")
        
        # Podsumowanie cen
        if prices:
            sorted_prices = sorted(prices)
            if len(sorted_prices) > 1:
                parts.append(f"\nCeny BMW {model_name}: od {sorted_prices[0]} do {sorted_prices[-1]}.")
            else:
                parts.append(f"\nCena BMW {model_name}: {sorted_prices[0]}.")
        
        full_text = "\n".join(parts)
        
        # Zbierz metadane
        first_row = group.iloc[0]
        body_style = _safe_val(first_row, "BodyStyle") or ""
        segment = _safe_val(first_row, "Segment") or ""
        series = _safe_val(first_row, "Series") or model_name
        
        documents.append({
            "text": full_text,
            "metadata": {
                "type": "csv",
                "category": "model_specs",
                "filename": "BMW_models.csv",
                "model": model_name,
                "series": series,
                "body_style": body_style,
                "segment": segment,
                "variants_count": len(variant_texts),
                "title": f"BMW {model_name} - specyfikacja techniczna",
            }
        })
    
    return documents


# ============================================
# KLASYFIKACJA PLIKÓW TXT
# ============================================

# Mapowanie nazw plików → kategoria treści
TXT_CATEGORY_MAP = {
    "leasing": "leasing",
    "serwis": "service",
    "trade_in": "service",
    "linki_akcesoria": "links",
    "linki_dostepne": "links",
    "linki_katalogi": "links",
    "linki_konfigurator": "links",
    "motocykle": "motorcycles",
    "premium_selection": "sales",
    "sedan": "model_specs",
    "modele_bmw": "model_specs",
    "modele_uzupelnienie": "model_specs",
    "silniki_bmw": "model_specs",
    "wyposazenie": "model_specs",
}


def classify_txt_file(filename: str) -> str:
    """Klasyfikuje plik TXT do kategorii na podstawie nazwy pliku"""
    filename_lower = filename.lower()
    for keyword, category in TXT_CATEGORY_MAP.items():
        if keyword in filename_lower:
            return category
    return "general"
