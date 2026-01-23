import re

def kw_to_hp(value: str):
    try:
        num = float(re.findall(r"[\d.,]+", value.replace(",", "."))[0])
        return round(num * 1.35962)
    except Exception:
        return None

def lbft_to_nm(value: str):
    try:
        num = float(re.findall(r"[\d.,]+", value.replace(",", "."))[0])
        return round(num * 1.35582)
    except Exception:
        return None

def cm3_to_liters(value: str):
    try:
        num = float(re.findall(r"[\d.,]+", value.replace(",", "."))[0])
        return round(num / 1000, 2)
    except Exception:
        return None

def parse_seconds(value: str):
    try:
        num = float(re.findall(r"[\d.,]+", value.replace(",", "."))[0])
        return num
    except Exception:
        return None

def normalize_specs(specs: dict):
    normalized = {}
    for key, value in specs.items():
        if not value:
            continue
        key_lower = key.lower()
        value_lower = value.lower()
        try:
            if "kw" in value_lower:
                normalized[key_lower + "_hp"] = kw_to_hp(value)
            elif "hp" in value_lower:
                num = re.findall(r"[\d.,]+", value.replace(",", "."))[0]
                normalized[key_lower + "_hp"] = float(num)
            elif "lb-ft" in value_lower or "lbft" in value_lower:
                normalized[key_lower + "_nm"] = lbft_to_nm(value)
            elif "nm" in value_lower:
                num = re.findall(r"[\d.,]+", value.replace(",", "."))[0]
                normalized[key_lower + "_nm"] = float(num)
            elif "cm3" in value_lower or "cc" in value_lower:
                normalized[key_lower + "_l"] = cm3_to_liters(value)
            elif "s" in value_lower or "sek" in key_lower:
                normalized[key_lower + "_s"] = parse_seconds(value)
            else:
                normalized[key_lower] = value
        except Exception:
            normalized[key_lower] = value
    return normalized
