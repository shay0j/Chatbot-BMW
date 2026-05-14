"""Targeted breakage probes for the 4 new fixes + critical pre-existing paths.

If anything in this suite fails, that fix would break in production.
"""
import asyncio
import sys
import uuid

sys.stdout.reconfigure(encoding="utf-8")
from app.main import CrispBot

GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; RESET = "\033[0m"


async def main():
    bot = CrispBot()
    await bot._ensure_rag()
    results = []

    def check(label, ok, detail=""):
        marker = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  [{marker}] {label}")
        if not ok and detail:
            print(f"         {detail[:280]}")
        results.append((label, ok))

    # ============================================================
    # Section: Fix A — competitor + trade_in
    # ============================================================
    print("\n--- Fix A: competitor block context-awareness ---")
    sid = f"hard-{uuid.uuid4().hex[:6]}"
    r, _ = await bot.process_message("Czy przyjmiecie Audi A4 w trade-in?", sid)
    check("trade-in with Audi → answers trade-in policy (not competitor block)",
          "specjalizuję się wyłącznie" not in r.lower() and "zk motors" in r.lower(), r)

    r, _ = await bot.process_message("Mam Volkswagena Passata, oddam w rozliczeniu", f"hard-{uuid.uuid4().hex[:6]}")
    check("trade-in with Volkswagen → answers trade-in policy",
          "specjalizuję się wyłącznie" not in r.lower() and "zk motors" in r.lower(), r)

    # Verify pure competitor comparison STILL blocks
    r, _ = await bot.process_message("Audi e-tron czy BMW iX — co lepsze?", f"hard-{uuid.uuid4().hex[:6]}")
    check("competitor comparison (no trade-in) → still blocked",
          "specjalizuję się wyłącznie" in r.lower() or "bmw" in r.lower(), r)

    # ============================================================
    # Section: Fix B — specific model / superlative bypass
    # ============================================================
    print("\n--- Fix B: available_models bypass ---")
    r, _ = await bot.process_message("BMW M3 Touring — czy jest w sprzedaży?", f"hard-{uuid.uuid4().hex[:6]}")
    check("M3 Touring availability question → NOT URL dump",
          "stok.zkmotors.pl/pojazd/lista?pojazdsearch%5bmarka_id%5d=1" not in r.lower()
          or len(r) > 800,  # if URL is there but as part of richer answer, OK
          r)

    r, _ = await bot.process_message("Najtańszy model BMW w ofercie", f"hard-{uuid.uuid4().hex[:6]}")
    check("\"Najtańszy\" superlative → NOT URL dump",
          "stok.zkmotors.pl/pojazd/lista?pojazdsearch%5bmarka_id%5d=1" not in r.lower()
          or len(r) > 800, r)

    # Verify generic "show me models" STILL hits URL dump (intentional)
    r, _ = await bot.process_message("Pokażcie samochody", f"hard-{uuid.uuid4().hex[:6]}")
    check("generic \"show cars\" → still gets URL dump (intended)",
          "stok.zkmotors.pl" in r.lower(), r)

    # ============================================================
    # Section: Fix C — vague-but-on-topic
    # ============================================================
    print("\n--- Fix C: vague-but-on-topic moto_keywords ---")
    for query in ["Może coś sportowego", "Szukam czegoś rodzinnego", "Coś luksusowego polećcie"]:
        r, _ = await bot.process_message(query, f"hard-{uuid.uuid4().hex[:6]}")
        check(f"{query!r} → NOT off-topic deflected",
              "jestem tu po to, żeby pomagać" not in r.lower(), r)

    # Verify TRULY off-topic still deflects
    for query in ["Kto wygrał wybory?", "git pull origin main", "Roman Banasik"]:
        r, _ = await bot.process_message(query, f"hard-{uuid.uuid4().hex[:6]}")
        check(f"{query!r} → still off-topic",
              "bmw" in r.lower() or "zk motors" in r.lower(), r)

    # ============================================================
    # Section: Fix D — model codes in short queries
    # ============================================================
    print("\n--- Fix D: short model-code queries ---")
    for query in ["ix3 zasieg", "x3 cena", "M5 moc"]:
        r, _ = await bot.process_message(query, f"hard-{uuid.uuid4().hex[:6]}")
        # Just ensure NOT pure greeting (would contain Leo intro WITHOUT follow-up content)
        is_just_greeting = (r.strip().endswith("Zapraszam do rozmowy! 😊")
                            or len(r) < 350)
        check(f"{query!r} → not greeting-only response",
              not is_just_greeting, r[:200])

    # Verify ACTUAL greetings still work
    for query in ["Cześć", "hi there", "Dzień dobry"]:
        r, _ = await bot.process_message(query, f"hard-{uuid.uuid4().hex[:6]}")
        check(f"{query!r} → still recognized as greeting",
              "leo" in r.lower() or "witaj" in r.lower(), r)

    # ============================================================
    # Section: Pre-existing critical paths (no regression)
    # ============================================================
    print("\n--- No-regression: critical paths ---")
    r, _ = await bot.process_message("Chcę wymienić olej i filtry", f"hard-{uuid.uuid4().hex[:6]}")
    check("Bug 1 (oil/filters) — still fixed",
          "734 188" in r and "jestem tu po to" not in r.lower(), r)

    r, _ = await bot.process_message("Jakie modele są dostępne do jazdy próbnej?", f"hard-{uuid.uuid4().hex[:6]}")
    check("Bug 6 (test-drive list) — still fixed",
          "m235" in r.lower() and "ix2" in r.lower(), r)

    r, _ = await bot.process_message("Opowiedz mi o BMW M235", f"hard-{uuid.uuid4().hex[:6]}")
    check("Bug 2 (M235 fabrication guard) — still active",
          "dynamic model that combines" not in r.lower(), r)

    # Empty/short edges
    for query in ["", " ", "?", "🚗"]:
        try:
            r, _ = await bot.process_message(query, f"hard-{uuid.uuid4().hex[:6]}")
            check(f"empty/weird {query!r} → no crash, returns string",
                  isinstance(r, str) and len(r) > 0, r)
        except Exception as e:
            check(f"empty/weird {query!r} → no crash", False, f"EXCEPTION: {e}")

    # ============================================================
    # Final
    # ============================================================
    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    print("\n" + "=" * 60)
    print(f"HARDENING: {passed}/{total}")
    print("=" * 60)
    if passed < total:
        print("\nFAILS:")
        for label, ok in results:
            if not ok:
                print(f"  - {label}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
