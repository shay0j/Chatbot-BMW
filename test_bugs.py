"""
Test script for all 7 bugs from bugs_extracted.txt
Run: python test_bugs.py

Tests the bot's process_message() directly — no server needed.
Each test prints the query, response, and a PASS/FAIL check.
"""
import asyncio
import sys
import os

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, ".")


# ── Helpers ──────────────────────────────────────────────────

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def check(label: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"  {status} {label}" + (f" -- {detail}" if detail else ""))
    return condition


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── Main test suite ──────────────────────────────────────────

async def run_tests():
    # Import bot after path is set
    from app.main import CrispBot, get_rag_service, RAG_AVAILABLE

    bot = CrispBot()

    # Make sure RAG is loaded
    print("Initializing RAG service...")
    rag = await get_rag_service()
    bot.rag_service = rag
    health = await rag.health_check()
    print(f"RAG status: {health.get('status')} | docs: {(await rag.get_stats()).get('documents_in_store', '?')}")
    print(f"RAG available: {RAG_AVAILABLE}")

    results = {"pass": 0, "fail": 0}

    def track(passed: bool):
        results["pass" if passed else "fail"] += 1

    # ────────────────────────────────────────────────────────
    # BUG #1: Wrong link (buyback instead of stock)
    # ────────────────────────────────────────────────────────
    separator("BUG #1: Wrong link (buyback vs stock)")

    response, _ = await bot.process_message(
        "Will I get a link where I can see all the cars available for sale?",
        "test_bug1"
    )
    print(f"  Q: Will I get a link where I can see all the cars available for sale?")
    print(f"  A: {response[:300]}...")

    p = check("Contains stock link (stok.zkmotors.pl)", "stok.zkmotors.pl" in response)
    track(p)
    p = check("Does NOT contain buyback link (bmw.pl/pl/odkup)", "odkup" not in response)
    track(p)

    # Also test Polish variant
    response2, _ = await bot.process_message(
        "Chce zobaczyc jakie samochody macie w ofercie",
        "test_bug1b"
    )
    print(f"\n  Q: Chce zobaczyc jakie samochody macie w ofercie")
    print(f"  A: {response2[:300]}...")

    p = check("Contains stock link", "stok.zkmotors.pl" in response2)
    track(p)

    # ────────────────────────────────────────────────────────
    # BUG #2: Double response / scooter query
    # ────────────────────────────────────────────────────────
    separator("BUG #2: Scooter query")

    response, _ = await bot.process_message(
        "Czy skutery sa w ofercie?",
        "test_bug2"
    )
    print(f"  Q: Czy skutery sa w ofercie?")
    print(f"  A: {response[:400]}...")

    p = check("Mentions 'nie ma skuterow' or similar", any(kw in response.lower() for kw in ["nie ma skuter", "nie ma skuterow", "brak skuter"]))
    track(p)
    p = check("Single coherent response (no self-correction)", response.count("https://") <= 3)
    track(p)

    # ────────────────────────────────────────────────────────
    # BUG #3: Price filtering (208k+ for <150k budget)
    # ────────────────────────────────────────────────────────
    separator("BUG #3: Price filtering")

    response, _ = await bot.process_message(
        "Jakie modele sa dostepne do 150 000 PLN?",
        "test_bug3"
    )
    print(f"  Q: Jakie modele sa dostepne do 150 000 PLN?")
    print(f"  A: {response[:500]}...")

    p = check("Does NOT suggest 208,500 PLN car as fitting budget", "208" not in response.split("150")[0] if "150" in response else "208 500" not in response)
    track(p)
    p = check("Mentions used BMW or budget limitation", any(kw in response.lower() for kw in ["uzyw", "premium selection", "budzet", "budżet", "najlepszeoferty", "nie ma modelu", "moze nie byc"]))
    track(p)

    # ────────────────────────────────────────────────────────
    # BUG #4: 4 Series GC door count
    # ────────────────────────────────────────────────────────
    separator("BUG #4: 4 Series GC doors")

    response, _ = await bot.process_message(
        "Ile drzwi ma BMW Seria 4 Gran Coupe?",
        "test_bug4"
    )
    print(f"  Q: Ile drzwi ma BMW Seria 4 Gran Coupe?")
    print(f"  A: {response[:400]}...")

    p = check("Mentions 5 doors", "5" in response and "drzwi" in response.lower())
    track(p)
    p = check("Does NOT say 3 doors", "3 drzwi" not in response.lower())
    track(p)

    # ────────────────────────────────────────────────────────
    # BUG #5: Chinese characters
    # ────────────────────────────────────────────────────────
    separator("BUG #5: Chinese characters")

    response, _ = await bot.process_message(
        "Jakie kolory lakieru sa dostepne dla BMW M5?",
        "test_bug5"
    )
    print(f"  Q: Jakie kolory lakieru sa dostepne dla BMW M5?")
    print(f"  A: {response[:400]}...")

    import re
    cjk_pattern = re.compile(r'[\u2e80-\u9fff\u3000-\u303f\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]')
    has_cjk = bool(cjk_pattern.search(response))
    p = check("No CJK characters in response", not has_cjk,
              f"Found CJK: {cjk_pattern.findall(response)}" if has_cjk else "Clean")
    track(p)

    # ────────────────────────────────────────────────────────
    # BUG #6: Missing stock link when listing available cars
    # ────────────────────────────────────────────────────────
    separator("BUG #6: Stock link in 'available cars' response")

    response, _ = await bot.process_message(
        "Gdzie moge zobaczyc samochody dostepne od reki?",
        "test_bug6"
    )
    print(f"  Q: Gdzie moge zobaczyc samochody dostepne od reki?")
    print(f"  A: {response[:400]}...")

    p = check("Contains stock link", "stok.zkmotors.pl" in response or "najlepszeoferty" in response)
    track(p)

    # ────────────────────────────────────────────────────────
    # BUG #7: Trade-in context loss (BMW XM specs instead)
    # ────────────────────────────────────────────────────────
    separator("BUG #7: Trade-in context")

    # Simulate 2-turn conversation
    session = "test_bug7"
    await bot.process_message(
        "Czy moge sprzedac lub zostawic swoje auto w rozliczeniu u was?",
        session
    )
    response, _ = await bot.process_message(
        "Jakie warunki musi spelniac ten samochod?",
        session
    )
    print(f"  Q1: Czy moge sprzedac lub zostawic swoje auto w rozliczeniu u was?")
    print(f"  Q2: Jakie warunki musi spelniac ten samochod?")
    print(f"  A2: {response[:500]}...")

    p = check("Response is about trade-in, NOT BMW XM specs", "xm" not in response.lower() or "odkup" in response.lower() or "trade" in response.lower() or "rozliczeni" in response.lower())
    track(p)
    p = check("Mentions odkup/trade-in/wymiana", any(kw in response.lower() for kw in ["odkup", "trade", "wymian", "rozliczeni", "wycen", "dowolnej marki"]))
    track(p)

    # ────────────────────────────────────────────────────────
    # SUMMARY
    # ────────────────────────────────────────────────────────
    separator("SUMMARY")
    total = results["pass"] + results["fail"]
    print(f"  {PASS}: {results['pass']}/{total}")
    print(f"  {FAIL}: {results['fail']}/{total}")
    if results["fail"] == 0:
        print(f"\n  ALL TESTS PASSED!")
    else:
        print(f"\n  {results['fail']} test(s) need attention.")
    print()


if __name__ == "__main__":
    asyncio.run(run_tests())
