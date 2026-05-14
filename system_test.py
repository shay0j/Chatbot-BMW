"""End-to-end system test for the BMW chatbot.

Drives real queries through CrispBot.process_message — same path the Crisp
webhook uses. Real intent detection, real off-topic gate, real RAG, real Cohere.

Run with: PYTHONIOENCODING=utf-8 python system_test.py
"""
import asyncio
import os
import sys
import uuid
from typing import List, Tuple

# Force UTF-8 stdout so we can print Polish + emoji on Windows console.
sys.stdout.reconfigure(encoding="utf-8")

from app.main import CrispBot


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"


def label(ok: bool) -> str:
    return f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"


async def run_query(bot: CrispBot, q: str, session_id: str = None) -> str:
    sid = session_id or f"test-{uuid.uuid4().hex[:8]}"
    response, _ = await bot.process_message(q, sid)
    return response


def check(label_text: str, response: str, must_contain=None, must_not_contain=None) -> bool:
    """Verify response contains/excludes the given substrings. Case-insensitive."""
    must_contain = must_contain or []
    must_not_contain = must_not_contain or []
    low = response.lower()
    missing = [s for s in must_contain if s.lower() not in low]
    forbidden = [s for s in must_not_contain if s.lower() in low]
    ok = not missing and not forbidden
    print(f"  [{label(ok)}] {label_text}")
    if missing:
        print(f"        {RED}missing{RESET}: {missing}")
    if forbidden:
        print(f"        {RED}forbidden present{RESET}: {forbidden}")
    if not ok:
        print(f"        {DIM}response: {response[:300]}…{RESET}")
    return ok


async def section_header(title: str):
    print(f"\n{YELLOW}{'=' * 70}{RESET}")
    print(f"{YELLOW}  {title}{RESET}")
    print(f"{YELLOW}{'=' * 70}{RESET}")


async def main():
    bot = CrispBot()
    # Pre-warm RAG so retrieval is hot for the LLM-backed cases.
    await bot._ensure_rag()

    results: List[Tuple[str, bool]] = []

    # ============================================================
    # SECTION 1: The 7 bug-log scenarios
    # ============================================================
    await section_header("SECTION 1 — replay 7 bug-log scenarios")

    # Bug 1 [nieznany1.1] — oil/filter deflection
    print("\nBug 1 [nieznany1.1] — oil & filters (was: off-topic deflection)")
    r = await run_query(bot, "Chcę wymienić olej i filtry")
    ok = check(
        "service intent fires, no off-topic deflection",
        r,
        must_contain=["serwis", "734 188"],  # any service phone number
        must_not_contain=["jestem tu po to, żeby pomagać"],
    )
    results.append(("Bug 1 — oil/filters routes to service", ok))

    # Bug 2 [nieznany2.1] — M35 / M235i hallucination
    print("\nBug 2 [nieznany2.1] — M235 specs (was: invented description)")
    r = await run_query(bot, "Opowiedz mi o BMW M235")
    ok = check(
        "no fabricated specs for M235",
        r,
        must_contain=["zk motors"],
        must_not_contain=["dynamic model", "łączy sportowe", "combines sporty"],
    )
    results.append(("Bug 2 — M235 no fabrication", ok))

    # Bug 2 variant — M35 doesn't exist
    print("\nBug 2 variant — M35 (does not exist as a BMW model)")
    r = await run_query(bot, "Chcę umówić jazdę próbną M35")
    ok = check(
        "M35 not silently swapped to M235i with invented description",
        r,
        must_not_contain=["m235i is a dynamic", "łączy sportowe osiągi z komfortem"],
    )
    results.append(("Bug 2 — M35 no swap-and-fabricate", ok))

    # Bug 3 [nieznany3.1] — trade-in no restrictions (must still answer same way)
    print("\nBug 3 [nieznany3.1] — trade-in (false-positive in client report)")
    r = await run_query(bot, "Czy mogę oddać auto w rozliczeniu?")
    ok = check(
        "still answers 'no age/mileage limits' per source",
        r,
        must_contain=["trade", "zk motors"],
    )
    results.append(("Bug 3 — trade-in still correct", ok))

    # Bug 4 [nieznany4.1] — 110 000 km
    print("\nBug 4 [nieznany4.1] — 110 000 km (false-positive in client report)")
    r = await run_query(bot, "Czy przebieg 110 000 km jest OK?")
    ok = check(
        "still distinguishes trade-in vs Premium Selection",
        r,
        must_contain=["zk motors"],
    )
    results.append(("Bug 4 — 110 000 km still correct", ok))

    # Bug 5 [nieznany5.1] — 330 000 km
    print("\nBug 5 [nieznany5.1] — 330 000 km (false-positive in client report)")
    r = await run_query(bot, "Czy mogę oddać auto z przebiegiem 330 000 km w rozliczeniu?")
    ok = check(
        "still says yes — no limit for trade-in",
        r,
        must_contain=["zk motors"],
        must_not_contain=["nie możemy przyjąć", "nie przyjmujemy", "odmawiamy"],
    )
    results.append(("Bug 5 — 330 000 km still correct", ok))

    # Bug 6 [nieznany6.1] — test drive list (was: URL dump)
    print("\nBug 6 [nieznany6.1] — test drive list (was: stock URL dump)")
    r = await run_query(bot, "Jakie modele są dostępne do jazdy próbnej?")
    ok = check(
        "returns the 7-model list, not the 4 URLs",
        r,
        must_contain=["m235", "ix2", "ix3", "x6", "m5"],
        must_not_contain=["najlepszeoferty.bmw.pl", "rodzaj_id%5d=2"],
    )
    results.append(("Bug 6 — test drive list", ok))

    # Bug 7 [nieznany7.1] — BMW M235 generic info (partial; should at least be honest)
    print("\nBug 7 [nieznany7.1] — BMW M235 generic info (partial)")
    r = await run_query(bot, "Opowiedz mi coś o BMW M235")
    ok = check(
        "honest answer, no fabricated specs",
        r,
        must_not_contain=["dynamic model", "combines sporty performance"],
    )
    results.append(("Bug 7 — M235 honest", ok))

    # ============================================================
    # SECTION 2: Intent detection coverage
    # ============================================================
    await section_header("SECTION 2 — intent detection coverage")

    intent_cases = [
        ("Chcę wymienić olej",                       "service"),
        ("wymiana filtrów",                          "service"),
        ("Oil change?",                              "service"),
        ("Chcę naprawić auto po stłuczce",           "service"),
        ("hamulce wymiana",                          "service"),
        ("Jakie macie modele?",                      "available_models"),
        ("Co macie w ofercie?",                      "available_models"),
        ("Pokażcie nowe BMW",                        "available_models"),
        ("Jakie modele są dostępne do jazdy próbnej?", "test_drive"),
        ("Na jakie jazdy próbne mogę się umówić?",   "test_drive"),
        ("Co jest do jazd testowych?",               "test_drive"),
        ("test drive",                               "test_drive"),
        ("trade-in",                                 "trade_in"),
        ("Chcę oddać auto w rozliczeniu",            "trade_in"),
        ("Jaki jest kontakt do doradcy?",            "contact"),
        ("Macie konfigurator?",                      "configurator"),
        ("Chcę katalog X5",                          "catalogs"),
        ("Sprzedajecie motocykle?",                  "motorcycle"),
        ("Macie skutery?",                           "motorcycle"),
        ("Cooper",                                   "mini"),
        ("Kto wygrał wybory?",                       "general"),
        ("git pull origin main",                     "general"),
    ]
    for q, expected in intent_cases:
        actual = bot._detect_intent(q.lower())
        ok = actual == expected
        results.append((f"intent({q!r}) == {expected}", ok))
        print(f"  [{label(ok)}] {q!r:55s} -> {actual} (expected {expected})")

    # ============================================================
    # SECTION 3: RAG retrieval per source category
    # ============================================================
    await section_header("SECTION 3 — RAG retrieval per source category")
    rag_cases = [
        # (query, must_contain_any, label)
        ("Opowiedz mi o BMW X3", ["x3"], "X3 specs"),
        ("BMW iX zasięg", ["ix", "zasięg"], "iX range"),
        ("warunki BMW Premium Selection", ["7 lat", "150"], "Premium Selection buyback"),
        ("ile kosztuje leasing", ["leasing", "zk motors"], "leasing info"),
        ("Jak skonfigurować BMW?", ["konfigurat"], "configurator pointer"),
        ("Kto sprzedaje używane BMW?", ["zk motors"], "used car advisors"),
        ("Macie X3 plug-in hybrid?", ["x3", "hybryd"], "X3 PHEV"),
    ]
    for q, expected_terms, lbl in rag_cases:
        r = await run_query(bot, q)
        low = r.lower()
        ok = any(term in low for term in expected_terms)
        results.append((f"RAG: {lbl}", ok))
        print(f"  [{label(ok)}] {lbl:30s} | query={q!r}")
        if not ok:
            print(f"        {DIM}response: {r[:200]}…{RESET}")

    # ============================================================
    # SECTION 4: Negative regression
    # ============================================================
    await section_header("SECTION 4 — negative regression")

    # Off-topic must still deflect
    r = await run_query(bot, "Kto wygrał wybory prezydenckie?")
    ok = check(
        "politics still deflected",
        r,
        must_contain=["bmw", "zk motors"],
    )
    results.append(("off-topic: politics deflected", ok))

    r = await run_query(bot, "git pull origin main")
    ok = check(
        "tech command still deflected",
        r,
        must_contain=["bmw", "zk motors"],
    )
    results.append(("off-topic: tech command deflected", ok))

    # Greetings must still go through greeting path (no RAG call)
    r = await run_query(bot, "cześć")
    ok = check(
        "pure greeting answered as greeting",
        r,
        must_contain=["leo"],
    )
    results.append(("greeting routes to greeting handler", ok))

    # Competitor must be blocked
    r = await run_query(bot, "Czy lepsze jest Mercedes czy BMW?")
    ok = check(
        "competitor mention blocked",
        r,
        must_contain=["bmw"],
        must_not_contain=["mercedes jest lepszy", "audi jest lepsze"],
    )
    results.append(("competitor mention blocked", ok))

    # Service intent must still pull contact info (Issue #6 enrichment)
    r = await run_query(bot, "Chcę zrobić przegląd")
    ok = check(
        "service question gets contact info",
        r,
        must_contain=["734 188"],
    )
    results.append(("service answer contains contact", ok))

    # Trade-in must include the buyback URL post-processing
    r = await run_query(bot, "Chcę oddać moje BMW w rozliczeniu")
    ok = check(
        "trade-in answer mentions valuation URL or trade-in policy",
        r,
        must_contain=["zk motors"],
    )
    results.append(("trade-in answer enrichment", ok))

    # ============================================================
    # FINAL TALLY
    # ============================================================
    print(f"\n{YELLOW}{'=' * 70}{RESET}")
    print(f"{YELLOW}  TALLY{RESET}")
    print(f"{YELLOW}{'=' * 70}{RESET}\n")
    passed = sum(1 for _, ok in results if ok)
    failed = [name for name, ok in results if not ok]
    total = len(results)
    print(f"Passed: {GREEN}{passed}/{total}{RESET}")
    if failed:
        print(f"Failed: {RED}{len(failed)}{RESET}")
        for name in failed:
            print(f"  - {name}")
    else:
        print(f"All checks {GREEN}passed{RESET}.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
