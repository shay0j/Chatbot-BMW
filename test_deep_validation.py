import asyncio
import sys
import os
import time

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, ".")

PASS = "✅ [PASS]"
FAIL = "❌ [FAIL]"

def separator(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def check(label: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"  {status} {label}" + (f" -- {detail}" if detail else ""))
    return condition

async def run_tests():
    from app.main import CrispBot, get_rag_service, RAG_AVAILABLE, _last_processed
    
    bot = CrispBot()
    rag = await get_rag_service()
    bot.rag_service = rag
    
    print("🚀 Starting Deep Multi-Dimensional Validation...")
    print(f"RAG available: {RAG_AVAILABLE}")
    
    results = {"pass": 0, "fail": 0}
    def track(passed: bool): results["pass" if passed else "fail"] += 1

    # ---------------------------------------------------------
    # DIMENSION 1: PROMPT ECHOING & CAPS LOCK (Bug 1 Fix)
    # ---------------------------------------------------------
    separator("DIMENSION 1: Prompt Echoing & Formatting")
    
    q1 = "Czy BMW X5 ma napęd na przednie koła?"
    ans1, _ = await bot.process_message(q1, "test_dim1")
    print(f"User: {q1}")
    print(f"Bot: {ans1[:250]}...")
    
    p = check("Does not start by repeating question", not ans1.strip().startswith("Czy BMW X5"))
    track(p)
    p = check("Does not use explicit uppercase 'NIE.'", "NIE." not in ans1 and "TAK." not in ans1)
    track(p)

    # ---------------------------------------------------------
    # DIMENSION 2: ELECTRIC VEHICLE OFF-TOPIC BYPASS (Bug 3 Fix)
    # ---------------------------------------------------------
    separator("DIMENSION 2: EV Off-topic Bypass")
    
    q2 = "A elektryczne?"
    ans2, _ = await bot.process_message(q2, "test_dim2")
    print(f"User: {q2}")
    print(f"Bot: {ans2[:250]}...")
    
    p = check("Did not trigger off-topic fallback", "Mogę pomóc z:" not in ans2)
    track(p)
    
    # Another test for hybrids
    q2b = "Macie jakieś hybrydy plug-in z dużym zasięgiem i mocnymi bateriami?"
    ans2b, _ = await bot.process_message(q2b, "test_dim2_b")
    p = check("Hybrid phrases bypass off-topic", "Mogę pomóc z:" not in ans2b)
    track(p)

    # ---------------------------------------------------------
    # DIMENSION 3: TRADE-IN RAG ESCALATION (Bug 2 Fix)
    # ---------------------------------------------------------
    separator("DIMENSION 3: Trade-in RAG Retrieval & Linking")
    
    q3 = "Chcę zostawić auto w rozliczeniu. Ile lat maksymalnie może mieć moje auto?"
    ans3, _ = await bot.process_message(q3, "test_dim3")
    print(f"User: {q3}")
    print(f"Bot: {ans3[:250]}...")
    
    # Must NOT be just the generic response! 
    p = check("Bot answered specifically instead of repeating the blanket trade-in string", "Wymagane dokumenty: dowód rejestracyjny" not in ans3[:60])
    track(p)
    
    p = check("Appended the online valuation link", "bmw.pl/pl/odkup/" in ans3.lower())
    track(p)

    # ---------------------------------------------------------
    # DIMENSION 4: THREAD CONCURRENCY & DEDUPLICATION (Bug 3 Fix)
    # ---------------------------------------------------------
    separator("DIMENSION 4: Burst Protection & Deduplication")
    
    # We simulate sending the exact same message twice at the exact same timestamp
    from app.main import is_duplicate
    session = "test_burst_session"
    msg = "Ile kosztuje nowa seria 3?"
    ts1 = int(time.time() * 1000)
    
    # First message should not be a duplicate
    dup1 = is_duplicate(session, msg, ts1)
    p = check("First message passes dedup filter freely", dup1 is False)
    track(p)
    
    # Second message with 15 seconds later (inside the 30s envelope)
    ts2 = ts1 + 15000
    dup2 = is_duplicate(session, msg, ts2)
    p = check("Second message within 30s locked out as duplicate", dup2 is True)
    track(p)
    
    # Clean up state to not mess up true runtime
    if f"{session}:{msg}" in _last_processed:
        del _last_processed[f"{session}:{msg}"]

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------
    separator("TESTING COMPLETE")
    tot = results["pass"] + results["fail"]
    print(f"Passed: {results['pass']}/{tot}")
    print(f"Failed: {results['fail']}/{tot}")
    
    if results["fail"] == 0:
        print("\n🏆 ALL DIMENSIONS FUNCTIONING PERFECTLY!")

if __name__ == "__main__":
    asyncio.run(run_tests())
