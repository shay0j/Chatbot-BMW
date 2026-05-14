"""Same graded queries as diverse_live_graded.py but runs against LOCAL bot
with my fixes (calls process_message directly — no HTTP, no ngrok).

Uses an isolated session_id per query so context bleed is avoided.
"""
import asyncio
import sys
import time
import uuid
from typing import List, Tuple

sys.stdout.reconfigure(encoding="utf-8")

from app.main import CrispBot
from diverse_live_graded import TESTS, grade  # reuse test cases + grader

GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; RESET = "\033[0m"


async def main():
    bot = CrispBot()
    await bot._ensure_rag()

    rows = []
    by_cat = {}
    for i, (cat, q, must, must_not, note) in enumerate(TESTS, 1):
        t0 = time.time()
        try:
            sid = f"local-{uuid.uuid4().hex[:8]}"  # fresh session each query
            r, _ = await bot.process_message(q, sid)
        except Exception as e:
            r = f"<<EXC: {e}>>"
        dt = time.time() - t0
        g = grade(r, must, must_not)
        marker = {"PASS": "✅", "PARTIAL": "🟡", "FAIL": "❌"}[g]
        print(f"[{i:2d}/{len(TESTS)}] {marker} {g:7s} ({dt:5.1f}s) {cat:20s} {q[:60]}")
        if g != "PASS":
            if note:
                print(f"        note: {note}")
            print(f"        resp: {r[:250]!r}")
        rows.append((cat, q, g, dt, r, note))
        by_cat.setdefault(cat, []).append(g)

    print("\n" + "=" * 78); print("CATEGORY SUMMARY"); print("=" * 78)
    for cat, grades in by_cat.items():
        p = grades.count("PASS"); pa = grades.count("PARTIAL"); f = grades.count("FAIL")
        print(f"  {cat:25s}  PASS {p}/{len(grades)}  PARTIAL {pa}  FAIL {f}")
    total = len(rows)
    p = sum(1 for r in rows if r[2] == "PASS")
    pa = sum(1 for r in rows if r[2] == "PARTIAL")
    f = sum(1 for r in rows if r[2] == "FAIL")
    print("\n" + "=" * 78)
    print(f"OVERALL: PASS {p}/{total}   PARTIAL {pa}   FAIL {f}")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
