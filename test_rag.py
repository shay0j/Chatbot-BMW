"""Test script for Round 2 client issues."""
import httpx
import json
import time

BASE = "http://localhost:8000"

# Each test maps to a client issue
tests = [
    # Issue #1: Off-topic gibberish (should NOT get BMW info)
    {"q": "git pull origin main", "issue": "#1 Off-topic gibberish", "expect_not": ["BMW X4", "BMW X3", "BMW iX"], "expect": ["BMW i ZK Motors"]},
    
    # Issue #3: Motorcycle repair (should get repair info, NOT motorcycle links)
    {"q": "Czy naprawiacie motocykle?", "issue": "#3 Motorcycle repair", "expect_not": ["stok.zkmotors.pl"]},
    
    # Issue #4: Competitor comparison (BIGGEST - should NEVER discuss other brands)
    {"q": "A jak BMW M3 wypada w porównaniu z Alfa Romeo?", "issue": "#4 Competitor brand", "expect_not": ["Alfa Romeo", "Giulia", "510 HP", "510 KM"]},
    {"q": "Czy nie lepiej kupić mercedesa?", "issue": "#4 Competitor brand 2", "expect_not": ["Mercedes", "AMG"]},
    
    # Issue #5: Random names (should be off-topic — NOT BMW specs)
    {"q": "Paweł Piwpwarczyk", "issue": "#5 Random name", "expect_not": ["BMW iX", "BMW M5", "wycen"], "expect": ["BMW i ZK Motors"]},
    {"q": "Roman Banasik", "issue": "#5 Random name 2", "expect_not": ["BMW iX", "trade-in"], "expect": ["BMW i ZK Motors"]},
    
    # Issue #6: Service contact info (should include phone numbers)
    {"q": "O jakich godzinach przyjmujecie auta po stłuczkach?", "issue": "#6 Service hours", "expect": ["734 188"]},
    
    # Issue #7: Leasing obsession (connect request should NOT return leasing essay)
    {"q": "Połącz mnie z Krzysztofem Dudzikiem", "issue": "#7 Connect request", "expect_not": ["BMW iX", "trade-in", "wycen"], "expect": ["BMW i ZK Motors"]},
    
    # Regression: Model specs should still work
    {"q": "opowiedz mi o BMW X3", "issue": "Regression: X3 specs", "expect": ["KM", "SUV"]},
]

print("=" * 70)
print("BMW CHATBOT - ROUND 2 TEST SUITE")
print("=" * 70)

passed = 0
failed = 0

for i, test in enumerate(tests):
    # Use unique session per test to avoid context cross-contamination
    q = test["q"]
    
    print(f"\n{'─' * 70}")
    print(f"TEST {i+1}/{len(tests)}: {test['issue']}")
    print(f"  Query: \"{q}\"")
    
    try:
        r = httpx.get(f"{BASE}/test/query", params={"q": q}, timeout=60)
        data = r.json()
        response = data.get("response", "NO RESPONSE")
        
        # Check for expected content
        ok = True
        
        if "expect_not" in test:
            for bad in test["expect_not"]:
                if bad.lower() in response.lower():
                    print(f"  ❌ FAIL: Response contains '{bad}' (should NOT)")
                    ok = False
        
        if "expect" in test:
            for good in test["expect"]:
                if good.lower() not in response.lower():
                    print(f"  ❌ FAIL: Response missing '{good}' (should contain)")
                    ok = False
        
        if ok:
            print(f"  ✅ PASS")
            passed += 1
        else:
            failed += 1
        
        # Show truncated response
        print(f"  Response: {response[:200]}...")
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        failed += 1
    
    time.sleep(1)  # small delay between tests

print(f"\n{'=' * 70}")
print(f"RESULTS: {passed}/{len(tests)} passed, {failed} failed")
print(f"{'=' * 70}")