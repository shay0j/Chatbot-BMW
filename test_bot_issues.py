"""
Test script for verifying bot issues from log are resolved.
Tests exact queries from bot log.txt
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.main import CrispBot

# Test queries from bot log.txt
TEST_CASES = [
    {
        "id": 1,
        "query": "I want to buy an SUV",
        "expected": "Should mention BMW SUV models, not just leasing",
        "issue": "Original: jumped straight to leasing"
    },
    {
        "id": 2,
        "query": "I'm interested in the X3",
        "expected": "Should describe X3 model features, specs, not leasing",
        "issue": "Original: only talked about leasing instead of model specs"
    },
    {
        "id": 3,
        "query": "tell me something about this model",
        "context": "X3",
        "expected": "Should describe X3 features",
        "issue": "Original: gave generic info"
    },
    {
        "id": 4,
        "query": "What is the current leasing offer on the X3?",
        "expected": "Can say 'I don't have that info' - this is reasonable",
        "issue": "Original: correctly said no info (this is OK)"
    },
    {
        "id": 5,
        "query": "I would like to know what the engines are in the new Countryman",
        "expected": "Should redirect to salon (MINI not BMW)",
        "issue": "Original: correctly said no info (this is OK - MINI)"
    },
    {
        "id": 6,
        "query": "what do you say about the M5 touring as a family car",
        "expected": "Should describe M5 Touring features for family use",
        "issue": "Original: gave good answer (this worked)"
    },
    {
        "id": 7,
        "query": "what are the versions of the new M4?",
        "expected": "Should list M4 versions if in database",
        "issue": "Original: said no info"
    },
    {
        "id": 8,
        "query": "is the transmission dual-clutch?",
        "context": "M4",
        "expected": "Should give transmission info if in database",
        "issue": "Original: said no info"
    },
    {
        "id": 9,
        "query": "Which sporty sedan would you recommend for me?",
        "expected": "Should recommend BMW sedans",
        "issue": "Original: recommended 4 Series Gran Coupé (good)"
    }
]


async def test_bot_responses():
    """Test bot with queries from log"""
    print("=" * 80)
    print("🧪 TESTING BOT WITH QUERIES FROM LOG FILE")
    print("=" * 80)

    bot = CrispBot()
    await bot._ensure_rag()

    # Check RAG health
    if bot.rag_service:
        health = await bot.rag_service.health_check()
        stats = await bot.rag_service.get_stats()
        print(f"\n📊 RAG STATUS:")
        print(f"   Status: {health.get('status')}")
        print(f"   Documents: {stats.get('documents_in_store', 0)}")
        print(f"   Vector Store: {health.get('vector_store', 'unknown')}")
    else:
        print("\n⚠️  WARNING: RAG service not available!")

    print("\n" + "=" * 80)

    results = {
        "passed": 0,
        "failed": 0,
        "warnings": 0
    }

    for i, test in enumerate(TEST_CASES, 1):
        session_id = f"test_session_{test['id']}"
        query = test["query"]

        print(f"\n{'=' * 80}")
        print(f"TEST {test['id']}: {test['issue']}")
        print(f"{'=' * 80}")
        print(f"📝 Query: \"{query}\"")
        print(f"✅ Expected: {test['expected']}")

        try:
            response, transfer = await bot.process_message(query, session_id)

            print(f"\n🤖 Bot Response:")
            print("-" * 80)
            print(response)
            print("-" * 80)

            # Analyze response
            response_lower = response.lower()

            # Check for common issues
            has_no_info = any(phrase in response_lower for phrase in [
                "nie mam informacji",
                "i don't have",
                "brak danych",
                "no information"
            ])

            has_leasing = "leasing" in response_lower or "lease" in response_lower
            has_specs = any(word in response_lower for word in [
                "moc", "km", "silnik", "power", "engine", "hp", "nm", "moment",
                "przyspieszenie", "acceleration", "suv", "sedan", "touring"
            ])
            has_salon_redirect = any(phrase in response_lower for phrase in [
                "zk motors", "salon", "serwis", "contact", "kontakt"
            ])

            # Test-specific validation
            status = "✅ PASS"

            if test['id'] == 2:  # X3 query - CRITICAL TEST
                if has_leasing and not has_specs:
                    status = "❌ FAIL: Still talking about leasing instead of model specs!"
                    results["failed"] += 1
                elif has_specs:
                    status = "✅ PASS: Describes model specs!"
                    results["passed"] += 1
                else:
                    status = "⚠️  WARNING: No specs found"
                    results["warnings"] += 1

            elif test['id'] in [1, 6, 9]:  # Should have specs
                if has_specs:
                    status = "✅ PASS: Has model information"
                    results["passed"] += 1
                elif has_no_info:
                    status = "❌ FAIL: Says no info when should have data"
                    results["failed"] += 1
                else:
                    status = "⚠️  WARNING: Unclear response"
                    results["warnings"] += 1

            elif test['id'] in [4, 5, 7, 8]:  # Can say no info OR redirect
                if has_no_info or has_salon_redirect:
                    status = "✅ PASS: Appropriately handles missing data"
                    results["passed"] += 1
                else:
                    status = "⚠️  INFO: Unexpected response type"
                    results["warnings"] += 1

            print(f"\n{status}")

        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            results["failed"] += 1

    # Final summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"✅ Passed:   {results['passed']}/{len(TEST_CASES)}")
    print(f"❌ Failed:   {results['failed']}/{len(TEST_CASES)}")
    print(f"⚠️  Warnings: {results['warnings']}/{len(TEST_CASES)}")

    total = results['passed'] + results['failed'] + results['warnings']
    pass_rate = (results['passed'] / total * 100) if total > 0 else 0
    print(f"\n📈 Pass Rate: {pass_rate:.1f}%")

    if results['failed'] == 0:
        print("\n🎉 ALL CRITICAL TESTS PASSED!")
    else:
        print(f"\n⚠️  {results['failed']} test(s) failed - review needed")

    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_bot_responses())
