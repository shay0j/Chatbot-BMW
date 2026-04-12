"""
Comprehensive Bot Testing - Diverse Scenarios
Tests real-world conversations, edge cases, and multi-turn interactions
"""
import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.main import CrispBot


class BotTester:
    def __init__(self):
        self.bot = None
        self.results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "categories": {}
        }

    async def setup(self):
        """Initialize bot and RAG"""
        print("🔧 Initializing bot...")
        self.bot = CrispBot()
        await self.bot._ensure_rag()

        if self.bot.rag_service:
            health = await self.bot.rag_service.health_check()
            stats = await self.bot.rag_service.get_stats()
            print(f"✅ RAG Status: {health.get('status')}")
            print(f"✅ Documents: {stats.get('documents_in_store', 0)}")
        else:
            print("⚠️  WARNING: RAG not available!")

    def check_response(self, response: str, expectations: Dict[str, Any]) -> Dict[str, Any]:
        """Check if response meets expectations"""
        response_lower = response.lower()
        result = {
            "passed": True,
            "failures": [],
            "warnings": []
        }

        # Check for required keywords
        if "must_contain" in expectations:
            for keyword in expectations["must_contain"]:
                if keyword.lower() not in response_lower:
                    result["passed"] = False
                    result["failures"].append(f"Missing required: '{keyword}'")

        # Check for forbidden keywords
        if "must_not_contain" in expectations:
            for keyword in expectations["must_not_contain"]:
                if keyword.lower() in response_lower:
                    result["passed"] = False
                    result["failures"].append(f"Contains forbidden: '{keyword}'")

        # Check for any of these keywords
        if "should_contain_any" in expectations:
            found = False
            for keyword in expectations["should_contain_any"]:
                if keyword.lower() in response_lower:
                    found = True
                    break
            if not found:
                result["warnings"].append(f"Expected one of: {expectations['should_contain_any']}")

        # Check response length
        if "min_length" in expectations:
            if len(response) < expectations["min_length"]:
                result["passed"] = False
                result["failures"].append(f"Response too short: {len(response)} < {expectations['min_length']}")

        if "max_length" in expectations:
            if len(response) > expectations["max_length"]:
                result["warnings"].append(f"Response very long: {len(response)} > {expectations['max_length']}")

        # Check for specific patterns
        if "pattern_check" in expectations:
            check_func = expectations["pattern_check"]
            pattern_result = check_func(response)
            if not pattern_result:
                result["passed"] = False
                result["failures"].append("Failed pattern check")

        return result

    async def test_conversation(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Test a multi-turn conversation"""
        category = conversation.get("category", "general")
        session_id = f"test_{category}_{id(conversation)}"

        print(f"\n{'='*80}")
        print(f"🧪 TEST: {conversation['name']}")
        print(f"📁 Category: {category}")
        print(f"{'='*80}")

        all_passed = True
        turn_results = []

        for i, turn in enumerate(conversation["turns"], 1):
            query = turn["query"]
            expectations = turn.get("expectations", {})

            print(f"\n👤 Turn {i}: {query}")

            try:
                response, transfer = await self.bot.process_message(query, session_id)

                print(f"🤖 Response: {response[:300]}{'...' if len(response) > 300 else ''}")

                # Check expectations
                check_result = self.check_response(response, expectations)

                if check_result["passed"]:
                    print("   ✅ PASS")
                else:
                    print(f"   ❌ FAIL: {', '.join(check_result['failures'])}")
                    all_passed = False

                if check_result["warnings"]:
                    print(f"   ⚠️  {', '.join(check_result['warnings'])}")

                turn_results.append({
                    "turn": i,
                    "query": query,
                    "response": response,
                    "passed": check_result["passed"],
                    "failures": check_result["failures"],
                    "warnings": check_result["warnings"]
                })

            except Exception as e:
                print(f"   ❌ ERROR: {str(e)}")
                all_passed = False
                turn_results.append({
                    "turn": i,
                    "query": query,
                    "error": str(e),
                    "passed": False
                })

        return {
            "name": conversation["name"],
            "category": category,
            "passed": all_passed,
            "turns": turn_results
        }

    async def run_all_tests(self, test_suite: List[Dict[str, Any]]):
        """Run all tests and generate report"""
        print("\n" + "="*80)
        print("🚀 COMPREHENSIVE BOT TESTING")
        print("="*80)

        await self.setup()

        all_results = []

        for conversation in test_suite:
            result = await self.test_conversation(conversation)
            all_results.append(result)

            # Update stats
            self.results["total"] += 1
            category = result["category"]
            if category not in self.results["categories"]:
                self.results["categories"][category] = {"passed": 0, "failed": 0}

            if result["passed"]:
                self.results["passed"] += 1
                self.results["categories"][category]["passed"] += 1
            else:
                self.results["failed"] += 1
                self.results["categories"][category]["failed"] += 1

        # Print final report
        self.print_report(all_results)

    def print_report(self, all_results: List[Dict[str, Any]]):
        """Print final test report"""
        print("\n" + "="*80)
        print("📊 FINAL TEST REPORT")
        print("="*80)

        # Overall stats
        total = self.results["total"]
        passed = self.results["passed"]
        failed = self.results["failed"]
        pass_rate = (passed / total * 100) if total > 0 else 0

        print(f"\n📈 Overall Results:")
        print(f"   Total Tests: {total}")
        print(f"   ✅ Passed: {passed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   Pass Rate: {pass_rate:.1f}%")

        # Category breakdown
        print(f"\n📁 By Category:")
        for category, stats in self.results["categories"].items():
            cat_total = stats["passed"] + stats["failed"]
            cat_rate = (stats["passed"] / cat_total * 100) if cat_total > 0 else 0
            print(f"   {category:20s}: {stats['passed']}/{cat_total} ({cat_rate:.0f}%)")

        # Failed tests detail
        if failed > 0:
            print(f"\n❌ Failed Tests:")
            for result in all_results:
                if not result["passed"]:
                    print(f"\n   Test: {result['name']}")
                    for turn in result["turns"]:
                        if not turn.get("passed", False):
                            print(f"      Turn {turn['turn']}: {turn['query']}")
                            if "failures" in turn:
                                for failure in turn["failures"]:
                                    print(f"         - {failure}")
                            if "error" in turn:
                                print(f"         - Error: {turn['error']}")

        # Final verdict
        print("\n" + "="*80)
        if pass_rate >= 90:
            print("🎉 EXCELLENT! Bot is performing very well!")
        elif pass_rate >= 75:
            print("✅ GOOD! Bot is working well with minor issues")
        elif pass_rate >= 60:
            print("⚠️  FAIR! Some improvements needed")
        else:
            print("❌ NEEDS WORK! Multiple issues detected")
        print("="*80)


# ============================================
# TEST SUITE - DIVERSE SCENARIOS
# ============================================

TEST_SUITE = [
    # Category 1: Model Information Queries
    {
        "category": "model_info",
        "name": "Basic X3 Information",
        "turns": [
            {
                "query": "Tell me about BMW X3",
                "expectations": {
                    "must_contain": ["x3", "suv"],
                    "should_contain_any": ["moc", "km", "power", "silnik", "engine"],
                    "must_not_contain": ["nie mam", "i don't have", "brak"],
                    "min_length": 100
                }
            }
        ]
    },
    {
        "category": "model_info",
        "name": "X5 vs X7 Comparison",
        "turns": [
            {
                "query": "What's the difference between X5 and X7?",
                "expectations": {
                    "must_contain": ["x5", "x7"],
                    "should_contain_any": ["różnica", "difference", "większ", "mniejsz", "bigger", "smaller"],
                    "min_length": 150
                }
            }
        ]
    },
    {
        "category": "model_info",
        "name": "M Performance Models",
        "turns": [
            {
                "query": "Jakie masz modele M?",
                "expectations": {
                    "should_contain_any": ["m2", "m3", "m4", "m5", "m8"],
                    "must_not_contain": ["nie mam", "brak informacji"],
                    "min_length": 100
                }
            }
        ]
    },
    {
        "category": "model_info",
        "name": "Electric Models",
        "turns": [
            {
                "query": "Czy macie elektryczne BMW?",
                "expectations": {
                    "should_contain_any": ["i4", "i5", "i7", "ix", "elektryczny", "electric"],
                    "min_length": 80
                }
            }
        ]
    },

    # Category 2: Technical Specifications
    {
        "category": "technical",
        "name": "Engine Power Query",
        "turns": [
            {
                "query": "Ile mocy ma X5?",
                "expectations": {
                    "must_contain": ["x5"],
                    "should_contain_any": ["km", "moc", "power", "hp"],
                    "must_not_contain": ["nie wiem", "brak danych"],
                    "min_length": 80
                }
            }
        ]
    },
    {
        "category": "technical",
        "name": "Acceleration Performance",
        "turns": [
            {
                "query": "Jak szybko przyspiesza M3?",
                "expectations": {
                    "must_contain": ["m3"],
                    "should_contain_any": ["0-100", "przyspieszenie", "sekund", "s", "acceleration"],
                    "min_length": 60
                }
            }
        ]
    },
    {
        "category": "technical",
        "name": "Fuel Type Query",
        "turns": [
            {
                "query": "X3 ma silnik diesel czy benzynowy?",
                "expectations": {
                    "must_contain": ["x3"],
                    "should_contain_any": ["diesel", "benzyn", "petrol", "wariant", "variant", "oba", "both"],
                    "min_length": 60
                }
            }
        ]
    },

    # Category 3: Price and Sales
    {
        "category": "pricing",
        "name": "Price Information",
        "turns": [
            {
                "query": "Ile kosztuje BMW X3?",
                "expectations": {
                    "must_contain": ["x3"],
                    "should_contain_any": ["zł", "cena", "price", "kosztuje", "od"],
                    "min_length": 60
                }
            }
        ]
    },
    {
        "category": "pricing",
        "name": "Budget-Based Query",
        "turns": [
            {
                "query": "Jakie BMW mogę kupić za około 300 000 zł?",
                "expectations": {
                    "should_contain_any": ["seria", "series", "model"],
                    "min_length": 80
                }
            }
        ]
    },
    {
        "category": "pricing",
        "name": "Leasing Information",
        "turns": [
            {
                "query": "Jak działa leasing BMW?",
                "expectations": {
                    "must_contain": ["leasing"],
                    "should_contain_any": ["rata", "umowa", "koniec", "agreement", "monthly"],
                    "min_length": 100
                }
            }
        ]
    },

    # Category 4: Multi-turn Conversations
    {
        "category": "conversation",
        "name": "Multi-turn Model Inquiry",
        "turns": [
            {
                "query": "Szukam SUV-a",
                "expectations": {
                    "should_contain_any": ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "suv"],
                    "min_length": 100
                }
            },
            {
                "query": "Który jest największy?",
                "expectations": {
                    "must_contain": ["x7"],
                    "should_contain_any": ["największy", "biggest", "largest"],
                    "min_length": 50
                }
            },
            {
                "query": "Ile on kosztuje?",
                "expectations": {
                    "must_contain": ["x7"],
                    "should_contain_any": ["zł", "cena", "price"],
                    "min_length": 50
                }
            }
        ]
    },
    {
        "category": "conversation",
        "name": "Family Car Selection",
        "turns": [
            {
                "query": "Potrzebuję auto rodzinne",
                "expectations": {
                    "should_contain_any": ["rodzin", "family", "suv", "touring", "kombi"],
                    "min_length": 80
                }
            },
            {
                "query": "Jakie ma bagażnik?",
                "expectations": {
                    "should_contain_any": ["litr", "liter", "bagażnik", "cargo", "trunk"],
                    "min_length": 50
                }
            }
        ]
    },

    # Category 5: Edge Cases
    {
        "category": "edge_cases",
        "name": "Ambiguous Model Name",
        "turns": [
            {
                "query": "Opowiedz mi o trójce",
                "expectations": {
                    "should_contain_any": ["seria 3", "3 series", "series 3"],
                    "min_length": 80
                }
            }
        ]
    },
    {
        "category": "edge_cases",
        "name": "Mixed Language Query",
        "turns": [
            {
                "query": "What is the price of X5?",
                "expectations": {
                    "must_contain": ["x5"],
                    "min_length": 50
                }
            }
        ]
    },
    {
        "category": "edge_cases",
        "name": "Very Specific Technical",
        "turns": [
            {
                "query": "Jaki jest moment obrotowy X3 diesel?",
                "expectations": {
                    "must_contain": ["x3"],
                    "should_contain_any": ["nm", "moment", "torque", "diesel"],
                    "min_length": 50
                }
            }
        ]
    },
    {
        "category": "edge_cases",
        "name": "Non-existent Model",
        "turns": [
            {
                "query": "Opowiedz o BMW X9",
                "expectations": {
                    "should_contain_any": ["nie mam", "brak", "don't have", "salon", "kontakt"],
                    "min_length": 40
                }
            }
        ]
    },

    # Category 6: Service and Dealership
    {
        "category": "service",
        "name": "Service Center Query",
        "turns": [
            {
                "query": "Gdzie mogę zrobić przegląd BMW?",
                "expectations": {
                    "must_contain": ["zk motors"],
                    "should_contain_any": ["kielce", "radom", "rzeszów", "serwis"],
                    "min_length": 80
                }
            }
        ]
    },
    {
        "category": "service",
        "name": "Opening Hours",
        "turns": [
            {
                "query": "Jakie są godziny otwarcia salonu?",
                "expectations": {
                    "should_contain_any": ["9:00", "18:00", "poniedziałek", "sobota", "monday", "saturday"],
                    "min_length": 50
                }
            }
        ]
    },
    {
        "category": "service",
        "name": "Test Drive",
        "turns": [
            {
                "query": "Chciałbym umówić jazdę próbną",
                "expectations": {
                    "should_contain_any": ["salon", "kontakt", "zk motors", "contact", "test"],
                    "min_length": 60
                }
            }
        ]
    },

    # Category 7: Competitor and Off-topic
    {
        "category": "off_topic",
        "name": "MINI Models (Different Brand)",
        "turns": [
            {
                "query": "Opowiedz o MINI Cooper",
                "expectations": {
                    "must_contain": ["mini"],
                    "should_contain_any": ["salon", "zk motors", "kontakt", "doradca"],
                    "min_length": 60
                }
            }
        ]
    },
    {
        "category": "off_topic",
        "name": "Motorcycle Query",
        "turns": [
            {
                "query": "Jakie macie motocykle BMW?",
                "expectations": {
                    "must_contain": ["motocykl"],
                    "should_contain_any": ["stok", "link", "katalog", "salon"],
                    "min_length": 80
                }
            }
        ]
    },
    {
        "category": "off_topic",
        "name": "Completely Off-topic",
        "turns": [
            {
                "query": "Jaka jest pogoda?",
                "expectations": {
                    "should_contain_any": ["bmw", "pomoc", "help", "salon"],
                    "min_length": 30
                }
            }
        ]
    },

    # Category 8: Critical Regression Tests (from original log)
    {
        "category": "regression",
        "name": "X3 Should Not Return Leasing",
        "turns": [
            {
                "query": "Jestem zainteresowany X3",
                "expectations": {
                    "must_contain": ["x3"],
                    "should_contain_any": ["moc", "silnik", "km", "suv", "power", "engine"],
                    "must_not_contain": ["leasing jako główna odpowiedź"],
                    "min_length": 100,
                    "pattern_check": lambda r: ("leasing" not in r.lower()[:200] or
                                                  any(w in r.lower() for w in ["moc", "km", "silnik"]))
                }
            }
        ]
    },
    {
        "category": "regression",
        "name": "SUV Query Should Not Jump to Leasing",
        "turns": [
            {
                "query": "Chcę kupić SUV",
                "expectations": {
                    "should_contain_any": ["x1", "x3", "x5", "x7", "model", "moc"],
                    "min_length": 100,
                    "pattern_check": lambda r: ("leasing" not in r.lower()[:150] or
                                                  any(w in r.lower() for w in ["x1", "x3", "x5", "x7"]))
                }
            }
        ]
    }
]


async def main():
    """Run comprehensive tests"""
    tester = BotTester()
    await tester.run_all_tests(TEST_SUITE)


if __name__ == "__main__":
    asyncio.run(main())
