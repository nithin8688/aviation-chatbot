"""
Quick Test Script - Verify System is Production-Ready

Run this after making configuration changes to verify:
1. Accuracy (queries work correctly)
2. Performance (response time < 6 seconds)
3. Stability (no errors)

USAGE
─────
python quick_test.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_engine import get_rag_engine
from src.db_utils import get_total_chunks


# Test queries
TEST_QUERIES = [
    {
        "query": "What is baggage?",
        "expected_keywords": ["bags", "airport", "departure"],
        "max_time_s": 6
    },
    {
        "query": "What is the gazette of india?",
        "expected_keywords": ["official", "gazette", "notification"],
        "max_time_s": 6
    },
    {
        "query": "What is ILS?",
        "expected_keywords": ["Instrument Landing", "landing", "aircraft"],
        "max_time_s": 6
    },
    {
        "query": "What is pizza?",  # Off-topic
        "should_decline": True,
        "max_time_s": 3
    }
]


def run_quick_test():
    """Run quick test suite"""
    
    print("\n" + "="*70)
    print("QUICK TEST SUITE - Production Readiness Check")
    print("="*70)
    
    # ── Database check ───────────────────────────────────────────────
    print("\n📊 Step 1: Database Check")
    try:
        total_chunks = get_total_chunks()
        print(f"  ✅ Database connected: {total_chunks:,} chunks")
    except Exception as e:
        print(f"  ❌ Database error: {e}")
        print("  → Fix database connection before proceeding")
        return False
    
    # ── RAG engine check ─────────────────────────────────────────────
    print("\n🔄 Step 2: RAG Engine Initialization")
    try:
        rag_engine = get_rag_engine()
        print("  ✅ RAG engine loaded")
    except Exception as e:
        print(f"  ❌ RAG engine error: {e}")
        return False
    
    # ── Test queries ─────────────────────────────────────────────────
    print("\n🧪 Step 3: Test Queries")
    print("-" * 70)
    
    passed = 0
    failed = 0
    total_time = 0
    
    for i, test in enumerate(TEST_QUERIES, 1):
        query = test["query"]
        max_time = test["max_time_s"]
        
        print(f"\nTest {i}/{len(TEST_QUERIES)}: '{query}'")
        
        start = time.time()
        try:
            result = rag_engine.query(query)
            elapsed = time.time() - start
            total_time += elapsed
            
            answer = result.get("answer", "")
            num_sources = result.get("num_sources", 0)
            
            # Check timing
            timing_ok = elapsed <= max_time
            timing_status = "✅" if timing_ok else "⚠️"
            print(f"  {timing_status} Response time: {elapsed:.2f}s (max: {max_time}s)")
            
            # Check answer quality
            if test.get("should_decline"):
                # Off-topic query - should decline
                declined = any(phrase in answer.lower() for phrase in [
                    "not available", "cannot answer", "off-topic", "not in the documents"
                ])
                if declined:
                    print(f"  ✅ Correctly declined off-topic query")
                    passed += 1
                else:
                    print(f"  ❌ Should have declined, but answered anyway")
                    failed += 1
            else:
                # On-topic query - check keywords
                expected = test.get("expected_keywords", [])
                answer_lower = answer.lower()
                
                keywords_found = [kw for kw in expected if kw.lower() in answer_lower]
                keywords_missing = [kw for kw in expected if kw.lower() not in answer_lower]
                
                if len(keywords_found) >= len(expected) * 0.5:  # At least 50% of keywords
                    print(f"  ✅ Answer quality: Good")
                    print(f"     Keywords found: {', '.join(keywords_found[:3])}")
                    print(f"     Sources: {num_sources}")
                    passed += 1
                else:
                    print(f"  ⚠️ Answer quality: Poor")
                    print(f"     Missing keywords: {', '.join(keywords_missing)}")
                    failed += 1
                
                if not timing_ok:
                    print(f"  ⚠️ Performance issue: {elapsed:.2f}s > {max_time}s target")
        
        except Exception as e:
            elapsed = time.time() - start
            print(f"  ❌ Query failed: {e}")
            failed += 1
    
    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total tests: {len(TEST_QUERIES)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Success rate: {passed/len(TEST_QUERIES)*100:.0f}%")
    print(f"Average response time: {total_time/len(TEST_QUERIES):.2f}s")
    
    # ── Production readiness ─────────────────────────────────────────
    print("\n" + "="*70)
    print("PRODUCTION READINESS")
    print("="*70)
    
    avg_time = total_time / len(TEST_QUERIES)
    
    if passed == len(TEST_QUERIES) and avg_time <= 6:
        print("✅ READY FOR PRODUCTION!")
        print(f"   • All tests passed")
        print(f"   • Average time: {avg_time:.2f}s (acceptable)")
        print("\n🚀 You can deploy now!")
        return True
    
    elif passed >= len(TEST_QUERIES) * 0.75 and avg_time <= 8:
        print("⚠️ ALMOST READY - Minor issues")
        print(f"   • Pass rate: {passed/len(TEST_QUERIES)*100:.0f}%")
        print(f"   • Average time: {avg_time:.2f}s")
        print("\n💡 Recommendations:")
        
        if avg_time > 6:
            print("   • Reduce TOP_K_RETRIEVAL to 5 or 3 (in config.py)")
        if failed > 0:
            print("   • Review failed tests and adjust prompts")
        
        return False
    
    else:
        print("❌ NOT READY - Critical issues")
        print(f"   • Pass rate: {passed/len(TEST_QUERIES)*100:.0f}% (should be > 75%)")
        print(f"   • Average time: {avg_time:.2f}s (should be < 8s)")
        print("\n🔧 Required fixes:")
        print("   • Review OPTION_A_DEPLOY_GUIDE.md")
        print("   • Run diagnostic.py to identify bottlenecks")
        return False


if __name__ == "__main__":
    success = run_quick_test()
    sys.exit(0 if success else 1)