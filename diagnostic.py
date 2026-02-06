"""
Performance Diagnostic Script
Run this to identify exactly where time is spent in your RAG pipeline

USAGE
─────
python diagnostic.py

OUTPUT
──────
Will show timing breakdown:
  ⏱️ Embedding: 45ms
  ⏱️ Hybrid search: 52ms
  ⏱️ Gemini API: 3.2s
  ⏱️ TOTAL: 3.4s

This helps you identify bottlenecks.
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_engine import get_rag_engine
from src.db_utils import get_total_chunks, get_document_stats


def run_diagnostics():
    """Run comprehensive performance diagnostics"""
    
    print("\n" + "="*70)
    print("PERFORMANCE DIAGNOSTIC REPORT")
    print("="*70)
    
    # ── Database stats ───────────────────────────────────────────────
    print("\n📊 Database Status:")
    try:
        total_chunks = get_total_chunks()
        doc_stats = get_document_stats()
        
        print(f"  Total chunks: {total_chunks:,}")
        print(f"  Documents: {len(doc_stats)}")
        print("\n  Chunks per document:")
        for doc_name, chunk_count in doc_stats:
            print(f"    • {doc_name}: {chunk_count:,} chunks")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # ── RAG Engine initialization ────────────────────────────────────
    print("\n🔄 Initializing RAG Engine...")
    init_start = time.time()
    try:
        rag_engine = get_rag_engine()
        init_time = time.time() - init_start
        print(f"  ✅ Initialized in {init_time:.2f}s")
    except Exception as e:
        print(f"  ❌ Initialization failed: {e}")
        return
    
    # ── Test query ───────────────────────────────────────────────────
    test_query = "What is baggage?"
    print(f"\n🧪 Test Query: '{test_query}'")
    print("-" * 70)
    
    total_start = time.time()
    
    try:
        result = rag_engine.query(test_query)
        
        total_time = time.time() - total_start
        
        # Display results
        print(f"\n✅ Query completed in {total_time:.2f}s")
        print(f"\n📝 Answer preview:")
        answer = result.get("answer", "")
        print(f"  {answer[:200]}...")
        
        print(f"\n📚 Sources retrieved: {result.get('num_sources', 0)}")
        
        # Performance analysis
        print("\n" + "="*70)
        print("PERFORMANCE BREAKDOWN")
        print("="*70)
        
        if total_time > 5:
            print(f"⚠️  TOTAL TIME: {total_time:.2f}s (TARGET: 3-5s)")
            print("\nLikely bottlenecks:")
            
            if total_time > 10:
                print("  🔴 CRITICAL: Response time over 10 seconds")
                print("  Possible causes:")
                print("    • BM25 index rebuilding on every query (should be cached)")
                print("    • Database connection not using pool")
                print("    • Network latency to Gemini API")
                print("    • Too many chunks being processed")
            
            elif total_time > 7:
                print("  🟡 WARNING: Response time 7-10 seconds")
                print("  Possible causes:")
                print("    • Gemini API slow (normal range: 2-4s)")
                print("    • BM25 cache not working")
                print("    • Database queries not optimized")
            
            elif total_time > 5:
                print("  🟡 MINOR: Response time 5-7 seconds")
                print("  Possible causes:")
                print("    • Gemini API on slower side")
                print("    • Network latency")
        
        else:
            print(f"✅ GOOD: Total time {total_time:.2f}s (within 3-5s target)")
        
        # Check console output for detailed timing
        print("\n💡 Check console output above for detailed timing:")
        print("   Look for lines like:")
        print("   ⏱️  Embedding: XXms")
        print("   ⏱️  Hybrid search: XXms")
        print("   ⏱️  Gemini API: X.Xs")
        
    except Exception as e:
        print(f"\n❌ Query failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ── Configuration check ──────────────────────────────────────────
    print("\n" + "="*70)
    print("CONFIGURATION CHECK")
    print("="*70)
    
    from src.config import (
        USE_HYBRID_SEARCH,
        USE_HYDE,
        USE_RERANKING,
        HYBRID_ALPHA,
        TOP_K_RETRIEVAL
    )
    
    print(f"\n🔧 Current settings:")
    print(f"  USE_HYBRID_SEARCH: {USE_HYBRID_SEARCH} {'✅' if USE_HYBRID_SEARCH else '❌ SHOULD BE TRUE'}")
    print(f"  USE_HYDE: {USE_HYDE} {'⚠️ Adds ~2s' if USE_HYDE else '✅ Disabled (fast)'}")
    print(f"  USE_RERANKING: {USE_RERANKING} {'⚠️ Adds ~500ms' if USE_RERANKING else '✅ Disabled (fast)'}")
    print(f"  HYBRID_ALPHA: {HYBRID_ALPHA}")
    print(f"  TOP_K_RETRIEVAL: {TOP_K_RETRIEVAL}")
    
    print("\n💡 Recommendations:")
    if not USE_HYBRID_SEARCH:
        print("  🔴 CRITICAL: Enable USE_HYBRID_SEARCH = True")
    
    if USE_HYDE:
        print("  🟡 Consider disabling USE_HYDE for faster responses")
    
    if USE_RERANKING:
        print("  🟡 Consider disabling USE_RERANKING for faster responses")
    
    if USE_HYBRID_SEARCH and not USE_HYDE and not USE_RERANKING:
        print("  ✅ Configuration optimized for speed")
    
    # ── BM25 cache check ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("BM25 CACHE STATUS")
    print("="*70)
    
    try:
        from src.hybrid_search import _bm25_cache
        
        if _bm25_cache["index"] is not None:
            print(f"  ✅ BM25 index cached ({_bm25_cache['last_count']} chunks)")
            print("  Cache is working - subsequent queries will be fast")
        else:
            print("  ⚠️ BM25 index not built yet")
            print("  Will be built on first query (one-time cost)")
    except Exception as e:
        print(f"  ⚠️ Could not check cache: {e}")
    
    print("\n" + "="*70)
    print("END OF DIAGNOSTIC REPORT")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_diagnostics()