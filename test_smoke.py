"""
test_smoke.py — quick sanity check for the core RAG pipeline.
Does NOT require Ollama to be running.
"""
import sys

def run():
    print("── Multilingual Policy Assistant — Smoke Test ──\n")

    # 1. Config loads
    try:
        from config import settings
        print(f"✓ Config loaded  (policy_data_path='{settings.policy_data_path}')")
    except Exception as e:
        print(f"✗ Config failed: {e}")
        sys.exit(1)

    # 2. Database + policies load
    try:
        from database import PolicyDatabase
        db = PolicyDatabase()
        db.load_policies_from_directory(settings.policy_data_path)
        assert len(db.policies) > 0, "No policies loaded — make sure policy_data/ contains .json files"
        print(f"✓ Policies loaded  ({len(db.policies)} documents)")
    except Exception as e:
        print(f"✗ Policy loading failed: {e}")
        sys.exit(1)

    # 3. Embedding model + FAISS index
    try:
        assert db.index is not None, "FAISS index was not built"
        print(f"✓ FAISS index built  (dim={db._dimension}, docs={len(db.policies)})")
    except Exception as e:
        print(f"✗ Index check failed: {e}")
        sys.exit(1)

    # 4. Semantic search
    try:
        results = db.search_policies("EV subsidy benefits", language="en", top_k=3)
        assert len(results) > 0, "Search returned no results. Check SIMILARITY_THRESHOLD in config.py."
        top = results[0]
        print(f"✓ Search works  (top result: '{top.title}', score={top.score:.3f})")
    except Exception as e:
        print(f"✗ Search failed: {e}")
        sys.exit(1)

    # 5. Language detection
    try:
        kn_lang = db.detect_language("ಕರ್ನಾಟಕ ಸರ್ಕಾರದ ನೀತಿ ದಾಖಲೆ")
        print(f"✓ Language detection  (Kannada→'{kn_lang}')")
    except Exception as e:
        print(f"✗ Language detection failed: {e}")
        sys.exit(1)

    # 6. Translation (requires internet)
    try:
        translated = db.translate_from_english("Hello", target_lang="hi")
        print(f"✓ Translation works  ('Hello' → '{translated}')")
    except Exception as e:
        print(f"✗ Translation failed (non-fatal): {e}")

    print("\n── All smoke tests passed ✓ ──")
    print("\nNext step: Start the server with: uvicorn main:app --reload")

if __name__ == "__main__":
    run()
