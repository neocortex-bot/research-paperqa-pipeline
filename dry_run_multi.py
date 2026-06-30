"""
Dry Run Multi-Config: Test PaperQA retrieval with different settings.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()
from paperqa import Settings, Docs
from pdf_vector_storage import PDFVectorStorage
import asyncio

Q = "Which signs, symptoms, and biomarkers (e.g., NT-proBNP, BNP) have the highest diagnostic accuracy in differentiating PPCM from other causes of dyspnea and edema in the peripartum period?"

CONFIGS = [
    # (name, evidence_k, mmr_lambda, chunk_chars)
    ("A: k=40, mmr=1.0 (default)",   40, 1.0, 5000),
    ("B: k=40, mmr=0.5 (diverse)",    40, 0.5, 5000),
    ("C: k=60, mmr=0.7",             60, 0.7, 5000),
    ("D: k=60, mmr=0.3 (max diverse)",60, 0.3, 5000),
]

KEYWORDS = [
    "BNP|NT-proBNP|nt-probnp",
    "ejection fraction|lvef|LVEF",
    "third heart sound|S3 gallop",
    "physiolog|remodeling|normal pregnancy",
    "consensus|position statement|Heart Failure Association",
    "bromocriptine|dopamine agonist",
]

import re

async def run_test(name, evidence_k, mmr_lambda, chunk_chars):
    print(f"\n{'='*60}")
    print(f"  TEST: {name}")
    print(f"{'='*60}")
    
    vector_storage = PDFVectorStorage(pdf_dir="./pdf", storage_dir="./vector_storage")
    docs = vector_storage.get_docs()
    if not docs:
        print("  ❌ No embeddings")
        return None
    
    settings = Settings()
    settings.answer.evidence_k = evidence_k
    settings.texts_index_mmr_lambda = mmr_lambda
    settings.parsing.reader_config = {"chunk_chars": chunk_chars, "overlap": 250}
    
    texts = await docs.retrieve_texts(Q, k=evidence_k, settings=settings)
    
    # Stats
    all_text = " ".join([(t.text or "") for t in texts])
    total_chars = len(all_text)
    
    # Content keywords check
    matches = {}
    for kw in KEYWORDS:
        count = len(re.findall(kw.lower(), all_text.lower()))
        matches[kw.split("|")[0]] = count
    
    # Top docs estimate from content fingerprint
    unique_refs = set()
    for t in texts:
        txt = (t.text or "").lower()
        # Detect which PDF based on content fingerprint
        if "bromocriptine" in txt and "sliwa" in txt:
            unique_refs.add("Sliwa2010-Br")
        elif "bromocriptine" in txt:
            unique_refs.add("Sliwa2010")
        elif "position statement" in txt or "heart failure association" in txt:
            unique_refs.add("Sliwa2010-ESC")
        elif "sanghavi" in txt.lower():
            unique_refs.add("Sanghavi2014")
        elif "hagen" in txt.lower() or "ropac" in txt.lower():
            unique_refs.add("Hagen2020")
        elif "chu" in txt.lower():
            unique_refs.add("Chu2020")
        elif "paauw" in txt.lower():
            unique_refs.add("Paauw2017")
        elif "heidenreich" in txt.lower():
            unique_refs.add("Heidenreich2022")
    
    print(f"  📊 {len(texts)} chunks, {total_chars:,} total chars")
    print(f"  📚 Docs detected: {unique_refs or 'unknown'}")
    print(f"  🔍 Keywords found:")
    for kw, count in sorted(matches.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"     ✅ {kw}: {count}x")
    
    # Coverage score: higher = better content diversity
    non_zero = sum(1 for c in matches.values() if c > 0)
    coverage = f"{non_zero}/{len(KEYWORDS)}"
    print(f"  🎯 Coverage: {coverage}")
    
    return {"texts": texts, "coverage": coverage, "docs": unique_refs, "matches": matches}

async def main():
    print("="*60)
    print("  DRY RUN MULTI-CONFIG")
    print("  Membandingkan 4 konfigurasi retrieval")
    print("  Token: ~4 query embedding ≈ $0.0004")
    print("="*60)
    
    results = []
    for name, ek, mmr, cc in CONFIGS:
        r = await run_test(name, ek, mmr, cc)
        if r:
            results.append((name, r))
    
    print(f"\n\n{'='*60}")
    print(f"  RANKING")
    print(f"{'='*60}")
    for name, r in sorted(results, key=lambda x: sum(x[1]['matches'].values()), reverse=True):
        print(f"  {name}")
        print(f"    Coverage: {r['coverage']}, Docs: {r['docs']}")
        print(f"    Keyword hits: {sum(r['matches'].values())}")

if __name__ == "__main__":
    asyncio.run(main())
