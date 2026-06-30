"""
Dry Run: Test PaperQA retrieval quality for different settings.
Loads existing embeddings (no re-embed cost), queries the retriever,
and shows what chunks would be used for answering.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from paperqa import Settings, Docs
from pdf_vector_storage import PDFVectorStorage
import asyncio

QUESTIONS = {
    "3": "Which signs, symptoms, and biomarkers (e.g., NT-proBNP, BNP) have the highest diagnostic accuracy in differentiating PPCM from other causes of dyspnea and edema in the peripartum period?",
}

async def test_retrieval(question_num, evidence_k=40, mmr_lambda=1.0):
    """Test what chunks are retrieved for a question."""
    q = QUESTIONS[question_num]
    print(f"\n{'='*70}")
    print(f"  DRY RUN: Question {question_num}")
    print(f"  evidence_k={evidence_k}, mmr_lambda={mmr_lambda}")
    print(f"  {q[:120]}...")
    print(f"{'='*70}")
    
    # Load docs from saved embeddings
    pdf_dir = "./pdf"
    vector_storage_dir = "./vector_storage"
    vector_storage = PDFVectorStorage(pdf_dir=pdf_dir, storage_dir=vector_storage_dir)
    docs = vector_storage.get_docs()
    
    if docs is None:
        print("❌ No saved embeddings found.")
        return
    
    print(f"✅ Embeddings loaded")
    
    # Configure settings
    settings = Settings()
    settings.answer.evidence_k = evidence_k
    settings.texts_index_mmr_lambda = mmr_lambda
    
    # TEST: Raw retrieval
    print(f"\n--- Retrieving top {evidence_k} chunks ---")
    try:
        texts = await docs.retrieve_texts(q, k=evidence_k, settings=settings)
        print(f"✅ Retrieved {len(texts)} text chunks")
        
        # Document distribution
        doc_names = {}
        for t in texts:
            name = t.doc.name if hasattr(t, 'doc') and hasattr(t.doc, 'name') else 'unknown'
            doc_names[name] = doc_names.get(name, 0) + 1
        
        print(f"\n📊 {len(doc_names)} unique documents retrieved:")
        for name, count in sorted(doc_names.items(), key=lambda x: -x[1]):
            print(f"   {count:2d} chunks | {name[:70]}")
        
        # Top 5 chunks full content
        print(f"\n📄 Top 5 chunks content:")
        for i, t in enumerate(texts[:5]):
            name = t.doc.name if hasattr(t, 'doc') and hasattr(t.doc, 'name') else 'unknown'
            score = str(t.score) if hasattr(t, 'score') else 'N/A'
            text = t.text[:300].replace('\n', ' ').strip() if hasattr(t, 'text') else 'N/A'
            print(f"\n--- [{i+1}] {name[:60]} (score={score}) ---")
            print(f"  {text}...")
        
        # Check if key content was retrieved
        print(f"\n🔍 Content check:")
        all_text = " ".join([(t.text or "") for t in texts])
        checks = [
            ("NT-proBNP/BNP", "nt-probnp" in all_text.lower() or " bnp " in all_text.lower()),
            ("LVEF cutoff", "lvef" in all_text.lower() or "ejection fraction" in all_text.lower()),
            ("S3 gallop", "third heart sound" in all_text.lower() or "s3" in all_text.lower()),
            ("Sliwa 2010", "sliwa" in all_text.lower()),
            ("ESC consensus", "position statement" in all_text.lower() or "heart failure association" in all_text.lower()),
        ]
        for label, found in checks:
            print(f"   {'✅' if found else '❌'} {label}")
        
        return texts
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    print("="*70)
    print("  DRY RUN RETRIEVAL — PaperQA Chunk Inspection")
    print("  Tanpa generate answer, cuma lihat apa yang di-retrieve")
    print("  Token cost: ~1 query embedding = < $0.0001")
    print("="*70)
    
    # Test question 3 (biomarkers)
    print("\n═══ CONFIG A: evidence_k=40, mmr=1.0 (default) ═══")
    await test_retrieval("3", evidence_k=40, mmr_lambda=1.0)

if __name__ == "__main__":
    asyncio.run(main())
