"""
Test: Multi-Query retrieval (HyDE + document-like phrase)
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()
from paperqa import Settings, Docs
from pdf_vector_storage import PDFVectorStorage
from process_all_questions import generate_hypothetical_answer
import asyncio

Q1 = "What is the exact definition of peripartum cardiomyopathy (PPCM) according to the most recent international consensus, and how does it differentiate between de novo PPCM and underlying chronic heart failure with reduced ejection fraction (HFrEF) that predates pregnancy?"

DEFINITION_PHRASES = [
    "EF is nearly always reduced below 45%",
    "diagnosis of exclusion",
    "idiopathic cardiomyopathy",
    "HF secondary to left ventricular systolic dysfunction",
    "no other cause of HF is found",
    "We propose the following simplified definition",
    "position statement",
    "Heart Failure Association",
]

async def main():
    print("=" * 70)
    print("TEST: Multi-Query Retrieval")
    print("=" * 70)
    
    # Generate HyDE
    hyde = await generate_hypothetical_answer(Q1)
    
    # Build multi-angle query
    hyde_snippet = hyde[:300].strip()
    hyde_lines = hyde_snippet.split('\n')
    doc_like_phrase = ""
    for line in hyde_lines:
        lower = line.lower()
        if any(w in lower for w in ['definition', 'diagnostic criteria', 'is characterized', 'is defined', 'we propose']):
            doc_like_phrase = line.strip()
            break
    if not doc_like_phrase and hyde_lines:
        doc_like_phrase = hyde_lines[0].strip()
    if len(doc_like_phrase) > 200:
        doc_like_phrase = doc_like_phrase[:200]
    
    retrieval_q = f"{Q1}\n\nKey aspects: {hyde_snippet}\n\n{doc_like_phrase}"
    
    print(f"\nRetrieval query ({len(retrieval_q)} chars)")
    print(f"  Doc-like phrase: {doc_like_phrase[:100]}...")
    
    # Retrieve
    vector_storage = PDFVectorStorage(pdf_dir="./pdf", storage_dir="./vector_storage")
    docs = vector_storage.get_docs()
    
    settings = Settings()
    settings.answer.evidence_k = 60
    settings.texts_index_mmr_lambda = 0.7
    
    texts = await docs.retrieve_texts(retrieval_q, k=60, settings=settings)
    print(f"\nRetrieved {len(texts)} chunks")
    
    # Check definition phrases
    full_text = " ".join([(t.text or "").lower() for t in texts])
    
    print("\n📋 DEFINITION PHRASES:")
    all_found = True
    for phrase in DEFINITION_PHRASES:
        found = phrase.lower() in full_text
        if not found:
            all_found = False
        print(f"  {'✅' if found else '❌'} {phrase}")
    
    # Show definition chunk if found
    for i, t in enumerate(texts):
        text = (t.text or "").lower()
        if "idiopathic cardiomyopathy" in text:
            print(f"\n📄 DEFINITION CHUNK #{i+1}:")
            print(f"  {t.text[:500]}")
            break
    else:
        # Show doc names
        print(f"\n📚 Document distribution:")
        doc_names = {}
        for t in texts:
            name = t.doc.name if hasattr(t, 'doc') and hasattr(t.doc, 'name') else 'unknown'
            doc_names[name] = doc_names.get(name, 0) + 1
        for name, count in sorted(doc_names.items(), key=lambda x: -x[1])[:10]:
            print(f"  {count:2d} | {name[:70]}")

if __name__ == "__main__":
    asyncio.run(main())
