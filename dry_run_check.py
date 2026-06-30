"""
Test: For question 1, check if the ESC definition chunk is retrieved with k=60, mmr=0.7
Checks for specific text from Sliwa 2010 page 2
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()
from paperqa import Settings, Docs
from pdf_vector_storage import PDFVectorStorage
import asyncio

Q1 = "What is the exact definition of peripartum cardiomyopathy (PPCM) according to the most recent international consensus, and how does it differentiate between de novo PPCM and underlying chronic heart failure with reduced ejection fraction (HFrEF) that predates pregnancy?"

# Key phrases from Sliwa 2010 page 2 that define PPCM
DEFINITION_PHRASES = [
    "EF is nearly always reduced below 45%",
    "diagnosis of exclusion",
    "idiopathic cardiomyopathy",
    "HF secondary to left ventricular systolic dysfunction",
    "towards the end of pregnancy",
    "no other cause of HF is found",
    "We propose the following simplified definition",
    "position statement",
    "Heart Failure Association",
    "Working Group on peripartum cardiomyopathy",
    "ESC",
    "European Society of Cardiology",
]

async def main():
    vector_storage = PDFVectorStorage(pdf_dir="./pdf", storage_dir="./vector_storage")
    docs = vector_storage.get_docs()
    
    settings = Settings()
    settings.answer.evidence_k = 60
    settings.texts_index_mmr_lambda = 0.7
    
    print("=" * 70)
    print("TEST: Retrieve texts for Q1 with k=60, mmr=0.7")
    print("=" * 70)
    
    texts = await docs.retrieve_texts(Q1, k=60, settings=settings)
    
    print(f"\nRetrieved {len(texts)} chunks\n")
    
    # Check for definition phrases
    full_text = " ".join([(t.text or "").lower() for t in texts])
    
    print("📋 DEFINITION PHRASES FOUND IN RETRIEVED CHUNKS:")
    all_found = True
    for phrase in DEFINITION_PHRASES:
        found = phrase.lower() in full_text
        if not found:
            all_found = False
        print(f"  {'✅' if found else '❌'} {phrase}")
    
    print(f"\n{'✅ ALL PHRASES PRESENT' if all_found else '❌ SOME PHRASES MISSING'}")
    
    # If definition found, show which chunk
    def_keyword = "idiopathic cardiomyopathy"
    for i, t in enumerate(texts[:10]):
        text = (t.text or "").lower()
        if def_keyword in text:
            print(f"\n📄 DEFINITION CHUNK #{i+1}:")
            print(f"  {t.text[:500]}...")
            break
    else:
        print(f"\n❌ Definition chunk NOT in top 60!")
        
        # Show what IS in top 5
        print("\n📄 TOP 5 CHUNKS (preview):")
        for i, t in enumerate(texts[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"  {t.text[:200]}...")

if __name__ == "__main__":
    asyncio.run(main())
