import os
import pickle
import hashlib
import json
import glob
import re
from datetime import datetime
import asyncio
import types
from paperqa import Docs, Settings
from typing import Dict, List, Optional, Set, Tuple, Union

# For PDF processing to filter out references
import fitz  # PyMuPDF

# Monkey patch PaperQA's metadata fetching to completely disable it
import paperqa.clients

# Replace the metadata client query method with a no-op function that returns None
async def dummy_query(*args, **kwargs):
    return None

# Apply the patch to all metadata client classes
for attr_name in dir(paperqa.clients):
    attr = getattr(paperqa.clients, attr_name)
    if isinstance(attr, type) and hasattr(attr, 'query') and callable(attr.query):
        attr.query = dummy_query
        
# ---------------------------------------------------------------------------
# Reference section detection helpers
# ---------------------------------------------------------------------------

# Headers that signal the start of a reference section (case-insensitive)
_REF_HEADERS = [
    "references", "bibliography", "works cited", "literature cited",
    "reference list", "cited references", "references and notes",
    "references & notes", "bibliographical references",
]

# Regex: match a line that is ONLY a reference header (with optional whitespace)
_REF_HEADER_RE = re.compile(
    r"^\s*(" + "|".join(map(re.escape, _REF_HEADERS)) + r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Regex: numbered reference entries like [1], [2] or 1.  2.   3.
_REF_NUMBERED_RE = re.compile(r"(?:^|\n)\s*(?:\[\d+\]|\d+\.)\s", re.MULTILINE)

# Regex: author-year patterns like "Smith (2020)" or "(Smith, 2020)"
_REF_AUTHOR_YEAR_RE = re.compile(
    r"\b[A-Z][a-z]+(?:\s(?:et al\.|and\s[A-Z][a-z]+))?\s*\(\d{4}\)"
)



def _score_reference_page(text: str) -> float:
    """Return a confidence score [0-10] that *text* is from a reference section."""
    score = 0.0

    # --- Strategy 1: header line (strongest signal) ---------------------------
    header_matches = _REF_HEADER_RE.findall(text)
    # Boost score if header appears near the top of the page
    for header in header_matches:
        pos = text.lower().find(header.lower())
        # Header in first 30 % of page content → stronger signal
        if pos >= 0 and pos < len(text) * 0.3:
            score += 5.0
        else:
            score += 3.0

    # --- Strategy 2: word-boundary match as fallback --------------------------
    # Catch headers that aren't on a line of their own (e.g. "7. REFERENCES")
    lower = text.lower()
    for header in _REF_HEADERS:
        idx = lower.find(header)
        if idx >= 0:
            # Make sure it's a word boundary, not "preferences" etc.
            before = lower[idx - 1] if idx > 0 else " "
            after_char = lower[idx + len(header)] if idx + len(header) < len(lower) else " "
            if not before.isalpha() and not after_char.isalpha():
                # Found genuine header word — check if near top of page
                if idx < len(text) * 0.3:
                    score += 3.0
                else:
                    score += 2.0
                break  # one header bonus is enough

    # --- Strategy 3: numbered reference entries -------------------------------
    numbered = _REF_NUMBERED_RE.findall(text)
    # Each numbered entry adds to score (capped)
    score += min(len(numbered), 8) * 0.8

    # --- Strategy 4: author-year citations in running text --------------------
    author_year = _REF_AUTHOR_YEAR_RE.findall(text)
    score += min(len(author_year), 6) * 0.4

    return min(score, 10.0)


def preprocess_pdf(pdf_path: str) -> str:
    """
    Process a PDF to exclude reference sections and create a filtered version.

    Uses a multi-strategy scoring system:
      1. Reference header detection (line-exact + word-boundary)
      2. Numbered entry patterns ([1], 1., etc.)
      3. Author-year citation patterns

    Returns the path to the filtered PDF (or the original if filtering is not
    needed / fails).
    """
    try:
        pdf_path = os.path.abspath(pdf_path)

        if not os.path.exists(pdf_path):
            print(f"PDF file not found: {pdf_path}")
            return pdf_path

        base_name = os.path.basename(pdf_path)
        dir_name = os.path.dirname(pdf_path)
        filtered_path = os.path.join(dir_name, f"filtered_{base_name}")

        # Reuse existing filtered version
        if os.path.exists(filtered_path):
            print(f"Using existing filtered version: {filtered_path}")
            return filtered_path

        print(f"Opening PDF for preprocessing: {pdf_path}")
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"PDF has {total_pages} pages")

        # ---------------------------------------------------------------
        # Phase 1 — Score every page in the second half of the document
        # ---------------------------------------------------------------
        search_start = max(0, int(total_pages * 0.3))
        page_scores: list[tuple[int, float]] = []

        for page_num in range(search_start, total_pages):
            text = doc[page_num].get_text("text")
            score = _score_reference_page(text)
            page_scores.append((page_num, score))

        # ---------------------------------------------------------------
        # Phase 2 — Find the best candidate reference-start page
        # ---------------------------------------------------------------
        # Heuristic: a reference section starts when:
        #   (a) a page has score >= threshold, AND
        #   (b) at least one of the next 2 pages also has score >= threshold/2
        #       (or we are on the last page)

        threshold = 3.0
        reference_page_start = total_pages  # default: no filtering

        # Guard: reference section must start after at least 25% of the document
        # (prevents false positives from citations in abstract/introduction)
        min_ref_page = max(1, int(total_pages * 0.25))

        for i, (page_num, score) in enumerate(page_scores):
            if score < threshold:
                continue
            # Skip pages that are too early in the document
            if page_num < min_ref_page:
                continue
            # Check consistency: at least one neighbour page also looks like refs
            next_i = i + 1
            next_next_i = i + 2
            has_neighbour = (
                (next_i < len(page_scores) and page_scores[next_i][1] >= threshold * 0.5)
                or (next_next_i < len(page_scores) and page_scores[next_next_i][1] >= threshold * 0.5)
                or page_num >= total_pages - 2  # last 2 pages → accept
            )
            if has_neighbour:
                reference_page_start = page_num
                print(
                    f"Reference section detected starting at page {page_num + 1} "
                    f"of {total_pages} (score={score:.1f})"
                )
                # Log all scored pages for debugging
                for pn, sc in page_scores:
                    if sc >= 1.0:
                        print(f"  Page {pn + 1}: score={sc:.1f}")
                break

        # ---------------------------------------------------------------
        # Phase 3 — Remove reference pages from the document
        # ---------------------------------------------------------------
        if reference_page_start < total_pages:
            pages_before_refs = reference_page_start
            ref_page_count = total_pages - pages_before_refs
            print(
                f"Creating filtered PDF without reference section "
                f"(pages {reference_page_start + 1}-{total_pages}, "
                f"{ref_page_count} page(s))"
            )
            new_doc = fitz.open()
            for i in range(pages_before_refs):
                new_doc.insert_pdf(doc, from_page=i, to_page=i)
            new_doc.save(filtered_path)
            new_doc.close()
            doc.close()
            print(f"Saved filtered PDF: {filtered_path}")
            return filtered_path

        # No reference section detected
        print("No reference section detected; using original PDF")
        doc.close()
        return pdf_path

    except Exception as e:
        print(f"Error preprocessing PDF {pdf_path}: {str(e)}")
        return pdf_path  # Return original on error

class PDFVectorStorage:
    """
    A simple vector storage system for PaperQA document embeddings.
    Stores embeddings for PDF documents to avoid re-embedding on each run.
    """
    
    def __init__(self, pdf_dir="./pdf", storage_dir="./vector_storage"):
        """
        Initialize the vector storage system.
        
        Args:
            pdf_dir: Directory containing PDF files
            storage_dir: Directory to store vector embeddings
        """
        self.pdf_dir = pdf_dir
        self.storage_dir = storage_dir
        self.storage_file = os.path.join(storage_dir, "pdf_embeddings.pkl")
        self.hash_file = os.path.join(storage_dir, "pdf_hashes.pkl")
        
        # Create storage directory if it doesn't exist
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
    
    def _get_file_hash(self, filepath):
        """Calculate MD5 hash of a file to detect changes."""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            buf = f.read(65536)  # Read in 64k chunks
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    
    def _get_pdf_hashes(self):
        """Get MD5 hashes of all PDF files in the directory."""
        # Ensure pdf_dir is an absolute path
        pdf_dir_abs = os.path.abspath(self.pdf_dir)
        pdf_files = glob.glob(os.path.join(pdf_dir_abs, "*.pdf"))
        # Exclude filtered versions (they are derived from originals)
        pdf_files = [f for f in pdf_files if "filtered_" not in os.path.basename(f)]
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {pdf_dir_abs}")
            
        print(f"Found {len(pdf_files)} PDF files in {pdf_dir_abs}")
        # Print first few files for debugging
        for i, pdf in enumerate(pdf_files[:3]):
            print(f"  Sample file {i+1}: {pdf}")
        if len(pdf_files) > 3:
            print(f"  ... and {len(pdf_files)-3} more files")
        
        hashes = {}
        for pdf_file in pdf_files:
            try:
                with open(pdf_file, "rb") as f:
                    content = f.read()
                    hashes[os.path.basename(pdf_file)] = hashlib.md5(content).hexdigest()
            except Exception as e:
                print(f"Warning: Could not hash file {pdf_file}: {str(e)}")
        
        return hashes
    
    def _save_hashes(self, pdf_hashes):
        """Save PDF file hashes to detect changes."""
        with open(self.hash_file, 'wb') as f:
            pickle.dump(pdf_hashes, f)
    
    def _load_hashes(self):
        """Load saved PDF file hashes."""
        if os.path.exists(self.hash_file):
            with open(self.hash_file, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def _save_docs(self, docs):
        """Save PaperQA Docs object to storage."""
        with open(self.storage_file, 'wb') as f:
            pickle.dump(docs, f)
        print(f"Saved embeddings to {self.storage_file}")
    
    def _load_docs(self):
        """Load PaperQA Docs object from storage."""
        if os.path.exists(self.storage_file):
            with open(self.storage_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    async def get_docs_async(self):
        """Asynchronous method to load or create PaperQA Docs embeddings.
        Uses an append-only approach that only processes new PDFs while keeping existing embeddings.
        
        Returns:
            Docs: PaperQA Docs object with embeddings
        """
        # Check if embeddings need to be created or updated
        current_hashes = self._get_pdf_hashes()
        saved_hashes = self._load_hashes()
        
        # Get list of PDF files - ensure we use absolute paths
        pdf_dir_abs = os.path.abspath(self.pdf_dir)
        pdf_files = glob.glob(os.path.join(pdf_dir_abs, "*.pdf"))
        # Exclude filtered versions (they are derived from originals)
        pdf_files = [f for f in pdf_files if "filtered_" not in os.path.basename(f)]
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {pdf_dir_abs}")
            
        print(f"Found {len(pdf_files)} PDF files in {pdf_dir_abs}")
        # Print first few files for debugging
        for i, pdf in enumerate(pdf_files[:3]):
            print(f"  Sample file {i+1}: {pdf}")
        if len(pdf_files) > 3:
            print(f"  ... and {len(pdf_files)-3} more files")
        
        # Compare current and saved hashes to identify new or changed files
        new_or_changed_files = []
        unchanged_files = []
        
        for pdf_file in pdf_files:
            pdf_name = os.path.basename(pdf_file)
            if pdf_name not in saved_hashes or saved_hashes[pdf_name] != current_hashes[pdf_name]:
                new_or_changed_files.append(pdf_file)
            else:
                unchanged_files.append(pdf_file)
        
        # If no changes and embeddings exist, just load existing embeddings
        if not new_or_changed_files and os.path.exists(self.storage_file):
            print("No changes detected in PDF files. Using existing embeddings.")
            return self._load_docs()
        
        start_time = datetime.now()
        
        # Load existing embeddings if available, otherwise create new Docs object
        if os.path.exists(self.storage_file) and len(unchanged_files) > 0:
            print(f"Loading existing embeddings and appending {len(new_or_changed_files)} new/changed files...")
            docs = self._load_docs()
        else:
            print(f"Creating new embeddings for {len(new_or_changed_files)} files...")
            docs = Docs()
        
        # Extra safety - ensure metadata clients are empty
        docs._metadata_clients = []
        
        # Process new or changed files
        if new_or_changed_files:
            print(f"Processing {len(new_or_changed_files)} new or changed PDF files...")
            
            for pdf_file in new_or_changed_files:
                pdf_basename = os.path.basename(pdf_file)
                print(f"Processing {pdf_basename}...")
                
                # Preprocess the PDF to filter out reference sections
                try:
                    # Make sure we're using absolute paths
                    pdf_abs_path = os.path.abspath(pdf_file)
                    if not os.path.exists(pdf_abs_path):
                        print(f"ERROR: File not found: {pdf_abs_path}")
                        continue
                        
                    filtered_pdf = preprocess_pdf(pdf_abs_path)
                    print(f"Adding {os.path.basename(filtered_pdf)} to embeddings...")
                    
                    # Create settings that use our model to avoid costly gpt-4o rate limits
                    add_settings = Settings()
                    add_settings.parsing.enrichment_llm = "gpt-5.4-nano-2026-03-17"
                    
                    # Add the filtered PDF to docs
                    await docs.aadd(
                        filtered_pdf, 
                        content_only=True, 
                        disable_metadata=True,
                        force_filename=pdf_basename,  # Keep original filename for references
                        settings=add_settings,
                    )
                    
                    # If we created a filtered version, clean it up if it's not the original
                    if filtered_pdf != pdf_abs_path and os.path.exists(filtered_pdf):
                        # Keep filtered files for now for debugging
                        pass
                        # Uncomment to clean up: os.remove(filtered_pdf)
                    
                    # Save progress incrementally after each successful PDF
                    self._save_docs(docs)
                    self._save_hashes(current_hashes)
                    print(f"  Progress saved ({pdf_basename} done)")
                except Exception as e:
                    print(f"Error processing {pdf_basename}: {str(e)}")
                    try:
                        # Fallback to original file if preprocessing fails
                        if os.path.exists(pdf_abs_path):
                            await docs.aadd(
                                pdf_abs_path, 
                                content_only=True, 
                                disable_metadata=True,
                                force_filename=pdf_basename,
                                settings=add_settings
                            )
                        else:
                            print(f"CRITICAL ERROR: Cannot find file {pdf_abs_path}")
                    except Exception as inner_e:
                        print(f"CRITICAL ERROR adding {pdf_basename}: {str(inner_e)}")
                        continue
            
            # Save updated embeddings and hashes
            self._save_docs(docs)
            self._save_hashes(current_hashes)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            print(f"Embedding completed in {duration:.2f} seconds")
        else:
            print("No new files to process.")
        
        return docs

    def get_docs(self):
        """
        Synchronous version that returns saved embeddings if available.
        If embeddings need to be created, raises an error suggesting to use the async version.
        
        Returns:
            PaperQA Docs object with embeddings if already saved
        """
        # Check if we need to update embeddings
        current_hashes = self._get_pdf_hashes()
        saved_hashes = self._load_hashes()
        
        # If embeddings exist and are up-to-date, return them
        if os.path.exists(self.storage_file):
            if len(current_hashes) == len(saved_hashes):
                # Check if any file changed
                need_update = False
                for pdf_name, current_hash in current_hashes.items():
                    if pdf_name not in saved_hashes or saved_hashes[pdf_name] != current_hash:
                        need_update = True
                        break
                        
                if not need_update:
                    print("Loading saved embeddings...")
                    return self._load_docs()
        
        # If we need to update, raise error
        raise RuntimeError(
            "Embeddings need to be created or updated. Use the async version 'await vector_storage.get_docs_async()' instead."
        )
