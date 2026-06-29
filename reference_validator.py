import os
import re
import glob
from typing import List, Dict, Set, Tuple

class ReferenceValidator:
    """
    A utility class to validate references in PaperQA responses against actual PDF files.
    """
    
    def __init__(self, pdf_dir="./pdf"):
        """
        Initialize the reference validator.
        
        Args:
            pdf_dir: Directory containing PDF files
        """
        self.pdf_dir = pdf_dir
        self.pdf_files = self._get_pdf_files()
        self.pdf_basenames = self._get_pdf_basenames()
        
    def _get_pdf_files(self) -> List[str]:
        """Get full paths of all PDF files in the directory."""
        return glob.glob(os.path.join(self.pdf_dir, "*.pdf"))
    
    def _get_pdf_basenames(self) -> Set[str]:
        """Get basenames of all PDF files without extension."""
        return {os.path.splitext(os.path.basename(f))[0].lower() for f in self.pdf_files}
    
    def extract_references(self, text: str) -> List[str]:
        """
        Extract reference identifiers from a text.
        
        Args:
            text: Text containing references in format (author2023title)
            
        Returns:
            List of reference identifiers
        """
        # Match patterns like (author2023title) or (author2023title pages x-y)
        pattern = r'\(([a-zA-Z0-9]+\d{4}[a-zA-Z0-9_]+)(?:\s+pages\s+\d+-\d+)?\)'
        matches = re.findall(pattern, text)
        return matches
    
    def validate_references(self, text: str) -> Tuple[bool, List[str], List[str]]:
        """
        Validate references in a text against available PDF files.
        
        Args:
            text: Text containing references
            
        Returns:
            Tuple of (all_valid, valid_refs, invalid_refs)
        """
        references = self.extract_references(text)
        
        valid_refs = []
        invalid_refs = []
        
        for ref in references:
            ref_lower = ref.lower()
            # Check if reference matches any PDF basename
            if any(ref_lower in pdf_name for pdf_name in self.pdf_basenames):
                valid_refs.append(ref)
            else:
                invalid_refs.append(ref)
        
        all_valid = len(invalid_refs) == 0
        return all_valid, valid_refs, invalid_refs
    
    def clean_invalid_references(self, text: str) -> str:
        """
        Remove invalid references from text.
        
        Args:
            text: Text containing references
            
        Returns:
            Text with invalid references removed
        """
        _, valid_refs, invalid_refs = self.validate_references(text)
        
        # Replace invalid references with a note
        cleaned_text = text
        for ref in invalid_refs:
            pattern = r'\(' + ref + r'(?:\s+pages\s+\d+-\d+)?\)'
            cleaned_text = re.sub(pattern, "[Reference not found in provided documents]", cleaned_text)
            
        return cleaned_text
