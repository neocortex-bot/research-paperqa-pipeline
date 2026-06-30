# Research PaperQA Pipeline

An automated, general-purpose literature review pipeline using **PaperQA** with enhanced PDF preprocessing, reference section filtering, HyDE (Hypothetical Document Embeddings) retrieval, and structured CSV output.

**Use for any research topic** — just swap PDFs and questions.

## Features

- 🔍 **Automated PDF ingestion** — Extracts text from PDFs, builds vector embeddings
- 🧹 **Reference section filtering** — Multi-strategy detection (headers + numbered entries + author-year patterns) to automatically remove reference/bibliography pages from RAG context
- 💡 **HyDE retrieval** — Generates hypothetical answers to improve document retrieval relevance (clean extraction — no "Hypothetical Answer:" pollution in output)
- 📊 **Structured CSV output** — Saves Q&A results to pipe-delimited CSV
- ⚡ **Incremental embeddings** — Only re-processes new/changed PDFs; saves progress per-file
- 🔄 **Parallel processing** — Optional concurrent question processing with rate-limit handling
- 🎯 **PDF keyword filtering** — Only embed relevant PDFs with `--pdf-keywords`
- 🧠 **Custom PaperQA prompts** — Removes paranoid "I cannot answer" behavior; tuned for medical/scientific literature

## Project Structure

```
research-paperqa-pipeline/
├── process_all_questions.py   # Main entry point — CLI for question answering
├── pdf_vector_storage.py      # PDF embedding + reference section filtering logic
├── reference_validator.py     # Validates citations against available PDFs
├── research_questions.json    # Example research questions
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
├── .gitignore                 # Files to exclude from git
└── command.txt                # Example usage command
```

## Requirements

- Python 3.10+
- OpenAI API key (or compatible endpoint)
- PDF files for analysis in a `./pdf/` directory

## Installation

```bash
# Clone the repo
git clone https://github.com/neocortex-bot/research-paperqa-pipeline.git
cd research-paperqa-pipeline

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Usage

### 1. Prepare PDFs

Place your PDF files in a `./pdf/` directory. The system will:
- Automatically detect and filter out reference/bibliography sections (94%+ detection rate)
- Build vector embeddings incrementally (only re-processes new/changed files)

**Tip:** Use `--pdf-keywords` to only embed relevant PDFs by filename:
```bash
python process_all_questions.py --pdf-keywords ppcm cardiomyopathy peripartum
```

### 2. Prepare Questions

Edit `research_questions.json` with your research questions:
```json
{
  "1": "Your first research question?",
  "2": "Your second research question?"
}
```

### 3. Run the Pipeline

```bash
# Process a single question (e.g., question 5)
python process_all_questions.py --start=5 --end=5

# Process a range of questions
python process_all_questions.py --start=1 --end=10

# Process questions in parallel
python process_all_questions.py --start=1 --end=10 --parallel --max-concurrent=3
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `gpt-5.4-nano-2026-03-17` | OpenAI model for answering |
| `--max-sources` | 15 | Max documents to cite per question |
| `--evidence-k` | 40 | Evidence chunks to retrieve (higher = better coverage) |
| `--min-words` | 500 | Minimum answer length |
| `--use-hyde` / `--no-hyde` | HyDE on | Enable/disable HyDE retrieval |
| `--pdf-keywords` | none | Only embed PDFs with these keywords in filename |
| `--pdf-exclude` | none | Exclude PDFs with these keywords in filename |
| `--start` / `--end` | all | Question number range |
| `--parallel` | off | Enable parallel processing |
| `--max-concurrent` | 1 | Max concurrent workers |
| `--question` | none | Single ad-hoc question (no CSV output) |

### One-shot query (no CSV)
```bash
python process_all_questions.py --question "What is the role of MRI in PPCM diagnosis?"
```

## How It Works

### Reference Filtering
The `preprocess_pdf()` function uses a **multi-strategy scoring system**:
1. **Header detection** — matches standalone "References", "Bibliography", etc.
2. **Word-boundary fallback** — catches headers like "7. REFERENCES"
3. **Numbered entry patterns** — detects `[1]`, `[2]`, `1.`, `2.` patterns
4. **Author-year patterns** — detects "Smith (2020)", "et al." citations

### HyDE (clean mode)
The pipeline generates a hypothetical answer, extracts key medical terms, and appends them as "Key aspects:" — improving embedding similarity WITHOUT polluting the answerer's prompt with "Hypothetical Answer:" text.

### Custom Prompts
PaperQA's default `prompts.qa` contains "If the context provides insufficient information reply 'I cannot answer.'" This pipeline overrides it with a balanced prompt that:
- Removes the paranoid refusal instruction
- Recognizes position statements and consensus documents as authoritative
- Encourages evidence synthesis rather than refusal

## Output

Results are saved to `research_answers.csv` (semicolon-delimited format):

```
Question Number;Question;Answer
1;"What is...?";"Peripartum cardiomyopathy is..."
2;"What are...?";"LVEF <35% is..."
```

## Custom Model Support

For newer OpenAI models (gpt-5.x series), the script uses `max_completion_tokens` instead of the legacy `max_tokens` parameter.

## Tips for Optimal Results

1. **Filter relevant PDFs first** — Use `--pdf-keywords` to avoid embedding irrelevant documents
2. **Higher evidence_k for complex questions** — Start with 40, increase to 60 if answers miss key sources
3. **Review embeddings** — Existing embeddings are reused; to force re-embedding, delete `vector_storage/`
4. **Multiple topics** — Create separate question JSON files for different research topics

## License

MIT
