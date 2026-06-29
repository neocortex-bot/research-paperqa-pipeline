# Research PaperQA Pipeline

An automated literature review pipeline using **PaperQA** with enhanced PDF preprocessing, reference section filtering, HyDE (Hypothetical Document Embeddings) retrieval, and structured CSV output.

## Features

- 🔍 **Automated PDF ingestion** — Extracts text from PDFs, builds vector embeddings
- 🧹 **Reference section filtering** — Multi-strategy detection (headers + numbered entries + author-year patterns) to automatically remove reference/bibliography pages from RAG context
- 💡 **HyDE retrieval** — Generates hypothetical answers to improve document retrieval relevance
- 📊 **Structured output** — Saves Q&A results to pipe-delimited CSV
- ⚡ **Incremental embeddings** — Only re-processes new/changed PDFs; saves progress per-file
- 🔄 **Parallel processing** — Optional concurrent question processing with rate-limit handling

## Project Structure

```
research-paperqa-pipeline/
├── process_all_questions.py   # Main entry point — CLI for question answering
├── pdf_vector_storage.py      # PDF embedding + reference section filtering logic
├── reference_validator.py     # Validates citations against available PDFs
├── research_questions.json    # Example research questions
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
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
- Automatically detect and filter out reference/bibliography sections (94%+ detection rate across real-world PDFs)
- Build vector embeddings incrementally (only re-processes new/changed files)

### 2. Prepare Questions

Edit `research_questions.json` with your research questions in format:
```json
{
  "1": "Your first research question?",
  "2": "Your second research question?"
}
```

### 3. Run the Pipeline

```bash
# Process a single question
python process_all_questions.py --start=1 --end=1 --max-sources=8 --min-words=400

# Process a range of questions
python process_all_questions.py --start=1 --end=5 --max-sources=5 --min-words=200

# Process all questions in parallel
python process_all_questions.py --parallel --max-concurrent=3 --max-sources=8

# Ask an ad-hoc question (no CSV output)
python process_all_questions.py --question "What is the definition of PPCM?"
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `gpt-4o-mini` | OpenAI model for answering (`gpt-5.4-nano-2026-03-17`, `gpt-4o-mini`, etc.) |
| `--max-sources` | 5 | Max document sources per question |
| `--min-words` | 0 | Minimum word count for answers |
| `--start` / `--end` | all | Question number range |
| `--parallel` | off | Enable parallel processing |
| `--max-concurrent` | 1 | Max concurrent workers |
| `--use-hyde` / `--no-hyde` | HyDE on | Enable/disable HyDE retrieval |

## How Reference Filtering Works

The `preprocess_pdf()` function in `pdf_vector_storage.py` uses a **multi-strategy scoring system**:

1. **Header detection** (line-exact regex with `re.MULTILINE`) — matches standalone "References", "Bibliography", etc.
2. **Word-boundary fallback** — catches headers like "7. REFERENCES" or "References and Notes"
3. **Numbered entry patterns** — detects `[1]`, `[2]`, `1.`, `2.` patterns
4. **Author-year patterns** — detects `Smith (2020)`, `et al.` citations

Each page scores 0–10; pages with consistent high scores trigger filtering.

## Output

Results are saved to `research_answers.csv` (pipe-delimited format):

```
Question Number | Question | Answer
1 | "What is...?" | "Peripartum cardiomyopathy is..."
```

## Custom Model Support

For newer OpenAI models (gpt-5.x series), the script uses `max_completion_tokens` instead of the legacy `max_tokens` parameter. Configure the model via:

```bash
python process_all_questions.py --model=gpt-5.4-nano-2026-03-17
```

Or set in the script defaults.

## License

MIT
