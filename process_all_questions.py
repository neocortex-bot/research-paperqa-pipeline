import os
import time
import json
import csv
import random
import asyncio
import argparse
import math
from datetime import datetime
from paperqa import Docs, Settings
from dotenv import load_dotenv
from reference_validator import ReferenceValidator
from pdf_vector_storage import PDFVectorStorage
from reference_validator import ReferenceValidator
from openai import OpenAI

# Set the API key directly from the .env file
from dotenv import load_dotenv
load_dotenv()

# Initialize OpenAI client (for HyDE generation only)
client = OpenAI()

async def generate_hypothetical_answer(question, model="gpt-5.4-nano-2026-03-17"):
    """Generate a hypothetical answer to the question using an LLM.
    This is the first step of HyDE (Hypothetical Document Embeddings).
    """
    print(f"Generating hypothetical answer for HyDE: {question[:80]}...")
    
    try:
        prompt = f"""Based on your knowledge, write a few sentences that would appear in a MEDICAL CONSENSUS DOCUMENT or POSITION STATEMENT defining this condition or answering this question.

IMPORTANT: Use the EXACT LANGUAGE and FORMAT of medical position statements (e.g., ESC, AHA, ACC guidelines).
Use phrases like "We propose the following definition:", "is defined as", "diagnostic criteria include", "is characterized by".
Include specific numbers, thresholds, and cutoffs where relevant (e.g., LVEF <45%, >60mm, etc.).
Be precise and formal — this text will be used as a search query to find matching documents.

Question: {question}

Formal definition/criteria text:"""

        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,  # Lower temperature for more consistent/formal output
        }
        if model.startswith("gpt-5"):
            kwargs["max_completion_tokens"] = 500
        else:
            kwargs["max_tokens"] = 500
        
        response = client.chat.completions.create(**kwargs)
        hypothetical_answer = response.choices[0].message.content.strip()
        print(f"HyDE generated ({len(hypothetical_answer)} chars)")
        return hypothetical_answer
    except Exception as e:
        print(f"Error generating hypothetical answer: {str(e)}")
        return f"This is a hypothetical answer about {question}"

async def query_paperqa(question, docs, model="gpt-5.4-nano-2026-03-17", max_sources=15, 
                         min_words=500, max_retries=5, pdf_dir="./pdf", use_hyde=True,
                         evidence_k=60, mmr_lambda=0.7, embedding_model="text-embedding-3-small"):
    """Query PaperQA with a question and get the answer.
    
    Key improvements over original:
    - Removed paranoid "CRITICAL INSTRUCTION" that caused "I cannot answer"
    - Custom PaperQA prompts to encourage thorough answers
    - Higher evidence_k for better retrieval coverage
    - No relevance cutoff to include marginal evidence
    """
    print(f"Processing question: {question[:100]}...")
    if min_words > 0:
        print(f"Requested minimum word count: {min_words}")
    
    settings = Settings()
    settings.llm = model
    settings.summary_llm = model
    settings.agent.agent_llm = model
    
    # --- RETRIEVAL IMPROVEMENTS ---
    # Get more evidence chunks for better coverage across PDFs
    settings.answer.evidence_k = evidence_k
    # Don't discard marginally-relevant evidence
    settings.answer.evidence_relevance_score_cutoff = 0
    # Max sources the answer can cite
    settings.answer.answer_max_sources = max_sources
    # MMR diversity: 0.7 = balance between relevance and diversity across docs
    settings.texts_index_mmr_lambda = mmr_lambda
    # Local embedding model (free, no API costs, no rate limits)
    settings.embedding = embedding_model
    
    # --- ANSWER PROMPT OVERRIDE ---
    # PaperQA default prompts.qa contains: "If the context provides insufficient information reply \"I cannot answer.\""
    # This causes the model to refuse even when partial info exists. We override it.
    custom_qa_prompt = (
        "Answer the question below using the provided context.\n\n"
        "Context:\n\n{context}\n\n---\n\n"
        "Question: {question}\n\n"
        "Write a thorough, evidence-based answer. Synthesize information from multiple sources when possible. "
        "IMPORTANT: Position statements, consensus statements, and working group reports from "
        "international societies (ESC, AHA, ACC, etc.) ARE consensus documents. Phrases like "
        "\"we propose\" or \"we recommend\" in these documents represent the formal consensus position. "
        "Do not dismiss them as non-consensus just because of hedging language.\n\n"
        "For each statement, cite the source using citation keys like {example_citation}. "
        "Only cite from the context above. Write in the style of a scientific article.\n\n"
        "{prior_answer_prompt}Answer ({answer_length}):"
    )
    settings.prompts.qa = custom_qa_prompt
    
    # System prompt: expert tone, understands medical literature conventions
    settings.prompts.system = (
        "You are a medical research assistant synthesizing information from scientific papers. "
        "Your audience is a medical professional, so use precise terminology.\n\n"
        "GUIDELINES:\n"
        "- Position statements, clinical guidelines, and consensus documents from major cardiology "
        "societies (ESC, AHA, ACC) ARE the gold standard for 'international consensus.'\n"
        "- Do not over-interpret hedging language like 'we propose' or 'simplified definition' — "
        "these are standard rhetorical conventions in position papers.\n"
        "- Provide the best available answer based on the evidence in context. "
        "If the context provides partial information, synthesize it rather than refusing to answer.\n"
        "- Distinguish clearly between: (a) what is explicitly stated in the context, "
        "(b) what can be reasonably inferred, and (c) what is not addressed."
    )
    
    # Set answer length
    if min_words > 0:
        settings.answer.answer_length = f"at least {min_words} words, synthesizing all available evidence"
    else:
        settings.answer.answer_length = "comprehensive, using all available evidence"
    
    # --- HyDE APPROACH (clean - only for retrieval, not polluting answer) ---
    # HyDE improves retrieval by generating a hypothetical answer first,
    # extracting key medical terms from it, and appending them as Keywords
    # to the question. This improves embedding similarity WITHOUT putting
    # "Hypothetical Answer:" junk into the answerer's prompt.
    if use_hyde:
        try:
            print("Using HyDE for improved retrieval...")
            hypothetical_answer = await generate_hypothetical_answer(question, model=model)
            
            # Extract key medical concepts from HyDE for better retrieval
            # Use the key framing and terms from HyDE as search keywords
            # Take first ~300 chars of HyDE as the key conceptual framing
            hyde_snippet = hypothetical_answer[:300].strip()
            
            # Multi-angle query: conceptual (HyDE) + exact phrase (document language)
            # Extract the most document-like phrase from HyDE (first sentence with "definition" or "criteria")
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
            
            retrieval_question = f"{question}\n\nKey aspects: {hyde_snippet}\n\n{doc_like_phrase}"
        except Exception as e:
            print(f"Error in HyDE: {str(e)}. Falling back to standard retrieval.")
            retrieval_question = question
    else:
        retrieval_question = question
    
    # Models to try in case of rate limits
    fallback_models = [
        model,
        "gpt-4o-mini-2024-07-18"
    ]

    # Initialize reference validator
    validator = ReferenceValidator(pdf_dir=pdf_dir)
    
    # Initialize retry counter
    retry_count = 0
    base_backoff = 2
    
    while retry_count <= max_retries:
        try:
            print(f"Querying with model: {settings.llm} (evidence_k={evidence_k})")
            session = await asyncio.wait_for(
                docs.aquery(retrieval_question, settings=settings), 
                timeout=120  # Increased timeout for larger evidence_k
            )
            
            # Extract clean answer from PaperQA's PQASession
            full_answer = str(session)
            
            # PaperQA output format is typically:
            # Question: {question}
            # 
            # {answer_text}
            # 
            # References
            # ...
            #
            # OR sometimes the model includes the question in its response.
            # Clean approach: split on "References\n" to get answer body,
            # then remove any "Question:" prefix lines.
            
            # Split on References section
            parts = full_answer.split("\nReferences\n", 1)
            if len(parts) > 1:
                answer_body = parts[0]
            else:
                answer_body = full_answer
            
            # Remove "Question: ..." prefix and HyDE metadata from the answer
            # PaperQA output: "Question: {query}\n\n{answer}"
            # Split on first blank line after Question: prefix
            lines = answer_body.split("\n")
            
            # Find where the question header ends and answer begins
            # The header is: "Question: ..." + "Key aspects: ..." + optional doc_like_phrase
            # The answer starts after a blank line following all header lines
            answer_start = 0
            seen_content = False
            header_lines_count = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    if seen_content:
                        # Blank line after header content - answer starts next
                        answer_start = i + 1
                        break
                    continue
                if stripped.startswith("Question:"):
                    header_lines_count += 1
                    continue
                if stripped.startswith("Key aspects:"):
                    header_lines_count += 1
                    continue
                # Any other non-blank line before answer is still header
                if not seen_content:
                    header_lines_count += 1
                    seen_content = True
            
            if answer_start > 0 and answer_start < len(lines):
                clean_answer = "\n".join(lines[answer_start:]).strip()
            else:
                # Fallback: take everything after the first blank line following "Question:"
                clean_answer = answer_body.split("Question:", 1)[-1].strip()
                # If there's a blank line, take everything after it
                if "\n\n" in clean_answer:
                    clean_answer = clean_answer.split("\n\n", 1)[-1].strip()
            
            if not clean_answer or clean_answer == "I cannot answer.":
                print("WARNING: Answer is empty or 'I cannot answer.' - will retry with adjusted settings")
                # Don't raise - just note it
                pass
            
            # Validate references
            all_valid, valid_refs, invalid_refs = validator.validate_references(full_answer)
            
            if not all_valid:
                print(f"Warning: Found {len(invalid_refs)} invalid references: {', '.join(invalid_refs)}")
                print(f"Valid references: {', '.join(valid_refs) if valid_refs else 'None'}")
                
                cleaned_answer = validator.clean_invalid_references(full_answer)
                # Extract clean answer part again
                parts = cleaned_answer.split("\nReferences\n", 1)
                if len(parts) > 1:
                    answer_body = parts[0]
                else:
                    answer_body = cleaned_answer
                lines = answer_body.split("\n")
                clean_lines = [l for l in lines if not l.startswith("Question:")]
                clean_answer = "\n".join(clean_lines).strip()
                
                return clean_answer, cleaned_answer
            
            return clean_answer, str(session)
            
        except Exception as e:
            error_str = str(e).lower()
            retry_count += 1
            
            if retry_count > max_retries:
                print(f"Max retries ({max_retries}) reached. Giving up.")
                raise
            
            jitter = random.uniform(0.1, 0.5)
            backoff_time = (base_backoff ** retry_count) + jitter
            
            if "rate limit" in error_str or "too many requests" in error_str:
                print(f"Rate limit hit. Trying fallback model and backoff.")
                current_model_index = fallback_models.index(settings.llm) if settings.llm in fallback_models else -1
                if current_model_index < len(fallback_models) - 1:
                    settings.llm = fallback_models[current_model_index + 1]
                    print(f"Falling back to {settings.llm}")
                
            elif "connection" in error_str or "timeout" in error_str or "network" in error_str:
                print(f"Connection error. Retrying in {backoff_time:.2f} seconds.")
            else:
                print(f"Error: {e}. Retrying in {backoff_time:.2f} seconds.")
            
            print(f"Retry {retry_count}/{max_retries} after {backoff_time:.2f}s")
            await asyncio.sleep(backoff_time)


async def main():
    parser = argparse.ArgumentParser(
        description="Process research questions using PaperQA with PDF documents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output arguments
    io_group = parser.add_argument_group('Input/Output Options')
    io_group.add_argument('--questions-json', type=str, default='research_questions.json', 
                       help="Path to JSON file containing questions in format {question_number: question_text}")
    io_group.add_argument('--output-csv', type=str, default='research_answers.csv', 
                       help="Path to output CSV file for answers (pipe-delimited format)")
    io_group.add_argument('--pdf-dir', type=str, default='./pdf', 
                       help="Directory containing PDF files to analyze")
    io_group.add_argument('--vector-storage-dir', type=str, default='./vector_storage', 
                       help="Directory for storing vector embeddings")
    io_group.add_argument('--question', type=str,
                       help="Process a single ad-hoc question (will not write to CSV)")
    
    # Model and retrieval settings
    model_group = parser.add_argument_group('Model and Retrieval Settings')
    model_group.add_argument('--model', type=str, 
                          choices=['gpt-5.4-nano-2026-03-17', 'gpt-4o-mini-2024-07-18'],
                          default='gpt-5.4-nano-2026-03-17', 
                          help="OpenAI model to use for question answering.")
    model_group.add_argument('--embedding-model', type=str,
                          default='text-embedding-3-large',
                          help="Embedding model for vector search. Use 'text-embedding-3-small' or 'text-embedding-3-large' for OpenAI API, "
                               "or any sentence-transformers model (e.g. BAAI/bge-base-en-v1.5) for local embedding.")
    model_group.add_argument('--max-sources', type=int, default=15, 
                          help="Maximum number of document sources to cite per question")
    model_group.add_argument('--evidence-k', type=int, default=60,
                          help="Number of evidence chunks to retrieve per question. Higher = better coverage but more tokens.")
    model_group.add_argument('--mmr-lambda', type=float, default=0.7,
                          help="MMR diversity lambda (0-1). 0=max diverse, 1=max relevant. 0.7 is sweet spot.")
    model_group.add_argument('--min-words', type=int, default=500,
                          help="Minimum word count for the answer. PaperQA will try to reach this length if possible.")
    
    # Retrieval enhancement options
    retrieval_group = parser.add_argument_group('Retrieval Enhancement Options')
    retrieval_group.add_argument('--use-hyde', dest='use_hyde', action='store_true', 
                             help="Use HyDE (Hypothetical Document Embeddings) for improved retrieval")
    retrieval_group.add_argument('--no-hyde', dest='use_hyde', action='store_false', 
                             help="Disable HyDE retrieval enhancement")
    
    # Processing control
    processing_group = parser.add_argument_group('Processing Control')
    processing_group.add_argument('--parallel', action='store_true', 
                              help="Process questions in parallel using multiple workers")
    processing_group.add_argument('--max-concurrent', type=int, default=1, 
                              help="Maximum number of concurrent workers for processing questions when using parallel mode")
    processing_group.add_argument('--start', type=int, default=None, 
                              help="Starting question number (inclusive)")
    processing_group.add_argument('--end', type=int, default=None, 
                              help="Ending question number (inclusive). If not specified, process all questions.")
    
    # Set defaults
    parser.set_defaults(use_hyde=True)
    
    args = parser.parse_args()
    
    # Configuration from arguments
    questions_json = args.questions_json
    output_csv = args.output_csv
    pdf_dir = args.pdf_dir
    vector_storage_dir = args.vector_storage_dir
    model = args.model
    embedding_model = args.embedding_model
    max_sources = args.max_sources
    evidence_k = args.evidence_k
    mmr_lambda = args.mmr_lambda
    start_question = args.start
    end_question = args.end
    parallel_processing = args.parallel
    max_concurrent = args.max_concurrent
    use_hyde = args.use_hyde
    
    print(f"\n{'='*60}")
    print(f"  Research PaperQA Pipeline")
    print(f"{'='*60}")
    print(f"  Selected model: {model}")
    print(f"  Embedding model: {embedding_model}")
    print(f"  Max sources per question: {max_sources}")
    print(f"  Evidence chunks (k): {evidence_k}")
    print(f"  MMR diversity lambda: {mmr_lambda}")
    print(f"  Question range: {start_question or 'first'} to {end_question or 'last'}")
    print(f"  HyDE retrieval: {'Yes' if use_hyde else 'No'}")
    print(f"  Parallel processing: {'Yes' if parallel_processing else 'No'}")
    if parallel_processing:
        print(f"  Max concurrent questions: {max_concurrent}")
    print(f"  PDF directory: {pdf_dir}")
    print(f"  Vector storage: {vector_storage_dir}")
    print(f"  Questions file: {questions_json}")
    print(f"  Output CSV: {output_csv}")
    print(f"{'='*60}\n")
    
    # Single ad-hoc question mode
    if args.question:
        ad_hoc_question = args.question.strip()
        print("\nSingle-question mode: will NOT write to CSV.")
        
        # Initialize vector storage and load docs
        print("\nInitializing vector storage system...")
        vector_storage = PDFVectorStorage(pdf_dir=pdf_dir, storage_dir=vector_storage_dir, embedding_model=embedding_model)
        
        try:
            docs = vector_storage.get_docs()
            print("Using existing embeddings from storage.")
        except RuntimeError:
            print("Creating/updating embeddings asynchronously...")
            docs = await vector_storage.get_docs_async()
        
        try:
            clean_answer, full_response = await query_paperqa(
                ad_hoc_question,
                docs,
                model=model,
                max_sources=max_sources,
                min_words=args.min_words,
                pdf_dir=pdf_dir,
                use_hyde=use_hyde,
                evidence_k=evidence_k,
                mmr_lambda=mmr_lambda,
                embedding_model=embedding_model
            )
            print("\nAnswer:")
            print(clean_answer)
            print("\nFull response:")
            print(full_response)
        except Exception as e:
            print(f"Error processing ad-hoc question: {str(e)}")
        
        return
    
    # Convert start/end question for filtering
    if start_question is not None:
        start_question = str(start_question)
    if end_question is not None:
        end_question = str(end_question)
    
    start_question = int(start_question) if start_question else None
    end_question = int(end_question) if end_question else None
    
    # Load questions
    with open(questions_json, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"Loaded {len(questions)} questions from {questions_json}")
    
    # Use vector storage to get embeddings
    print("\nInitializing vector storage system...")
    vector_storage = PDFVectorStorage(pdf_dir=pdf_dir, storage_dir=vector_storage_dir, embedding_model=embedding_model)
    
    try:
        docs = vector_storage.get_docs()
        print("Using existing embeddings from storage.")
    except RuntimeError:
        print("Creating/updating embeddings asynchronously...")
        docs = await vector_storage.get_docs_async()
    
    # Process questions based on range
    question_keys = sorted([int(k) if k.isdigit() else k for k in questions.keys()])
    question_keys = [str(k) for k in question_keys]
    
    # Apply range filtering
    if start_question is not None:
        question_keys = [k for k in question_keys if k.isdigit() and int(k) >= start_question]
    if end_question is not None:
        question_keys = [k for k in question_keys if k.isdigit() and int(k) <= end_question]
    
    # Check if CSV file exists
    file_exists = os.path.isfile(output_csv)
    print(f"CSV file {'exists' if file_exists else 'will be created'}: {output_csv}")
    
    if question_keys:
        print(f"Will process {len(question_keys)} questions: {', '.join(question_keys)}")
    else:
        print("No questions match the specified range.")
        return
    
    # Function to process a single question
    async def process_question(q_num):
        if q_num not in questions:
            print(f"Question {q_num} not found in questions file.")
            return None
            
        question = questions[q_num]
        print(f"\n{'='*60}")
        print(f"Processing question {q_num}: {question[:120]}")
        print(f"{'='*60}")
        
        # Check if already answered
        question_exists = False
        if os.path.exists(output_csv):
            with open(output_csv, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='|')
                for row in reader:
                    if len(row) > 0 and row[0] == str(q_num):
                        question_exists = True
                        break
        
        if question_exists:
            print(f"Question {q_num} already exists in {output_csv}, skipping...")
            return q_num
            
        try:
            clean_answer, full_response = await query_paperqa(
                question, 
                docs, 
                model=model,
                max_sources=max_sources,
                min_words=args.min_words,
                pdf_dir=pdf_dir,
                use_hyde=use_hyde,
                evidence_k=evidence_k,
                mmr_lambda=mmr_lambda,
                embedding_model=embedding_model
            )
            
            # Save to CSV (append)
            with open(output_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                if not os.path.getsize(output_csv) if os.path.exists(output_csv) else 0:
                    writer.writerow(['Question Number', 'Question', 'Answer'])
                
                if not question_exists:
                    writer.writerow([q_num, question, clean_answer])
                    print(f"Answer for question {q_num} saved to {output_csv}")
                else:
                    print(f"Question {q_num} already exists in {output_csv}, skipping...")
            
            print(f"Answer saved to {output_csv}")
            print("\nFull response:")
            print(full_response)
            print("\n" + "-"*80)
            
            return q_num
        except Exception as e:
            print(f"Error processing question {q_num}: {str(e)}")
            print("Continuing with next question after a delay...")
            time.sleep(30)

        print("\n" + "-"*80)
        return q_num
    
    # Create CSV if it doesn't exist
    if not os.path.exists(output_csv):
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='|')
            writer.writerow(['Question Number', 'Question', 'Answer'])
    
    # Process questions
    if parallel_processing:
        print(f"\nProcessing {len(question_keys)} questions in parallel (max {max_concurrent} concurrent)...")
        
        batch_size = max_concurrent
        batches_needed = math.ceil(len(question_keys) / batch_size)
        
        for i in range(0, len(question_keys), batch_size):
            batch = question_keys[i:i+batch_size]
            print(f"\nProcessing batch {i//batch_size + 1}/{batches_needed} with {len(batch)} questions...")
            
            tasks = [process_question(q_num) for q_num in batch]
            
            try:
                batch_start = time.time()
                await asyncio.gather(*tasks)
                batch_duration = time.time() - batch_start
                print(f"Batch completed in {batch_duration:.2f}s")
                
                additional_delay = 3 + (random.random() * 3)
                print(f"Adding {additional_delay:.2f}s delay between batches...")
                time.sleep(additional_delay)
                    
            except Exception as e:
                print(f"Error in batch {i//batch_size + 1}: {str(e)}")
                time.sleep(30)
    else:
        print(f"\nProcessing {len(question_keys)} questions sequentially...")
        for q_num in question_keys:
            try:
                await process_question(q_num)
                delay = 3 + (random.random() * 3)
                print(f"Adding {delay:.2f}s delay between questions...")
                time.sleep(delay)
            except Exception as e:
                print(f"Error processing question {q_num}: {str(e)}")
                time.sleep(30)

    print(f"\n{'='*60}")
    print(f"  All questions processed. Results saved to {output_csv}")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())
