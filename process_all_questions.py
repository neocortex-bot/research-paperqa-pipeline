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

# Initialize OpenAI client
client = OpenAI()

async def generate_hypothetical_answer(question, model="gpt-5.4-nano-2026-03-17"):
    """Generate a hypothetical answer to the question using an LLM.
    This is the first step of HyDE (Hypothetical Document Embeddings).
    """
    print(f"Generating hypothetical answer for HyDE: {question}")
    
    try:
        # Use a specific prompt to guide the model to generate a hypothetical answer
        prompt = f"""Based on your knowledge, write a concise hypothetical answer to this question. 
        The answer should be factual and informative, but doesn't need citations.
        
        Question: {question}
        
        Hypothetical Answer:"""
        
        # gpt-5.x models use max_completion_tokens instead of max_tokens
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }
        if model.startswith("gpt-5"):
            kwargs["max_completion_tokens"] = 300
        else:
            kwargs["max_tokens"] = 300
        
        response = client.chat.completions.create(**kwargs)
        
        hypothetical_answer = response.choices[0].message.content.strip()
        print("Generated hypothetical answer for HyDE")
        return hypothetical_answer
    except Exception as e:
        print(f"Error generating hypothetical answer: {str(e)}")
        # Return a minimal answer if generation fails
        return f"This is a hypothetical answer about {question}"

async def query_paperqa(question, docs, model="gpt-5.4-nano-2026-03-17", max_sources=5, min_words=0, max_retries=5, pdf_dir="./pdf", use_hyde=True):
    """Query PaperQA with a question and get the answer.
    Optionally uses HyDE (Hypothetical Document Embeddings) to improve retrieval.
    
    Args:
        min_words: Minimum word count for the answer. PaperQA will try to reach this length if possible.
    """
    print(f"Processing question: {question}")
    if min_words > 0:
        print(f"Requested minimum word count: {min_words}")
    
    settings = Settings()
    settings.llm = "gpt-5.4-nano-2026-03-17"  # Force using gpt5.4 nano model
    settings.summary_llm = "gpt-5.4-nano-2026-03-17"  # Set summary_llm to use gpt5.4 nano model
    settings.agent.agent_llm = "gpt-5.4-nano-2026-03-17"  # Set agent_llm to use gpt5.4 nano model
    
    # Set answer length based on min_words
    if min_words > 0:
        settings.answer.answer_length = f"at least {min_words} words, but can be longer if more information is available"
    
    settings.answer.answer_max_sources = max_sources
    
    # Implement HyDE approach if enabled
    if use_hyde:
        try:
            print("Using HyDE approach for improved retrieval...")
            # Step 1: Generate a hypothetical answer
            hypothetical_answer = await generate_hypothetical_answer(question, model=model)
            
            # Step 2: Combine the original question with the hypothetical answer for retrieval
            hyde_query = f"Question: {question}\n\nHypothetical Answer: {hypothetical_answer}"
            print("Created HyDE query by combining question with hypothetical answer")
            
            # We'll use this enhanced query for retrieval, but keep the original question for the final answer
            retrieval_question = hyde_query
        except Exception as e:
            print(f"Error in HyDE implementation: {str(e)}")
            print("Falling back to standard retrieval...")
            retrieval_question = question
    else:
        # Standard approach without HyDE
        retrieval_question = question
    
    # Add strong instructions to prevent using reference sections and improve citations
    enhanced_question = f"{retrieval_question}\n\nCRITICAL INSTRUCTION: You must ONLY cite from the main content of papers. NEVER cite from reference lists, bibliographies, or table of contents. These sections typically appear at the end of papers and contain only citations to other works, not actual content. If a citation points to pages 15-17 or higher in a paper, double-check that it's not from a reference section. Any citation from a reference section will be considered invalid and will result in an incorrect answer."
    
    # Models to try in case of rate limits, in order of preference
    fallback_models = [
        "gpt-5.4-nano-2026-03-17",
        "gpt-4o-mini-2024-07-18"
    ]

    # Initialize reference validator
    validator = ReferenceValidator(pdf_dir=pdf_dir)
    
    # Initialize retry counter and backoff time
    retry_count = 0
    base_backoff = 2  # seconds
    
    while retry_count <= max_retries:
        try:
            # Try to get answer with current model
            print(f"Querying with model: {settings.llm}")
            # Create query instance using the enhanced question
            session = await asyncio.wait_for(docs.aquery(enhanced_question, settings=settings), timeout=60)
            
            # When using HyDE, we need to make sure the output doesn't include the hypothetical answer
            if use_hyde:
                # Replace the enhanced question (which includes the hypothetical answer) with just the original question
                answer_text = str(session).replace(f"Question: {enhanced_question}\n\n", f"Question: {question}\n\n")
            else:
                answer_text = str(session).replace(f"Question: {enhanced_question}\n\n", "")
                
            clean_answer = answer_text.split("References", 1)[0].strip()
            
            # Validate references in the answer
            full_answer = str(session)
            all_valid, valid_refs, invalid_refs = validator.validate_references(full_answer)
            
            if not all_valid:
                print(f"Warning: Found {len(invalid_refs)} invalid references: {', '.join(invalid_refs)}")
                print(f"Valid references: {', '.join(valid_refs) if valid_refs else 'None'}")
                
                # Clean invalid references from the answer
                cleaned_answer = validator.clean_invalid_references(full_answer)
                # Extract clean answer part again
                cleaned_text = cleaned_answer.replace(f"Question: {question}\n\n", "")
                clean_answer = cleaned_text.split("References", 1)[0].strip()
                
                return clean_answer, cleaned_answer
            
            return clean_answer, str(session)
            
        except Exception as e:
            error_str = str(e).lower()
            retry_count += 1
            
            # Check if we've reached max retries
            if retry_count > max_retries:
                print(f"Max retries ({max_retries}) reached. Giving up.")
                raise
            
            # Calculate backoff time with jitter
            jitter = random.uniform(0.1, 0.5)  # 10-50% jitter
            backoff_time = (base_backoff ** retry_count) + jitter
            
            # Check for rate limit or connection errors
            if "rate limit" in error_str or "too many requests" in error_str:
                print(f"Rate limit hit. Trying fallback model and backoff.")
                # Try next model in fallback list
                current_model_index = fallback_models.index(settings.llm) if settings.llm in fallback_models else -1
                if current_model_index < len(fallback_models) - 1:
                    settings.llm = fallback_models[current_model_index + 1]
                    print(f"Falling back to {settings.llm}")
                
            elif "connection" in error_str or "timeout" in error_str or "network" in error_str:
                print(f"Connection error. Retrying in {backoff_time:.2f} seconds.")
            else:
                print(f"Error: {e}. Retrying in {backoff_time:.2f} seconds.")
            
            # Wait before retrying
            print(f"Retry {retry_count}/{max_retries} after {backoff_time:.2f}s")
            await asyncio.sleep(backoff_time)

async def main():
    # Set up argument parser with detailed help
    parser = argparse.ArgumentParser(
        description="Process research questions using PaperQA with PDF documents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Show defaults in help text
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
    model_group.add_argument('--max-sources', type=int, default=5, 
                          help="Maximum number of document sources to use per question")
    model_group.add_argument('--min-words', type=int, default=0,
                          help="Minimum word count for the answer. PaperQA will try to reach this length if possible based on available sources.")
    
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
    max_sources = args.max_sources
    start_question = args.start
    end_question = args.end
    parallel_processing = args.parallel
    max_concurrent = args.max_concurrent
    use_hyde = args.use_hyde
    
    print(f"\nSelected model: {model}")
    print(f"Maximum sources per question: {max_sources}")
    print(f"Question range: {start_question or 'first'} to {end_question or 'last'}")
    print(f"Parallel processing: {'Yes' if parallel_processing else 'No'}")
    if parallel_processing:
        print(f"Maximum concurrent questions: {max_concurrent}")
    print(f"PDF directory: {pdf_dir}")
    print(f"Vector storage directory: {vector_storage_dir}")
    print(f"Questions file: {questions_json}")
    print(f"Output CSV: {output_csv}")
    print(f"Use HyDE: {use_hyde}")
    print("")
    
    # Single ad-hoc question mode: do not write to CSV
    if args.question:
        ad_hoc_question = args.question.strip()
        print("\nSingle-question mode: will NOT write to CSV.")
        
        # Get PDF files
        pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {pdf_dir}")
            return
        
        print(f"Found {len(pdf_files)} PDF files")
        
        # Initialize vector storage system and load docs
        print("\nInitializing vector storage system...")
        vector_storage = PDFVectorStorage(pdf_dir=pdf_dir, storage_dir=vector_storage_dir)
        try:
            docs = vector_storage.get_docs()
            print("Using existing embeddings from storage.")
        except RuntimeError:
            print("Creating/updating embeddings asynchronously...")
            docs = await vector_storage.get_docs_async()
        
        # Query PaperQA once and print results
        try:
            clean_answer, full_response = await query_paperqa(
                ad_hoc_question,
                docs,
                model=model,
                max_sources=max_sources,
                min_words=args.min_words,
                pdf_dir=pdf_dir,
                use_hyde=use_hyde
            )
            print("\nAnswer:")
            print(clean_answer)
            print("\nFull response:")
            print(full_response)
        except Exception as e:
            print(f"Error processing ad-hoc question: {str(e)}")
        
        return
    
    # Convert start/end question to strings for later comparison
    if start_question is not None:
        start_question = str(start_question)
    if end_question is not None:
        end_question = str(end_question)
    
    # Set default values if empty
    start_question = int(start_question) if start_question else None
    end_question = int(end_question) if end_question else None
    
    # Load questions
    with open(questions_json, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"Loaded {len(questions)} questions from {questions_json}")
    
    # Get PDF files
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Use vector storage to get embeddings
    print("\nInitializing vector storage system...")
    vector_storage = PDFVectorStorage(pdf_dir=pdf_dir, storage_dir=vector_storage_dir)
    try:
        # Try to get existing embeddings first (faster if available)
        docs = vector_storage.get_docs()
        print("Using existing embeddings from storage.")
    except RuntimeError:
        # If embeddings need to be created/updated, use the async version
        print("Creating/updating embeddings asynchronously...")
        docs = await vector_storage.get_docs_async()
    
    # Check if CSV file exists
    file_exists = os.path.isfile(output_csv)
    print(f"CSV file {'exists' if file_exists else 'will be created'}: {output_csv}")
    
    # PDF files are already processed by the vector storage system
    # No need to add them again
    
    # Process questions based on range
    question_keys = sorted([int(k) if k.isdigit() else k for k in questions.keys()])
    question_keys = [str(k) for k in question_keys]  # Convert back to strings
    
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
        print(f"\nProcessing question {q_num}: {question}")
        
        # Check if the question has already been answered
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
            
        # Query PaperQA
        try:
            clean_answer, full_response = await query_paperqa(
                question, 
                docs, 
                model=model,
                max_sources=max_sources,
                min_words=args.min_words,
                pdf_dir=pdf_dir,
                use_hyde=use_hyde  # Use HyDE if enabled
            )
            
            # Save to CSV (always append) with pipe delimiter
            with open(output_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                # Only write header if file doesn't exist
                if not os.path.getsize(output_csv) if os.path.exists(output_csv) else 0:
                    writer.writerow(['Question Number', 'Question', 'Answer'])
                
                # Only write if question doesn't already exist
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
            time.sleep(30)  # Longer delay after error

        print("\n" + "-"*80)
        
        return q_num
    
    # Create CSV file if it doesn't exist
    if not os.path.exists(output_csv):
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='|')
            writer.writerow(['Question Number', 'Question', 'Answer'])
    
    # Process questions either in parallel or sequentially
    if parallel_processing:
        print(f"\nProcessing {len(question_keys)} questions in parallel (max {max_concurrent} concurrent)...")
        
        # Calculate optimal delay between batches based on rate limits
        # We'll be more conservative with the rate limits
        batch_size = max_concurrent
        batches_needed = math.ceil(len(question_keys) / batch_size)
        
        # Estimate time needed with rate limiting
        estimated_seconds = batches_needed * (60 / 200) * batch_size  # More conservative rate limit
        estimated_minutes = estimated_seconds / 60
        print(f"Estimated processing time: {estimated_minutes:.2f} minutes")
        
        # Process in batches with controlled concurrency
        for i in range(0, len(question_keys), batch_size):
            batch = question_keys[i:i+batch_size]
            print(f"\nProcessing batch {i//batch_size + 1}/{batches_needed} with {len(batch)} questions...")
            
            # Create tasks for this batch
            tasks = [process_question(q_num) for q_num in batch]
            
            try:
                # Start time for rate limiting
                batch_start = time.time()
                
                # Wait for all tasks in this batch to complete
                await asyncio.gather(*tasks)
                
                # Calculate time taken and add delay if needed to respect rate limits
                batch_duration = time.time() - batch_start
                target_duration = (len(batch) / 200) * 60  # Target seconds based on 200 RPM (more conservative)
                
                if batch_duration < target_duration:
                    delay = target_duration - batch_duration
                    print(f"Rate limit protection: Waiting {delay:.2f} seconds before next batch...")
                    time.sleep(delay)
                
                # Always add a small delay between batches to avoid connection issues
                additional_delay = 5 + (random.random() * 5)  # 5-10 seconds
                print(f"Adding {additional_delay:.2f} seconds delay between batches...")
                time.sleep(additional_delay)
                    
            except Exception as e:
                print(f"Error in batch {i//batch_size + 1}: {str(e)}")
                print("Continuing with next batch after a delay...")
                time.sleep(30)  # Longer delay after error
    else:
        print(f"\nProcessing {len(question_keys)} questions sequentially...")
        # Process questions sequentially
        for q_num in question_keys:
            try:
                await process_question(q_num)
                # Add a small delay between questions
                delay = 5 + (random.random() * 5)  # 5-10 seconds
                print(f"Adding {delay:.2f} seconds delay between questions...")
                time.sleep(delay)
            except Exception as e:
                print(f"Error processing question {q_num}: {str(e)}")
                print("Continuing with next question after a delay...")
                time.sleep(30)  # Longer delay after error

    print(f"\nAll questions processed. Results saved to {output_csv}")

if __name__ == "__main__":
    asyncio.run(main())
