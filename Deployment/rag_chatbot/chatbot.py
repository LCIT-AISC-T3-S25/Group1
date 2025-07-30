
"""
Production-ready RAG Chatbot with Enhanced Truncation Handling
Updated to properly detect and handle incomplete responses
"""

import torch
import numpy as np
import gc
import os
import logging
import re

from load_system import load_rag_system

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Global variables to store loaded components
embedder = None
query_rewriter = None
answer_generator = None
model_type = None
faiss_index = None
passages_list = None
passage_ids = None
clean_passages_df = None

def initialize_system():
    """Initialize the complete RAG system"""
    global embedder, query_rewriter, answer_generator, model_type
    global faiss_index, passages_list, passage_ids, clean_passages_df

    try:
        logger.info("Initializing memory-optimized RAG system...")

        # Clear any existing models
        cleanup_memory()

        # Load data
        if not load_data():
            logger.error("Failed to load data")
            return False

        # Load models
        if not load_models():
            logger.error("Failed to load models")
            return False

        logger.info("RAG system initialized successfully")
        return True

    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        return False

def cleanup_memory():
    """Clean up existing models and memory"""
    global embedder, query_rewriter, answer_generator

    for var_name in ['query_model', 'answer_model', 'query_rewriter', 'answer_generator']:
        if var_name in globals():
            del globals()[var_name]

    embedder = None
    query_rewriter = None
    answer_generator = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_data():
    """Load RAG system data"""
    global faiss_index, passages_list, passage_ids, clean_passages_df

    try:
        logger.info("Loading RAG data...")

        # Load from checkpoint
        checkpoint_path = 'load_system.py'
        if os.path.exists(checkpoint_path):
            # Get the loaded data
            result = load_rag_system()
            faiss_index, passages_list, passage_ids, vectorstore, config = result

            logger.info(f"Loaded {len(passages_list)} passages")
            return True

        else:
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return False

    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        return False

def _hf_login(hf_token):
    import huggingface_hub
    try:
        huggingface_hub.login(token=hf_token)
        logger.info("✅ HuggingFace login successful")
        return True
    except Exception as e:
        logger.warning(f"HF login failed: {e}")
        return False

def _load_embedder():
    logger.info("Loading sentence embedder...")
    global embedder
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    logger.info("✅ Sentence embedder loaded successfully")

def _load_gemma_rewriter(hf_token):
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
    logger.info("Attempting to load Gemma-2-2b-it...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", token=hf_token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        token=hf_token,
        trust_remote_code=True
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False
    )

def _load_fallback_rewriter():
    logger.info("Loading DistilGPT2 as fallback query rewriter...")
    return pipeline(
        "text-generation", 
        model="distilgpt2",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.7,
        return_full_text=False
    )

def _load_answer_generator():
    global model_type
    try:
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        model_type = "TinyLlama"
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )
    except Exception:
        logger.warning("TinyLlama failed, loading DistilGPT2 answer generator")
        model_type = "DistilGPT2"
        return pipeline(
            "text-generation", 
            model="distilgpt2",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False
        )

def load_models():
    global embedder, query_rewriter, answer_generator, model_type
    try:
        # ... imports remain same ...
        logger.info(f"Transformers version: {transformers.__version__}")
        
        hf_token = os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
        if hf_token:
            logger.info("Hugging Face token found, logging in...")
            _hf_login(hf_token)
        else:
            logger.warning("No Hugging Face token found")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _load_embedder()
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Load query rewriter
        query_rewriter = None
        major, minor = map(int, transformers.__version__.split('.')[:2])
        if hf_token and (major > 4 or (major == 4 and minor >= 38)):
            try:
                query_rewriter = _load_gemma_rewriter(hf_token)
                logger.info("✅ Gemma query rewriter loaded")
            except Exception as e:
                logger.warning(f"Gemma loading failed: {str(e)}")
        
        if query_rewriter is None:
            try:
                query_rewriter = _load_fallback_rewriter()
                logger.info("✅ Fallback query rewriter loaded")
            except Exception as e:
                logger.warning(f"Fallback rewriter failed: {e}")

        # Load answer generator
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        answer_generator = _load_answer_generator()
        logger.info(f"✅ Answer generator loaded ({model_type})")
        return True

    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        return False

def is_sentence_complete(text):
    """
    Enhanced sentence completion detection
    Returns True if the text ends with a complete sentence
    """
    if not text or not text.strip():
        return False

    text = text.strip()

    # Check for sentence-ending punctuation
    if not re.search(r'[.!?]["\']?\s*$', text):
        return False

    # Check for incomplete patterns
    incomplete_patterns = [
        r'\b(the|a|an|and|or|but|in|on|at|to|for|with|by)\s*$',  # Ends with articles/prepositions
        r'\b(is|are|was|were|has|have|had|will|would|could|should)\s*$',  # Ends with auxiliary verbs
        r'\b(such|like|including|especially|particularly)\s*$',  # Ends with transition words
        r'\b(because|since|although|while|when|where|which|that)\s*$',  # Ends with conjunctions
        r'\b(more|less|most|least|very|quite|rather|extremely)\s*$',  # Ends with modifiers
        r'\b(can|may|might|must|should|would|could)\s*$',  # Ends with modal verbs
    ]

    for pattern in incomplete_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False

    # Check for balanced parentheses and quotes
    open_parens = text.count('(') - text.count(')')
    open_quotes = text.count('"') % 2
    open_single_quotes = text.count("'") % 2

    if open_parens != 0 or open_quotes != 0 or open_single_quotes != 0:
        return False

    # Check minimum sentence length (avoid single word "sentences")
    sentences = re.split(r'[.!?]+', text)
    if sentences:
        last_sentence = sentences[-2] if len(sentences) > 1 else sentences[0]  # -2 because split creates empty last element
        if len(last_sentence.strip().split()) < 3:  # Less than 3 words
            return False

    return True

def detect_truncation_type(text):
    """
    Detect the type of truncation in the response
    Returns: 'complete', 'mid_sentence', 'mid_word', 'abrupt_end'
    """
    if not text or not text.strip():
        return 'abrupt_end'

    text = text.strip()

    # Check if complete
    if is_sentence_complete(text):
        return 'complete'

    # Check for mid-word truncation (ends with partial word)
    if re.search(r'\w+$', text) and not re.search(r'[.!?]\s*$', text):
        # Check if it's likely a partial word (common prefixes/suffixes)
        last_word = text.split()[-1] if text.split() else ""
        if len(last_word) > 2 and not last_word.endswith(('.', '!', '?', ',')):
            return 'mid_word'

    # Check for mid-sentence truncation
    if not re.search(r'[.!?]["\']?\s*$', text):
        return 'mid_sentence'

    return 'abrupt_end'

def fetch_additional_passages(query, current_passages, top_k=3):
    """
    Fetch additional passages for extending truncated responses
    Excludes already used passages
    """
    try:
        if embedder is None or faiss_index is None:
            return []

        # Get more passages than needed
        query_embedding = embedder.encode([query])
        scores, indices = faiss_index.search(query_embedding.astype(np.float32), top_k + len(current_passages))

        # Filter out already used passages
        used_indices = set()
        for passage in current_passages:
            # Find the index of this passage
            try:
                idx = passages_list.index(passage['passage'])
                used_indices.add(idx)
            except ValueError:
                continue

        # Get new passages
        additional_passages = []
        min_score_threshold = 0.25  # Lower threshold for additional passages

        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if (idx != -1 and idx < len(passages_list) and 
                score >= min_score_threshold and 
                idx not in used_indices and 
                len(additional_passages) < top_k):

                additional_passages.append({
                    'rank': len(current_passages) + len(additional_passages) + 1,
                    'passage_id': passage_ids[idx] if idx < len(passage_ids) else f"passage_{idx}",
                    'passage': passages_list[idx],
                    'score': float(score)
                })

        logger.info(f"Fetched {len(additional_passages)} additional passages for truncation handling")
        return additional_passages

    except Exception as e:
        logger.error(f"Error fetching additional passages: {e}")
        return []

def _get_continuation_prompt(original_response, query, additional_context):
    if model_type == "TinyLlama":
        return f"""<|system|>
You are a medical assistant. Complete the previous response in 1-2 sentences only.
<|user|>
Original question: {query}
{f"Additional context: {additional_context}" if additional_context else ""}

Previous incomplete response: {original_response}

Complete this response briefly (1-2 sentences):
<|assistant|>
"""
    return f"""Medical question: {query}
{f"Additional context: {additional_context}" if additional_context else ""}

Incomplete response: {original_response}

Complete briefly: """

def continue_generation(original_response, query, additional_context="", max_attempts=2):
    try:
        prompt = _get_continuation_prompt(original_response, query, additional_context)
        for attempt in range(max_attempts):
            response = answer_generator(
                prompt,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=answer_generator.tokenizer.eos_token_id,
                eos_token_id=answer_generator.tokenizer.eos_token_id,
                return_full_text=False
            )
            continuation = clean_generated_response(response[0]['generated_text'].strip(), query)
            if continuation and len(continuation) > 5:
                combined = original_response.rstrip() + " " + continuation
                return trim_to_sentences(combined, max_sentences=3)
        return trim_to_sentences(original_response, max_sentences=3)
    except Exception as e:
        logger.error(f"Error in continue_generation: {e}")
        return original_response
    
def trim_to_sentences(text, max_sentences=3):
    """
    Trim text to maximum number of sentences
    """
    if not text:
        return ""
    
    # Split by sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # Keep only first max_sentences
    trimmed_sentences = sentences[:max_sentences]
    
    # Join back
    result = ' '.join(trimmed_sentences).strip()
    
    # Ensure it ends with punctuation
    if result and not result.endswith(('.', '!', '?')):
        result += '.'
    
    return result    

def _handle_truncation_iteration(current_response, query, iteration, retrieved_passages):
    truncation_type = detect_truncation_type(current_response)
    if truncation_type == 'complete':
        return current_response, True
    
    logger.info(f"Iteration {iteration}: Handling {truncation_type} truncation")
    if iteration == 0:
        return continue_generation(current_response, query), False
    
    # Simple completion fallback
    if not current_response.rstrip().endswith(('.', '!', '?')):
        current_response = current_response.rstrip() + "."
    if len(current_response.strip()) < 30 and retrieved_passages:
        context_summary = retrieved_passages[0]['passage'][:100]
        current_response = f"Based on medical information: {context_summary}."
    return current_response, False

def handle_truncation_post_processing(response, query, retrieved_passages, max_iterations=2):
    try:
        current_response = trim_to_sentences(response, max_sentences=3)
        for iteration in range(max_iterations):
            current_response, is_complete = _handle_truncation_iteration(
                current_response, query, iteration+1, retrieved_passages
            )
            current_response = trim_to_sentences(current_response, max_sentences=3)
            if is_complete:
                break
        final_type = detect_truncation_type(current_response)
        return current_response, final_type
    except Exception as e:
        logger.error(f"Error in truncation post-processing: {e}")
        return trim_to_sentences(response, max_sentences=3), 'error'
 
def clean_generated_response(text, original_query):
    """Enhanced response cleaning with truncation awareness"""
    if not text:
        return ""
 
    # Remove common training artifacts
    cleaning_patterns = [
        r'<\|user\|>.*?(?=\n|$)',  # Remove <|user|> tokens
        r'<\|assistant\|>.*?(?=\n|$)',  # Remove <|assistant|> tokens
        r'Question:\s*.*?(?=\n|$)',  # Remove "Question:" lines
        r'Answer:\s*',  # Remove "Answer:" prefix
        r'Based on the (?:provided )?(?:medical )?(?:information|context)[:,]?\s*',  # Clean common prefixes
        r'\n\s*\n',  # Multiple newlines
    ]
 
    cleaned = text.strip()
 
    # Apply cleaning patterns
    for pattern in cleaning_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
 
    # Remove duplicate sentences
    sentences = [s.strip() for s in cleaned.split('.') if s.strip()]
    unique_sentences = []
    seen = set()
 
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if sentence_lower not in seen and len(sentence) > 10:
            unique_sentences.append(sentence)
            seen.add(sentence_lower)
 
    # Reconstruct text
    if unique_sentences:
        cleaned = '. '.join(unique_sentences)
        if not cleaned.endswith('.'):
            cleaned += '.'
 
    # Ensure the response is complete and relevant
    if len(cleaned.strip()) < 20:
        return ""
 
    return cleaned.strip()

def rewrite_query_optimized(original_query):
    """Optimized query rewriting with Gemma support"""
    if query_rewriter is None:
        # Simple fallback expansion
        medical_expansions = {
            'diabetes': 'diabetes mellitus blood glucose insulin',
            'cancer': 'cancer tumor malignant neoplasm oncology',
            'heart': 'heart cardiac cardiovascular',
            'covid': 'covid coronavirus sars-cov-2 pandemic',
            'pain': 'pain ache discomfort symptom',
            'fever': 'fever temperature pyrexia',
            'blood': 'blood plasma serum hematology'
        }

        expanded = original_query.lower()
        for term, expansion in medical_expansions.items():
            if term in expanded:
                expanded = expanded.replace(term, expansion)
                break

        return expanded if expanded != original_query.lower() else original_query

    try:
        # Use the loaded query rewriter
        prompt = f"Expand this medical question with relevant terms: {original_query}\nExpanded:"

        result = query_rewriter(
            prompt,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=query_rewriter.tokenizer.eos_token_id if hasattr(query_rewriter, 'tokenizer') else None,
            return_full_text=False
        )

        generated_text = result[0]['generated_text'].strip()

        # Clean up the response
        if "Expanded:" in generated_text:
            rewritten = generated_text.split("Expanded:")[-1].strip()
        else:
            rewritten = generated_text.strip()

        # Validate rewritten query
        if len(rewritten) > 5 and len(rewritten) < len(original_query) * 3:
            logger.info(f"Query rewritten: '{original_query}' -> '{rewritten}'")
            return rewritten

    except Exception as e:
        logger.warning(f"Query rewriting error: {e}")

    return original_query

def retrieve_passages_optimized(query, top_k=5):
    """Optimized retrieval with better error handling"""
    if embedder is None or faiss_index is None:
        logger.warning("Embedder or FAISS index not available")
        return query, []

    try:
        # Rewrite query
        rewritten = rewrite_query_optimized(query)

        # Embed query
        query_embedding = embedder.encode([rewritten])

        # Search FAISS
        scores, indices = faiss_index.search(query_embedding.astype(np.float32), top_k)

        # Filter results
        min_score_threshold = 0.3
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1 and idx < len(passages_list) and score >= min_score_threshold:
                results.append({
                    'rank': i+1,
                    'passage_id': passage_ids[idx] if idx < len(passage_ids) else f"passage_{idx}",
                    'passage': passages_list[idx],
                    'score': float(score)
                })

        logger.info(f"Retrieved {len(results)} passages for query: {query[:50]}...")
        return rewritten, results

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return query, []
  
  

def _is_medical_question(question):
    medical_keywords = [
        'disease', 'disorder', 'syndrome', 'treatment', 'therapy', 'medicine',
        'drug', 'medication', 'symptom', 'diagnosis', 'patient', 'clinical',
        'medical', 'health', 'cancer', 'tumor', 'infection', 'virus', 'bacteria',
        'gene', 'genetic', 'protein', 'enzyme', 'cell', 'tissue', 'organ',
        'blood', 'heart', 'brain', 'liver', 'kidney', 'lung', 'diabetes',
        'hypertension', 'covid', 'vaccine', 'antibody', 'immune', 'pathology',
        'surgery', 'procedure', 'chronic', 'acute', 'inflammation', 'pain',
        'fever', 'cough', 'headache', 'nausea', 'fatigue'
    ]
    
    non_medical_keywords = [
        'economy', 'politics', 'sports', 'cooking', 'travel', 'tariff',
        'weather', 'music', 'movie', 'game', 'fashion', 'shopping',
        'restaurant', 'hotel', 'vacation', 'concert', 'stock', 'investment',
        'software', 'app', 'website', 'internet', 'social media', 'smartphone',
        'computer', 'programming', 'artificial intelligence', 'blockchain',
        'cryptocurrency', 'gaming', 'streaming', 'podcast',
        'book', 'literature', 'poetry', 'theater', 'dance', 'photography',
        'painting', 'sculpture', 'museum', 'gallery', 'festival', 'comedy',
        'television', 'documentary', 'animation', 'celebrity',
        'school', 'university', 'college', 'education', 'teacher', 'student',
        'job', 'career', 'interview', 'resume', 'workplace', 'business',
        'entrepreneur', 'marketing', 'advertising', 'management',
        'home', 'apartment', 'furniture', 'decoration', 'garden', 'pets',
        'cat', 'dog', 'cleaning', 'organizing', 'DIY', 'craft', 'hobby',
        'exercise', 'fitness', 'yoga', 'meditation', 'mindfulness',
        'car', 'bus', 'train', 'airplane', 'bicycle', 'motorcycle',
        'city', 'country', 'beach', 'mountain', 'park', 'neighborhood',
        'building', 'architecture', 'bridge', 'road',
        'recipe', 'ingredient', 'bakery', 'cafe', 'bar', 'wine', 'beer',
        'vegetarian', 'vegan', 'diet', 'nutrition', 'grocery', 'farming',
        'family', 'friend', 'relationship', 'dating', 'marriage', 'wedding',
        'party', 'celebration', 'birthday', 'holiday', 'tradition', 'culture',
        'nature', 'wildlife', 'forest', 'ocean', 'river', 'lake', 'camping',
        'hiking', 'environment', 'climate', 'sustainability', 'recycling',
        'football', 'basketball', 'baseball', 'soccer', 'tennis', 'golf',
        'swimming', 'running', 'cycling', 'skiing', 'surfing', 'team',
        'tournament', 'championship', 'athlete', 'stadium', 'cricket',
        'store', 'mall', 'online shopping', 'brand', 'product', 'discount',
        'sale', 'coupon', 'delivery', 'shipping', 'return', 'warranty'
    ]

    question_lower = question.lower()
    has_medical = any(keyword in question_lower for keyword in medical_keywords)
    has_non_medical = any(keyword in question_lower for keyword in non_medical_keywords)
    
    return has_medical, has_non_medical

def _build_context(retrieved_passages):
    context_parts = []
    for i, p in enumerate(retrieved_passages[:3]):
        context_parts.append(f"Reference {i+1}: {p['passage'][:300]}")
    return "\n".join(context_parts)

def _generate_prompt(question, context):
    if model_type == "TinyLlama":
        return f"""<|system|>
You are a medical assistant. Provide a clear, complete answer based on the medical information provided. Do not include training artifacts or incomplete sentences.
<|user|>
Medical Information:
{context}

Question: {question}

Please provide a comprehensive answer about {question.lower()}.
<|assistant|>
"""
    return f"""Medical context: {context}

Question: {question}
Provide a complete medical answer: """

def _generate_answer(prompt):
    return answer_generator(
        prompt,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=answer_generator.tokenizer.eos_token_id if hasattr(answer_generator, 'tokenizer') else None,
        eos_token_id=answer_generator.tokenizer.eos_token_id if hasattr(answer_generator, 'tokenizer') else None,
        return_full_text=False
    )[0]['generated_text'].strip()

def _calculate_confidence(retrieved_passages, truncation_status):
    if not retrieved_passages:
        return 0.3
        
    avg_score = sum(p['score'] for p in retrieved_passages) / len(retrieved_passages)
    base_confidence = min(0.95, max(0.4, avg_score))

    if truncation_status == 'complete':
        return base_confidence
    elif truncation_status in ['mid_sentence', 'mid_word']:
        return base_confidence * 0.9
    return base_confidence * 0.8

def medical_chatbot(question):
    try:
        # Medical content check
        has_medical, has_non_medical = _is_medical_question(question)
        if has_non_medical and not has_medical:
            return {
                "result": "I can only answer medical and health-related questions. Please ask about diseases, treatments, symptoms, or medical conditions.",
                "context_check": "failed",
                "confidence": 0.0
            }

        # Retrieve passages
        rewritten_query, retrieved_passages = retrieve_passages_optimized(question, top_k=5)
        if not retrieved_passages:
            return {
                "result": "I couldn't find relevant medical information to answer your question. Please try rephrasing with more specific medical terms.",
                "context_check": "no_docs",
                "confidence": 0.3,
                "rewritten_query": rewritten_query
            }

        # Prepare context
        context = _build_context(retrieved_passages)
        prompt = _generate_prompt(question, context)
        truncation_status = 'complete'  # Default status

        try:
            # Generate initial answer
            raw_answer = _generate_answer(prompt)
            answer = clean_generated_response(raw_answer, question)
            
            # Handle truncation
            if answer:
                answer, truncation_status = handle_truncation_post_processing(answer, question, retrieved_passages)
            
            # Fallback if answer is too short
            if not answer or len(answer.strip()) < 30:
                context_summary = context[:200].replace('\n', ' ')
                answer = f"Based on the available medical information, {question.lower()} involves: {context_summary}..."
                truncation_status = 'fallback_used'
        
        except Exception as gen_error:
            logger.error(f"Text generation error: {gen_error}")
            context_summary = context[:300].replace('\n', ' ')
            answer = f"Based on the medical information available: {context_summary}"
            truncation_status = 'generation_error'

        # Final formatting
        if len(answer.strip()) < 30:
            answer = f"Regarding {question.lower()}: " + answer

        # Calculate confidence
        confidence = _calculate_confidence(retrieved_passages, truncation_status)

        return {
            "result": answer,
            "rewritten_query": rewritten_query,
            "retrieved_passages": retrieved_passages[:5],
            "context_check": "passed",
            "confidence": float(confidence),
            "model_used": model_type,
            "query_rewriter": "Gemma-2-2b-it" if query_rewriter else "Simple expansion",
            "memory_optimized": True,
            "truncation_status": truncation_status,
            "truncation_handled": truncation_status != 'complete'
        }

    except Exception as e:
        logger.error(f"Medical chatbot error: {str(e)}")
        return {
            "result": "I apologize, but I encountered an error processing your medical question. Please try again or rephrase your question.",
            "context_check": "error",
            "confidence": 0.0,
            "error": str(e),
            "truncation_status": "error"
        }

# Initialize system when module is imported (but quietly)
def lazy_init():
    """Initialize system only when first needed"""
    global embedder
    if embedder is None:
        logger.info("Performing lazy initialization...")
        initialize_system()

# Export the main function
__all__ = ['medical_chatbot', 'initialize_system', 'lazy_init']

# Only run initialization if this file is run directly (not imported)
if __name__ == "__main__":
    # Interactive mode
    print("🚀 ENHANCED RAG SETUP WITH TRUNCATION HANDLING")
    print("="*60)

    if initialize_system():
        print("✅ System initialized successfully!")

        # Test the system
        try:
            test_result = medical_chatbot("What is diabetes?")
            print("SYSTEM TEST:")
            print(f"   Status: {'✅ PASSED' if test_result['context_check'] == 'passed' else '❌ FAILED'}")
            print(f"   Answer: {test_result['result']}")
            print(f"   Confidence: {test_result.get('confidence', 'N/A')}")
            print(f"   Truncation Status: {test_result.get('truncation_status', 'N/A')}")
            print(f"   Truncation Handled: {test_result.get('truncation_handled', 'N/A')}")
            print(f"   Query Rewriter: {test_result.get('query_rewriter', 'N/A')}")
            if test_result['context_check'] == 'passed':
                print("\n✅ READY FOR USE WITH ENHANCED TRUNCATION HANDLING!")
            else:
                print(f"\n⚠️ Test failed: {test_result.get('error', 'Unknown error')}")
        except Exception as test_error:
            print(f"\n⚠️ Test failed with error: {test_error}")
    else:
        print("❌ System initialization failed!")
else:
    # Production mode - lazy initialization
    logger.info("Enhanced RAG chatbot module with truncation handling imported")