
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

def load_models():
    """Load all required models with Gemma support"""
    global embedder, query_rewriter, answer_generator, model_type

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
        from sentence_transformers import SentenceTransformer
        import huggingface_hub

        # Check transformers version
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")

        # Check for HF token
        hf_token = os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
        if hf_token:
            logger.info("Hugging Face token found, logging in...")
            try:
                huggingface_hub.login(token=hf_token)
                logger.info("✅ HuggingFace login successful")
            except Exception as e:
                logger.warning(f"HF login failed: {e}")
        else:
            logger.warning("No Hugging Face token found - will skip gated models")

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 1. Load embedder
        logger.info("Loading sentence embedder...")
        embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        logger.info("✅ Sentence embedder loaded successfully")

        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2. Load query rewriter - Try Gemma first
        logger.info("Loading query rewriter...")
        query_rewriter_loaded = False

        # Try Gemma with proper version check and token
        if hf_token and hasattr(transformers, '__version__'):
            version_parts = transformers.__version__.split('.')
            major, minor = int(version_parts[0]), int(version_parts[1])

            if major > 4 or (major == 4 and minor >= 38):
                try:
                    logger.info("Attempting to load Gemma-2-2b-it...")

                    # Configure quantization
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )

                    # Load tokenizer
                    query_tokenizer = AutoTokenizer.from_pretrained(
                        "google/gemma-2-2b-it",
                        token=hf_token,
                        trust_remote_code=True
                    )

                    # Load model
                    query_model = AutoModelForCausalLM.from_pretrained(
                        "google/gemma-2-2b-it",
                        quantization_config=quantization_config,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        token=hf_token,
                        trust_remote_code=True
                    )

                    # Create pipeline
                    query_rewriter = pipeline(
                        "text-generation",
                        model=query_model,
                        tokenizer=query_tokenizer,
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        return_full_text=False
                    )

                    query_rewriter_loaded = True
                    logger.info("✅ Gemma-2-2b-it query rewriter loaded successfully!")

                except Exception as e:
                    logger.warning(f"Gemma loading failed: {str(e)}")
                    logger.info("Will try fallback models...")
            else:
                logger.warning(f"Transformers {transformers.__version__} too old for Gemma (need 4.38.0+)")

        # Fallback query rewriters if Gemma failed
        if not query_rewriter_loaded:
            try:
                logger.info("Loading DistilGPT2 as fallback query rewriter...")
                query_rewriter = pipeline(
                    "text-generation", 
                    model="distilgpt2",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=0.7,
                    return_full_text=False
                )
                logger.info("✅ DistilGPT2 query rewriter loaded successfully")
                query_rewriter_loaded = True
            except Exception as e:
                logger.warning(f"DistilGPT2 failed: {e}")

        if not query_rewriter_loaded:
            logger.warning("No query rewriter loaded - will use simple fallback")
            query_rewriter = None

        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3. Load answer generator
        logger.info("Loading answer generator...")
        try:
            # Try TinyLlama
            answer_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

            # Fix tokenizer padding token
            if answer_tokenizer.pad_token is None:
                answer_tokenizer.pad_token = answer_tokenizer.eos_token

            answer_model = AutoModelForCausalLM.from_pretrained(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )

            # Create pipeline with better stopping criteria
            answer_generator = pipeline(
                "text-generation",
                model=answer_model,
                tokenizer=answer_tokenizer,
                max_new_tokens=150,  # Increased for complete responses
                do_sample=True,
                temperature=0.7,
                top_p=0.9,  # Added top_p for better quality
                pad_token_id=answer_tokenizer.eos_token_id,
                eos_token_id=answer_tokenizer.eos_token_id,
                return_full_text=False
            )
            model_type = "TinyLlama"
            logger.info("✅ TinyLlama answer generator loaded successfully")

        except Exception as e:
            logger.warning(f"TinyLlama loading failed: {str(e)}")
            try:
                # Fallback to DistilGPT2
                answer_generator = pipeline(
                    "text-generation", 
                    model="distilgpt2",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    return_full_text=False
                )
                model_type = "DistilGPT2"
                logger.info("✅ DistilGPT2 answer generator loaded successfully")
            except Exception as e2:
                logger.error(f"All answer generators failed: {str(e2)}")
                return False

        logger.info(f"✅ All models loaded successfully using {model_type}")
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

def continue_generation(original_response, query, additional_context="", max_attempts=2):
    """
    Continue generation from where it was truncated - optimized for concise responses
    """
    try:
        # Prepare continuation prompt for SHORT responses
        if model_type == "TinyLlama":
            continuation_prompt = f"""<|system|>
You are a medical assistant. Complete the previous response in 1-2 sentences only.
<|user|>
Original question: {query}
{f"Additional context: {additional_context}" if additional_context else ""}

Previous incomplete response: {original_response}

Complete this response briefly (1-2 sentences):
<|assistant|>
"""
        else:
            continuation_prompt = f"""Medical question: {query}
{f"Additional context: {additional_context}" if additional_context else ""}

Incomplete response: {original_response}

Complete briefly: """

        # Generate continuation with SHORTER limits
        for attempt in range(max_attempts):
            try:
                response = answer_generator(
                    continuation_prompt,
                    max_new_tokens=30,  # Much shorter for brief completion
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=answer_generator.tokenizer.eos_token_id if hasattr(answer_generator, 'tokenizer') else None,
                    eos_token_id=answer_generator.tokenizer.eos_token_id if hasattr(answer_generator, 'tokenizer') else None,
                    return_full_text=False
                )

                continuation = response[0]['generated_text'].strip()
                continuation = clean_generated_response(continuation, query)

                if continuation and len(continuation) > 5:
                    # Combine and trim to 2-3 sentences
                    combined_response = original_response.rstrip() + " " + continuation
                    combined_response = trim_to_sentences(combined_response, max_sentences=3)
                    logger.info(f"Successfully continued truncated response (attempt {attempt + 1})")
                    return combined_response

            except Exception as e:
                logger.warning(f"Continuation attempt {attempt + 1} failed: {e}")
                continue

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

def handle_truncation_post_processing(response, query, retrieved_passages, max_iterations=2):  # Reduced iterations
    """
    Main truncation handling function optimized for concise responses
    """
    try:
        original_response = response
        current_response = trim_to_sentences(response, max_sentences=3)  # Start with trimming
        iteration = 0
 
        while iteration < max_iterations:
            truncation_type = detect_truncation_type(current_response)
 
            if truncation_type == 'complete':
                logger.info(f"Response is complete after {iteration} iterations")
                break
 
            logger.info(f"Iteration {iteration + 1}: Detected {truncation_type} truncation")
 
            if truncation_type in ['mid_sentence', 'mid_word', 'abrupt_end']:
                # Strategy 1: Try to continue generation (concisely)
                if iteration == 0:
                    current_response = continue_generation(current_response, query)
 
                # Strategy 2: Fallback to simple completion
                else:
                    # Simple completion for concise response
                    if not current_response.rstrip().endswith(('.', '!', '?')):
                        current_response = current_response.rstrip() + "."
                   
                    # If too short, add brief context
                    if len(current_response.strip()) < 30:
                        context_summary = retrieved_passages[0]['passage'][:100] if retrieved_passages else ""
                        current_response = f"Based on medical information: {context_summary}."
                   
                    logger.info("Applied simple completion for concise response")
 
            # Always trim after each iteration
            current_response = trim_to_sentences(current_response, max_sentences=3)
            iteration += 1
 
        # Final validation and trimming
        final_response = trim_to_sentences(current_response, max_sentences=3)
        final_truncation_type = detect_truncation_type(final_response)
 
        logger.info(f"Final response length: {len(final_response)} characters")
        return final_response, final_truncation_type
 
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
  
  

def medical_chatbot(question):
    """Main medical chatbot function with enhanced truncation handling"""
    try:
        # Medical filtering
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

            # Technology & Digital
            'software', 'app', 'website', 'internet', 'social media', 'smartphone',
            'computer', 'programming', 'artificial intelligence', 'blockchain',
            'cryptocurrency', 'gaming', 'streaming', 'podcast',

            # Arts & Entertainment
            'book', 'literature', 'poetry', 'theater', 'dance', 'photography',
            'painting', 'sculpture', 'museum', 'gallery', 'festival', 'comedy',
            'television', 'documentary', 'animation', 'celebrity',

            # Education & Career
            'school', 'university', 'college', 'education', 'teacher', 'student',
            'job', 'career', 'interview', 'resume', 'workplace', 'business',
            'entrepreneur', 'marketing', 'advertising', 'management',

            # Home & Lifestyle
            'home', 'apartment', 'furniture', 'decoration', 'garden', 'pets',
            'cat', 'dog', 'cleaning', 'organizing', 'DIY', 'craft', 'hobby',
            'exercise', 'fitness', 'yoga', 'meditation', 'mindfulness',

            # Transportation & Places
            'car', 'bus', 'train', 'airplane', 'bicycle', 'motorcycle',
            'city', 'country', 'beach', 'mountain', 'park', 'neighborhood',
            'building', 'architecture', 'bridge', 'road',

            # Food & Dining
            'recipe', 'ingredient', 'bakery', 'cafe', 'bar', 'wine', 'beer',
            'vegetarian', 'vegan', 'diet', 'nutrition', 'grocery', 'farming',

            # Social & Relationships
            'family', 'friend', 'relationship', 'dating', 'marriage', 'wedding',
            'party', 'celebration', 'birthday', 'holiday', 'tradition', 'culture',

            # Nature & Environment
            'nature', 'wildlife', 'forest', 'ocean', 'river', 'lake', 'camping',
            'hiking', 'environment', 'climate', 'sustainability', 'recycling',

            # Sports & Recreation
            'football', 'basketball', 'baseball', 'soccer', 'tennis', 'golf',
            'swimming', 'running', 'cycling', 'skiing', 'surfing', 'team',
            'tournament', 'championship', 'athlete', 'stadium', 'cricket',

            # Shopping & Consumer
            'store', 'mall', 'online shopping', 'brand', 'product', 'discount',
            'sale', 'coupon', 'delivery', 'shipping', 'return', 'warranty'
        ]

        question_lower = question.lower()
        has_medical = any(keyword in question_lower for keyword in medical_keywords)
        has_non_medical = any(keyword in question_lower for keyword in non_medical_keywords)

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

        # Prepare context with better formatting
        context_parts = []
        for i, p in enumerate(retrieved_passages[:3]):
            context_parts.append(f"Reference {i+1}: {p['passage'][:300]}")
        context = "\n".join(context_parts)

        # Generate answer with improved prompts
        if model_type == "TinyLlama":
            prompt = f"""<|system|>
You are a medical assistant. Provide a clear, complete answer based on the medical information provided. Do not include training artifacts or incomplete sentences.
<|user|>
Medical Information:
{context}

Question: {question}

Please provide a comprehensive answer about {question.lower()}.
<|assistant|>
"""
        else:  # DistilGPT2 or others
            prompt = f"""Medical context: {context}

Question: {question}
Provide a complete medical answer: """

        # Generate response with better parameters
        try:
            response = answer_generator(
                prompt,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.3,  # Lower temperature for more focused responses
                top_p=0.9,
                repetition_penalty=1.1,  # Reduce repetition
                pad_token_id=answer_generator.tokenizer.eos_token_id if hasattr(answer_generator, 'tokenizer') else None,
                eos_token_id=answer_generator.tokenizer.eos_token_id if hasattr(answer_generator, 'tokenizer') else None,
                return_full_text=False
            )

            raw_answer = response[0]['generated_text'].strip()

            # Clean the response
            answer = clean_generated_response(raw_answer, question)

            # NEW: Apply truncation handling post-processing
            if answer:
                final_answer, truncation_status = handle_truncation_post_processing(
                    answer, question, retrieved_passages
                )
                answer = final_answer

            # If cleaning resulted in empty response, try fallback
            if not answer:
                # Extract key information from context for fallback
                context_summary = context[:200].replace('\n', ' ')
                answer = f"Based on the available medical information, {question.lower()} involves: {context_summary}..."
                truncation_status = 'fallback_used'

        except Exception as gen_error:
            logger.error(f"Text generation error: {gen_error}")
            # Fallback to context-based response
            context_summary = context[:300].replace('\n', ' ')
            answer = f"Based on the medical information available: {context_summary}"
            truncation_status = 'generation_error'

        # Final validation and enhancement
        if len(answer.strip()) < 30:
            answer = f"Regarding {question.lower()}: " + answer

        # Calculate confidence based on retrieval quality and truncation handling
        if retrieved_passages:
            avg_score = sum(p['score'] for p in retrieved_passages) / len(retrieved_passages)
            base_confidence = min(0.95, max(0.4, avg_score))

            # Adjust confidence based on truncation handling
            if truncation_status == 'complete':
                confidence = base_confidence
            elif truncation_status in ['mid_sentence', 'mid_word']:
                confidence = base_confidence * 0.9  # Slight reduction for handled truncation
            else:
                confidence = base_confidence * 0.8  # More reduction for problematic cases
        else:
            confidence = 0.3

        return {
            "result": answer,
            "rewritten_query": rewritten_query,
            "retrieved_passages": retrieved_passages[:5],
            "context_check": "passed",
            "confidence": float(confidence),
            "model_used": model_type,
            "query_rewriter": "Gemma-2-2b-it" if query_rewriter else "Simple expansion",
            "memory_optimized": True,
            "truncation_status": truncation_status,  # NEW: Track truncation handling
            "truncation_handled": truncation_status != 'complete'  # NEW: Flag for UI
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