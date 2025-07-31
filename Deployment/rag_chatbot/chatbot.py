"""
Production-ready RAG Chatbot with Enhanced Truncation Handling
SonarQube-optimized complete version
"""

import torch
import numpy as np
import gc
import os
import logging
import re
from typing import Dict, List, Tuple, Optional, Union

from load_system import load_rag_system

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PYTORCH_CUDA_CONFIG = 'expandable_segments:True'
EMBEDDER_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
GEMMA_MODEL = "google/gemma-2-2b-it"
TINYLLAMA_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DISTILGPT2_MODEL = "distilgpt2"
MIN_SCORE_THRESHOLD = 0.3
MAX_TRUNCATION_ITERATIONS = 2
MAX_SENTENCES = 3
MIN_CONTINUATION_LENGTH = 5
MIN_ANSWER_LENGTH = 30

# Set memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = PYTORCH_CUDA_CONFIG

# Global variables to store loaded components
embedder = None
query_rewriter = None
answer_generator = None
model_type = None
faiss_index = None
passages_list = None
passage_ids = None
clean_passages_df = None

# Precompiled regex patterns
SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+')
END_PUNCTUATION_PATTERN = re.compile(r'[.!?]\s*$')
MID_WORD_PATTERN = re.compile(r'\w+$')
INCOMPLETE_PATTERNS = [
    re.compile(r'\b(the|a|an|and|or|but|in|on|at|to|for|with|by)\s*$', re.IGNORECASE),
    re.compile(r'\b(is|are|was|were|has|have|had|will|would|could|should)\s*$', re.IGNORECASE),
    re.compile(r'\b(such|like|including|especially|particularly)\s*$', re.IGNORECASE),
    re.compile(r'\b(because|since|although|while|when|where|which|that)\s*$', re.IGNORECASE),
    re.compile(r'\b(more|less|most|least|very|quite|rather|extremely)\s*$', re.IGNORECASE),
    re.compile(r'\b(can|may|might|must|should|would|could)\s*$', re.IGNORECASE)
]

CLEANING_PATTERNS = [
    re.compile(r'<\|user\|>.*', re.IGNORECASE | re.MULTILINE),
    re.compile(r'<\|assistant\|>.*', re.IGNORECASE | re.MULTILINE),
    re.compile(r'Question:\s*.*', re.IGNORECASE | re.MULTILINE),
    re.compile(r'Answer:\s*', re.IGNORECASE),
    re.compile(r'Based on the (?:provided )?(?:medical )?(?:information|context)[:,]?\s*', re.IGNORECASE),
    re.compile(r'\n\s*\n', re.MULTILINE)
]

MEDICAL_KEYWORDS = [
    'disease', 'disorder', 'syndrome', 'treatment', 'therapy', 'medicine',
    'drug', 'medication', 'symptom', 'diagnosis', 'patient', 'clinical',
    'medical', 'health', 'cancer', 'tumor', 'infection', 'virus', 'bacteria',
    'gene', 'genetic', 'protein', 'enzyme', 'cell', 'tissue', 'organ',
    'blood', 'heart', 'brain', 'liver', 'kidney', 'lung', 'diabetes',
    'hypertension', 'covid', 'vaccine', 'antibody', 'immune', 'pathology',
    'surgery', 'procedure', 'chronic', 'acute', 'inflammation', 'pain',
    'fever', 'cough', 'headache', 'nausea', 'fatigue'
]

NON_MEDICAL_KEYWORDS = [
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

MEDICAL_EXPANSIONS = {
    'diabetes': 'diabetes mellitus blood glucose insulin',
    'cancer': 'cancer tumor malignant neoplasm oncology',
    'heart': 'heart cardiac cardiovascular',
    'covid': 'covid coronavirus sars-cov-2 pandemic',
    'pain': 'pain ache discomfort symptom',
    'fever': 'fever temperature pyrexia',
    'blood': 'blood plasma serum hematology'
}

def initialize_system() -> bool:
    """Initialize the complete RAG system"""
    global embedder, query_rewriter, answer_generator, model_type
    global faiss_index, passages_list, passage_ids, clean_passages_df

    try:
        logger.info("Initializing memory-optimized RAG system...")
        cleanup_memory()
        
        if not load_data():
            logger.error("Failed to load data")
            return False
            
        if not load_models():
            logger.error("Failed to load models")
            return False

        logger.info("RAG system initialized successfully")
        return True
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        return False

def cleanup_memory() -> None:
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

def load_data() -> bool:
    """Load RAG system data"""
    global faiss_index, passages_list, passage_ids, clean_passages_df

    try:
        logger.info("Loading RAG data...")
        checkpoint_path = 'load_system.py'
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return False

        result = load_rag_system()
        faiss_index, passages_list, passage_ids, _, _ = result
        logger.info(f"Loaded {len(passages_list)} passages")
        return True
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        return False

def _hf_login(hf_token: str) -> bool:
    """Login to HuggingFace Hub"""
    import huggingface_hub
    try:
        huggingface_hub.login(token=hf_token)
        logger.info("✅ HuggingFace login successful")
        return True
    except Exception as e:
        logger.warning(f"HF login failed: {e}")
        return False

def _load_embedder() -> None:
    """Load sentence embedder model"""
    from sentence_transformers import SentenceTransformer
    logger.info("Loading sentence embedder...")
    global embedder
    embedder = SentenceTransformer(EMBEDDER_MODEL)
    logger.info("✅ Sentence embedder loaded successfully")

def _load_gemma_rewriter(hf_token: str):
    """Load Gemma query rewriter with quantization"""
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
    logger.info("Attempting to load Gemma-2-2b-it...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL, token=hf_token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_MODEL,
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
    """Load fallback query rewriter (DistilGPT2)"""
    from transformers import pipeline
    logger.info("Loading DistilGPT2 as fallback query rewriter...")
    return pipeline(
        "text-generation", 
        model=DISTILGPT2_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.7,
        return_full_text=False
    )

def _load_answer_generator():
    """Load answer generator with fallback"""
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    global model_type
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(TINYLLAMA_MODEL)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            TINYLLAMA_MODEL,
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
    except Exception as e:
        logger.warning(f"TinyLlama failed: {e}, loading DistilGPT2 answer generator")
        model_type = "DistilGPT2"
        return pipeline(
            "text-generation", 
            model=DISTILGPT2_MODEL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False
        )

def load_models() -> bool:
    """Load all required models"""
    global embedder, query_rewriter, answer_generator, model_type
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
        import transformers
        import huggingface_hub
        
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
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load query rewriter
        query_rewriter = None
        version_parts = transformers.__version__.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1]) if len(version_parts) > 1 else 0
        
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        answer_generator = _load_answer_generator()
        logger.info(f"✅ Answer generator loaded ({model_type})")
        return True

    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        return False

def is_sentence_complete(text: str) -> bool:
    """Check if text ends with a complete sentence"""
    if not text or not text.strip():
        return False

    text = text.strip()

    if not END_PUNCTUATION_PATTERN.search(text):
        return False

    for pattern in INCOMPLETE_PATTERNS:
        if pattern.search(text):
            return False

    if text.count('(') != text.count(')') or text.count('"') % 2 != 0 or text.count("'") % 2 != 0:
        return False

    sentences = END_PUNCTUATION_PATTERN.split(text)
    if sentences:
        last_sentence = sentences[-2] if len(sentences) > 1 else sentences[0]
        if len(last_sentence.strip().split()) < 3:
            return False

    return True

def detect_truncation_type(text: str) -> str:
    """Detect the type of truncation in the response"""
    if not text or not text.strip():
        return 'abrupt_end'

    text = text.strip()

    if is_sentence_complete(text):
        return 'complete'

    if MID_WORD_PATTERN.search(text) and not END_PUNCTUATION_PATTERN.search(text):
        last_word = text.split()[-1] if text.split() else ""
        if len(last_word) > 2 and not last_word.endswith(('.', '!', '?', ',')):
            return 'mid_word'

    if not END_PUNCTUATION_PATTERN.search(text):
        return 'mid_sentence'

    return 'abrupt_end'

def _get_continuation_prompt(original_response: str, query: str, additional_context: str) -> str:
    """Generate continuation prompt based on model type"""
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

def continue_generation(
    original_response: str, 
    query: str, 
    additional_context: str = "", 
    max_attempts: int = 2
) -> str:
    """Continue generation from where it was truncated"""
    try:
        prompt = _get_continuation_prompt(original_response, query, additional_context)
        for attempt in range(max_attempts):
            try:
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
                if continuation and len(continuation) > MIN_CONTINUATION_LENGTH:
                    combined = original_response.rstrip() + " " + continuation
                    return trim_to_sentences(combined, MAX_SENTENCES)
            except Exception as e:
                logger.warning(f"Continuation attempt {attempt + 1} failed: {e}")
                
        return trim_to_sentences(original_response, MAX_SENTENCES)
    except Exception as e:
        logger.error(f"Error in continue_generation: {e}")
        return original_response
    
def trim_to_sentences(text: str, max_sentences: int = MAX_SENTENCES) -> str:
    """Trim text to maximum number of sentences"""
    if not text:
        return ""
    
    sentences = SENTENCE_SPLIT_PATTERN.split(text.strip())
    trimmed_sentences = sentences[:max_sentences]
    result = ' '.join(trimmed_sentences).strip()
    
    if result and not result.endswith(('.', '!', '?')):
        result += '.'
    
    return result    

def _handle_truncation_iteration(
    current_response: str, 
    query: str, 
    iteration: int, 
    retrieved_passages: List[Dict]
) -> Tuple[str, bool]:
    """Handle one iteration of truncation processing"""
    truncation_type = detect_truncation_type(current_response)
    if truncation_type == 'complete':
        return current_response, True
    
    logger.info(f"Iteration {iteration}: Handling {truncation_type} truncation")
    if iteration == 0:
        return continue_generation(current_response, query), False
    
    if not current_response.rstrip().endswith(('.', '!', '?')):
        current_response = current_response.rstrip() + "."
    if len(current_response.strip()) < MIN_ANSWER_LENGTH and retrieved_passages:
        context_summary = retrieved_passages[0]['passage'][:100]
        current_response = f"Based on medical information: {context_summary}."
    return current_response, False

def handle_truncation_post_processing(
    response: str, 
    query: str, 
    retrieved_passages: List[Dict], 
    max_iterations: int = MAX_TRUNCATION_ITERATIONS
) -> Tuple[str, str]:
    """Main truncation handling function"""
    try:
        current_response = trim_to_sentences(response, MAX_SENTENCES)
        for iteration in range(max_iterations):
            current_response, is_complete = _handle_truncation_iteration(
                current_response, query, iteration+1, retrieved_passages
            )
            current_response = trim_to_sentences(current_response, MAX_SENTENCES)
            if is_complete:
                break
        final_type = detect_truncation_type(current_response)
        return current_response, final_type
    except Exception as e:
        logger.error(f"Error in truncation post-processing: {e}")
        return trim_to_sentences(response, MAX_SENTENCES), 'error'
 
def clean_generated_response(text: str, original_query: str) -> str:
    """Enhanced response cleaning with truncation awareness"""
    if not text:
        return ""
 
    cleaned = text.strip()
 
    for pattern in CLEANING_PATTERNS:
        cleaned = pattern.sub('', cleaned)
 
    sentences = [s.strip() for s in cleaned.split('.') if s.strip()]
    unique_sentences = []
    seen = set()
 
    for sentence in sentences:
        if len(sentence) <= 10:
            continue
        sentence_lower = sentence.lower()
        if sentence_lower not in seen:
            unique_sentences.append(sentence)
            seen.add(sentence_lower)
 
    if unique_sentences:
        cleaned = '. '.join(unique_sentences)
        if not cleaned.endswith('.'):
            cleaned += '.'
    elif len(cleaned.strip()) < 20:
        return ""
 
    return cleaned.strip()

def rewrite_query_optimized(original_query: str) -> str:
    """Optimized query rewriting with Gemma support"""
    if query_rewriter is None:
        expanded = original_query.lower()
        for term, expansion in MEDICAL_EXPANSIONS.items():
            if term in expanded:
                return expanded.replace(term, expansion)
        return original_query

    try:
        prompt = f"Expand this medical question with relevant terms: {original_query}\nExpanded:"
        result = query_rewriter(
            prompt,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=query_rewriter.tokenizer.eos_token_id,
            return_full_text=False
        )

        generated_text = result[0]['generated_text'].strip()
        rewritten = generated_text.split("Expanded:")[-1].strip() if "Expanded:" in generated_text else generated_text.strip()

        if 5 < len(rewritten) < len(original_query) * 3:
            logger.info(f"Query rewritten: '{original_query}' -> '{rewritten}'")
            return rewritten
        return original_query
    except Exception as e:
        logger.warning(f"Query rewriting error: {e}")
        return original_query

def retrieve_passages_optimized(
    query: str, 
    top_k: int = 5
) -> Tuple[str, List[Dict]]:
    """Optimized retrieval with better error handling"""
    if embedder is None or faiss_index is None:
        logger.warning("Embedder or FAISS index not available")
        return query, []

    try:
        rewritten = rewrite_query_optimized(query)
        query_embedding = embedder.encode([rewritten])
        scores, indices = faiss_index.search(query_embedding.astype(np.float32), top_k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1 or idx >= len(passages_list) or score < MIN_SCORE_THRESHOLD:
                continue
                
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
  
def _is_medical_question(question: str) -> Tuple[bool, bool]:
    """Determine if question is medical-related"""
    question_lower = question.lower()
    has_medical = any(keyword in question_lower for keyword in MEDICAL_KEYWORDS)
    has_non_medical = any(keyword in question_lower for keyword in NON_MEDICAL_KEYWORDS)
    return has_medical, has_non_medical

def _build_context(retrieved_passages: List[Dict]) -> str:
    """Build context from retrieved passages"""
    return "\n".join(
        f"Reference {i+1}: {p['passage'][:300]}" 
        for i, p in enumerate(retrieved_passages[:3])
    )

def _generate_prompt(question: str, context: str) -> str:
    """Generate appropriate prompt based on model type"""
    if model_type == "TinyLlama":
        return f"""<|system|>
You are a medical assistant. Provide a clear, complete answer based on the medical information provided.
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

def _generate_answer(prompt: str) -> str:
    """Generate answer from the language model"""
    return answer_generator(
        prompt,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=answer_generator.tokenizer.eos_token_id,
        eos_token_id=answer_generator.tokenizer.eos_token_id,
        return_full_text=False
    )[0]['generated_text'].strip()

def _calculate_confidence(
    retrieved_passages: List[Dict], 
    truncation_status: str
) -> float:
    """Calculate confidence score for response"""
    if not retrieved_passages:
        return 0.3
        
    avg_score = sum(p['score'] for p in retrieved_passages) / len(retrieved_passages)
    base_confidence = min(0.95, max(0.4, avg_score))

    if truncation_status == 'complete':
        return base_confidence
    if truncation_status in ['mid_sentence', 'mid_word']:
        return base_confidence * 0.9
    return base_confidence * 0.8

def medical_chatbot(question: str) -> Dict:
    """Main medical chatbot function with enhanced truncation handling"""
    try:
        has_medical, has_non_medical = _is_medical_question(question)
        if has_non_medical and not has_medical:
            return {
                "result": "I can only answer medical and health-related questions. Please ask about diseases, treatments, symptoms, or medical conditions.",
                "context_check": "failed",
                "confidence": 0.0
            }

        rewritten_query, retrieved_passages = retrieve_passages_optimized(question, top_k=5)
        if not retrieved_passages:
            return {
                "result": "I couldn't find relevant medical information to answer your question. Please try rephrasing with more specific medical terms.",
                "context_check": "no_docs",
                "confidence": 0.3,
                "rewritten_query": rewritten_query
            }

        context = _build_context(retrieved_passages)
        prompt = _generate_prompt(question, context)
        truncation_status = 'complete'
        answer = ""

        try:
            raw_answer = _generate_answer(prompt)
            answer = clean_generated_response(raw_answer, question)
            
            if answer:
                answer, truncation_status = handle_truncation_post_processing(answer, question, retrieved_passages)
            
            if not answer or len(answer.strip()) < MIN_ANSWER_LENGTH:
                context_summary = context[:200].replace('\n', ' ')
                answer = f"Based on the available medical information, {question.lower()} involves: {context_summary}..."
                truncation_status = 'fallback_used'
        
        except Exception as gen_error:
            logger.error(f"Text generation error: {gen_error}")
            context_summary = context[:300].replace('\n', ' ')
            answer = f"Based on the medical information available: {context_summary}"
            truncation_status = 'generation_error'

        if answer and len(answer.strip()) < MIN_ANSWER_LENGTH:
            answer = f"Regarding {question.lower()}: {answer}"

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

def lazy_init() -> None:
    """Initialize system only when first needed"""
    global embedder
    if embedder is None:
        logger.info("Performing lazy initialization...")
        initialize_system()

# Export the main function
__all__ = ['medical_chatbot', 'initialize_system', 'lazy_init']

# Only run initialization if this file is run directly
if __name__ == "__main__":
    print("🚀 ENHANCED RAG SETUP WITH TRUNCATION HANDLING")
    print("="*60)

    if initialize_system():
        print("✅ System initialized successfully!")
        test_result = medical_chatbot("What is diabetes?")
        status = '✅ PASSED' if test_result['context_check'] == 'passed' else '❌ FAILED'
        print(f"SYSTEM TEST:\n   Status: {status}\n   Answer: {test_result['result']}")
    else:
        print("❌ System initialization failed!")