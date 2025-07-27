from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import sys
from datetime import datetime
import traceback
import gc
import torch

# Add current directory to path to import your chatbot
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable for your RAG system
rag_system = None
system_loaded = False
_initialization_attempted = False

def initialize_rag_system():
    """Initialize your existing RAG system"""
    global rag_system, system_loaded, _initialization_attempted
    
    if _initialization_attempted:
        return system_loaded
        
    try:
        logger.info("Initializing RAG system from chatbot.py...")
        _initialization_attempted = True
        
        # Import your chatbot.py - this will execute all the setup code
        import chatbot
        
        # Check if chatbot has an initialization function
        if hasattr(chatbot, 'initialize_system'):
            logger.info("Found initialize_system function, calling it...")
            chatbot.initialize_system()
        elif hasattr(chatbot, 'setup_rag_system'):
            logger.info("Found setup_rag_system function, calling it...")
            chatbot.setup_rag_system()
        else:
            logger.info("No explicit initialization function found, checking for global variables...")
            # Try to trigger initialization by accessing global variables
            if hasattr(chatbot, 'passages') and hasattr(chatbot, 'embedder') and hasattr(chatbot, 'faiss_index'):
                logger.info("Found global RAG components")
            else:
                logger.warning("RAG components may not be initialized properly")
        
        # Your chatbot.py creates a function called 'medical_chatbot'
        if hasattr(chatbot, 'medical_chatbot'):
            rag_system = chatbot.medical_chatbot
            
            # Test the function to ensure it's working
            logger.info("Testing medical_chatbot function...")
            test_response = rag_system("test query")
            logger.info(f"Test response: {test_response}")
            
            system_loaded = True
            logger.info("✅ medical_chatbot function loaded and tested successfully")
            return True
        else:
            logger.error("❌ medical_chatbot function not found in chatbot.py")
            logger.info(f"Available functions: {[name for name in dir(chatbot) if not name.startswith('_')]}")
            return False
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def safe_chatbot_call(query):
    """Safe wrapper for chatbot calls with error handling"""
    try:
        if not system_loaded or rag_system is None:
            logger.error("RAG system not loaded or is None")
            return {
                "result": "RAG system not properly initialized",
                "context_check": "error",
                "confidence": 0.0
            }
        
        logger.info(f"Calling rag_system with query: {query}")
        logger.info(f"rag_system type: {type(rag_system)}")
        
        # Call your existing medical_chatbot function
        response = rag_system(query)
        
        logger.info(f"Raw response type: {type(response)}")
        logger.info(f"Raw response: {response}")
        
        # Ensure response has required fields
        if isinstance(response, dict):
            # Extract confidence from your system's response
            confidence = 0.8  # Default confidence
            
            # Try to extract confidence from various possible fields
            if 'confidence' in response:
                confidence = response['confidence']
            elif 'retrieved_passages' in response and response['retrieved_passages']:
                # Calculate confidence from retrieval scores
                scores = [p.get('score', 0.5) for p in response['retrieved_passages']]
                confidence = max(scores) if scores else 0.5
            elif response.get('context_check') == 'passed':
                confidence = 0.8
            elif response.get('context_check') == 'no_docs':
                confidence = 0.3
            else:
                confidence = 0.5
                
            response['confidence'] = confidence
            return response
        else:
            # If response is just a string
            return {
                "result": str(response),
                "context_check": "passed",
                "confidence": 0.7
            }
            
    except Exception as e:
        logger.error(f"Chatbot call error: {str(e)}")
        return {
            "result": "I apologize, but I encountered an error processing your question. Please try again.",
            "context_check": "error",
            "confidence": 0.0,
            "error": str(e)
        }

# Replace @app.before_first_request with @app.before_request
@app.before_request
def ensure_initialization():
    """Initialize RAG system before first request"""
    global system_loaded, _initialization_attempted
    
    if not _initialization_attempted:
        logger.info("Starting RAG system initialization...")
        success = initialize_rag_system()
        if not success:
            logger.error("Failed to initialize RAG system on startup")

@app.route('/health', methods=['GET'])
def health_check():
    """Required health endpoint"""
    try:
        # Ensure system is initialized
        if not system_loaded:
            initialize_rag_system()
            
        # Check if system is loaded
        gpu_info = ""
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                gpu_info = f"GPU: {allocated:.1f}GB/{total:.1f}GB"
            except:
                gpu_info = "GPU: Available but info unavailable"
        
        status = {
            "status": "healthy" if system_loaded else "initializing",
            "timestamp": datetime.utcnow().isoformat(),
            "model_loaded": system_loaded,
            "service": "Memory-Optimized RAG Chatbot",
            "gpu_memory": gpu_info,
            "initialization_attempted": _initialization_attempted
        }
        
        return jsonify(status), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Required predict endpoint for RAG chatbot"""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json"
            }), 400
        
        data = request.get_json()
        
        # Validate required fields
        if 'query' not in data:
            return jsonify({
                "error": "Missing required field: 'query'"
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                "error": "Query cannot be empty"
            }), 400
        
        # Initialize system if not done yet
        if not system_loaded:
            logger.info("System not loaded, attempting initialization...")
            if not initialize_rag_system():
                return jsonify({
                    "error": "RAG system not available",
                    "details": "Failed to initialize the RAG system. Please check server logs."
                }), 500
        
        # Generate response using your RAG system
        response_data = safe_chatbot_call(query)
        
        # Format response according to requirements
        result = {
            "response": response_data.get("result", "No response generated"),
            "confidence": float(response_data.get("confidence", 0.5)),
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "context_check": response_data.get("context_check", "unknown"),
            "model_used": response_data.get("model_used", "memory-optimized"),
            "rewritten_query": response_data.get("rewritten_query", query)
        }
        
        # Include retrieved passages if available (for debugging)
        if "retrieved_passages" in response_data:
            result["passages_found"] = len(response_data["retrieved_passages"])
        
        # Include error info if present
        if "error" in response_data:
            result["error"] = response_data["error"]
        
        logger.info(f"Processed query: {query[:50]}... | Confidence: {result['confidence']}")
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Internal server error",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@app.route('/info', methods=['GET'])
def model_info():
    """Additional endpoint to provide model information"""
    try:
        info = {
            "model_name": "Memory-Optimized RAG Chatbot",
            "model_type": "Retrieval-Augmented Generation",
            "version": "1.0.0",
            "optimization": "4-bit quantized models, memory efficient",
            "components": {
                "embedder": "sentence-transformers/all-MiniLM-L6-v2",
                "query_rewriter": "google/gemma-2-2b-it (quantized)",
                "answer_generator": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "vector_store": "FAISS index",
                "fallback_models": ["DistilGPT2", "T5-small"]
            },
            "endpoints": {
                "/health": "GET - Health check",
                "/predict": "POST - Generate RAG response",
                "/info": "GET - Model information"
            },
            "input_format": {
                "query": "string - Medical question/query"
            },
            "output_format": {
                "response": "string - Generated medical response",
                "confidence": "float - Confidence score (0.0-1.0)",
                "query": "string - Original query",
                "timestamp": "string - Response timestamp"
            },
            "system_status": {
                "loaded": system_loaded,
                "memory_optimized": True,
                "gpu_available": torch.cuda.is_available(),
                "initialization_attempted": _initialization_attempted
            }
        }
        return jsonify(info), 200
        
    except Exception as e:
        logger.error(f"Info endpoint error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/initialize', methods=['POST'])
def force_initialize():
    """Manual initialization endpoint for debugging"""
    global _initialization_attempted, system_loaded
    try:
        _initialization_attempted = False  # Reset flag
        system_loaded = False
        
        success = initialize_rag_system()
        
        return jsonify({
            "success": success,
            "system_loaded": system_loaded,
            "message": "Initialization completed" if success else "Initialization failed",
            "timestamp": datetime.utcnow().isoformat()
        }), 200 if success else 500
        
    except Exception as e:
        logger.error(f"Force initialization error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": ["/health", "/predict", "/info", "/initialize"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "timestamp": datetime.utcnow().isoformat()
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Memory-Optimized RAG Chatbot service on port {port}")
    logger.info(f"Debug mode: {debug}")
    
    # Try to initialize system before starting server
    logger.info("Pre-startup initialization attempt...")
    initialize_rag_system()
    
    app.run(host='0.0.0.0', port=port, debug=debug)