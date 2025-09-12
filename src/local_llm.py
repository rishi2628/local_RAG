"""
Local LLM setup using GPT4All
Downloads and configures a local language model for the RAG pipeline.
"""

import os
from pathlib import Path
from gpt4all import GPT4All
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalLLM:
    def __init__(self, model_name="mistral-7b-instruct-v0.1.Q4_0.gguf", model_path="./models"):
        """
        Initialize Local LLM with GPT4All
        
        Args:
            model_name: Name of the model to download
            model_path: Path to store the model
        """
        self.model_name = model_name
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        self.model = None
        
    def download_model(self):
        """Download the specified model if not already present"""
        try:
            logger.info(f"Initializing GPT4All with model: {self.model_name}")
            
            # This will download the model if not present
            self.model = GPT4All(
                model_name=self.model_name,
                model_path=str(self.model_path),
                allow_download=True
            )
            
            logger.info(f"Model {self.model_name} ready for use")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading/loading model: {e}")
            return False
    
    def generate_response(self, prompt, max_tokens=512, temperature=0.7):
        """
        Generate response using the local LLM
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        if not self.model:
            logger.error("Model not loaded. Call download_model() first.")
            return None
            
        try:
            # Format prompt for instruction-following models
            formatted_prompt = f"### Human: {prompt}\n### Assistant:"
            
            with self.model.chat_session():
                response = self.model.generate(
                    prompt=formatted_prompt,
                    max_tokens=max_tokens,
                    temp=temperature,
                    top_p=0.9,
                    repeat_penalty=1.1,
                    streaming=False
                )
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model:
            return {
                "model_name": self.model_name,
                "model_path": str(self.model_path),
                "model_loaded": True
            }
        else:
            return {
                "model_name": self.model_name,
                "model_path": str(self.model_path),
                "model_loaded": False
            }

def main():
    """Test the local LLM setup"""
    # Initialize LLM
    llm = LocalLLM()
    
    # Download model
    logger.info("Downloading/Loading model...")
    success = llm.download_model()
    
    if success:
        logger.info("Model loaded successfully!")
        
        # Test generation
        test_prompt = "What is artificial intelligence?"
        logger.info(f"Testing with prompt: {test_prompt}")
        
        response = llm.generate_response(test_prompt, max_tokens=100)
        if response:
            logger.info(f"Response: {response}")
        
        # Print model info
        info = llm.get_model_info()
        logger.info(f"Model info: {info}")
    else:
        logger.error("Failed to load model")

if __name__ == "__main__":
    main()