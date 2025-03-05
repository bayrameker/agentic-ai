"""
LLM Model implementations and interface for the multi-model agent system.
This module defines a common interface for different LLM providers and implementations
for specific services like OpenAI, LLaMA, DeepSeek, and Phi-4.
"""

import json
import os
import requests
import logging
import importlib
import subprocess
from typing import Dict, List, Optional, Union, Any, Type
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelRegistry")

class LLMModel:
    """Base class interface for all LLM models."""
    def __init__(self, name):
        """
        Initialize the base model.
        
        Args:
            name (str): Unique identifier for this model instance
        """
        self.name = name
        
    def generate(self, prompt):
        """
        Generate a response from the LLM based on the provided prompt.
        
        Args:
            prompt (str): The input prompt including system instructions and user content
            
        Returns:
            str: The generated response from the model
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Each model must implement its own generate method")
    
    def __str__(self):
        return f"LLMModel({self.name})"


class OpenAIModel(LLMModel):
    """Implementation for OpenAI's API (GPT models)."""
    def __init__(self, api_key, model_name="gpt-3.5-turbo", name=None, temperature=0.7, max_tokens=1000):
        """
        Initialize an OpenAI model.
        
        Args:
            api_key (str): OpenAI API key
            model_name (str, optional): The specific OpenAI model to use
            name (str, optional): Custom identifier (defaults to model_name)
            temperature (float, optional): Controls randomness (0-2)
            max_tokens (int, optional): Maximum tokens to generate
        """
        super().__init__(name or f"openai_{model_name}")
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    def generate(self, prompt):
        """
        Generate text using OpenAI's API.
        
        Args:
            prompt (str): Input prompt
            
        Returns:
            str: Generated text
        """
        try:
            # Split the prompt into system and user parts if possible
            if "Kullanıcı verisi:" in prompt:
                parts = prompt.split("Kullanıcı verisi:", 1)
                system_prompt = parts[0].strip()
                user_prompt = parts[1].strip()
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            else:
                # If no explicit split, use the whole prompt as user message
                messages = [{"role": "user", "content": prompt}]
            
            # In real implementation:
            # from openai import OpenAI
            # client = OpenAI(api_key=self.api_key)
            # response = client.chat.completions.create(
            #     model=self.model_name,
            #     messages=messages,
            #     temperature=self.temperature,
            #     max_tokens=self.max_tokens
            # )
            # return response.choices[0].message.content
            
            # For this demo, just return a placeholder
            return f"[OpenAI {self.model_name} would generate response for this prompt]"
            
        except Exception as e:
            return f"Error: Failed to generate with OpenAI: {str(e)}"


class OllamaModel(LLMModel):
    """Implementation for local Ollama models (Llama, Mixtral, etc.)"""
    def __init__(self, model_name="llama3:70b", server_url="http://localhost:11434", 
                 name=None, temperature=0.7, num_ctx=4096, stream=False):
        """
        Initialize an Ollama model.
        
        Args:
            model_name (str): The name of the Ollama model (e.g., 'llama3:70b', 'mixtral:8x22b')
            server_url (str): The URL of the Ollama server
            name (str, optional): Custom identifier
            temperature (float, optional): Controls randomness
            num_ctx (int, optional): Context window size
            stream (bool, optional): Whether to stream the response
        """
        super().__init__(name or f"ollama_{model_name}")
        self.model_name = model_name
        self.server_url = server_url
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.stream = stream
        self.logger = logging.getLogger(f"OllamaModel_{self.model_name}")
        
    def generate(self, prompt):
        """
        Generate text using a local Ollama model.
        
        Args:
            prompt (str): Input prompt
            
        Returns:
            str: Generated text
        """
        try:
            self.logger.info(f"Generating response with {self.model_name}")
            self.logger.debug(f"Prompt: {prompt[:100]}...")
            
            import requests
            import json
            import time
            
            # Full API call with options
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": self.stream,
                "options": {
                    "temperature": self.temperature,
                    "num_ctx": self.num_ctx
                }
            }
            
            self.logger.info(f"Sending request to Ollama server at {self.server_url}/api/generate")
            start_time = time.time()
            
            # Handle streaming or non-streaming response
            try:
                if self.stream:
                    self.logger.info("Using streaming response")
                    response = requests.post(
                        f"{self.server_url}/api/generate",
                        json=payload,
                        stream=True,
                        timeout=30  # Add timeout for connection
                    )
                    
                    if response.status_code != 200:
                        self.logger.error(f"Ollama API returned status code {response.status_code}")
                        return f"Error: Ollama API returned status code {response.status_code}"
                    
                    response_text = ""
                    for line in response.iter_lines():
                        if line:
                            chunk = json.loads(line.decode("utf-8"))
                            response_text += chunk.get("response", "")
                    
                    elapsed_time = time.time() - start_time
                    self.logger.info(f"Streaming response completed in {elapsed_time:.2f} seconds")
                    self.logger.debug(f"Response: {response_text[:100]}...")
                    return response_text
                else:
                    self.logger.info("Using non-streaming response")
                    response = requests.post(
                        f"{self.server_url}/api/generate",
                        json=payload,
                        timeout=60  # Add timeout for completion
                    )
                    
                    if response.status_code != 200:
                        self.logger.error(f"Ollama API returned status code {response.status_code}: {response.text}")
                        return f"Error: Ollama API returned status code {response.status_code}"
                    
                    response_json = response.json()
                    elapsed_time = time.time() - start_time
                    self.logger.info(f"Response received in {elapsed_time:.2f} seconds")
                    self.logger.debug(f"Response: {response_json.get('response', '')[:100]}...")
                    return response_json.get("response", "")
            except requests.exceptions.ConnectionError:
                self.logger.error(f"Connection error: Could not connect to Ollama server at {self.server_url}")
                return f"Error: Could not connect to Ollama server at {self.server_url}. Is Ollama running?"
            except requests.exceptions.Timeout:
                self.logger.error("Request timed out while waiting for Ollama response")
                return "Error: Request timed out while waiting for Ollama response"
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse Ollama JSON response: {response.text[:200]}")
                return f"Error: Invalid JSON response from Ollama API"
                
        except Exception as e:
            self.logger.exception(f"Unexpected error during Ollama generation: {str(e)}")
            
            # Fall back to placeholder if actual API call fails
            self.logger.warning("Falling back to placeholder response")
            return f"[Ollama {self.model_name} would generate a response for: {prompt[:50]}...] (Error: {str(e)})"

    @classmethod
    def list_available_models(cls, server_url="http://localhost:11434"):
        """
        List all available models in the Ollama server.
        
        Args:
            server_url (str): The URL of the Ollama server
            
        Returns:
            list: List of available model names
        """
        model_logger = logging.getLogger("OllamaModelDiscovery")
        model_logger.info(f"Attempting to discover available Ollama models from {server_url}")
        
        try:
            import requests
            import time
            
            model_logger.info("Sending request to Ollama API to list available models")
            start_time = time.time()
            
            try:
                response = requests.get(f"{server_url}/api/tags", timeout=10)
                elapsed_time = time.time() - start_time
                model_logger.info(f"Ollama API response received in {elapsed_time:.2f} seconds")
                
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [model.get("name") for model in models]
                    
                    if model_names:
                        model_logger.info(f"Successfully discovered {len(model_names)} Ollama models: {', '.join(model_names[:5])}{' and more' if len(model_names) > 5 else ''}")
                    else:
                        model_logger.warning("No models found in Ollama server response")
                    
                    return model_names
                else:
                    model_logger.warning(f"Failed to list Ollama models: HTTP {response.status_code} - {response.text}")
                    return []
            except requests.exceptions.ConnectionError:
                model_logger.error(f"Connection error: Could not connect to Ollama server at {server_url}")
                model_logger.error("Is the Ollama server running? Check with 'ollama serve' or by visiting the server URL in a browser")
                return []
            except requests.exceptions.Timeout:
                model_logger.error("Request timed out while waiting for Ollama server response")
                return []
            except requests.exceptions.RequestException as e:
                model_logger.error(f"Request error while connecting to Ollama server: {str(e)}")
                return []
            except ValueError as e:  # Includes JSONDecodeError
                model_logger.error(f"Failed to parse Ollama server response: {str(e)}")
                return []
                
        except Exception as e:
            model_logger.exception(f"Unexpected error while discovering Ollama models: {str(e)}")
            return []


class AnthropicModel(LLMModel):
    """Implementation for Anthropic Claude models."""
    def __init__(self, api_key, model_name="claude-3-opus-20240229", 
                 name=None, max_tokens=1000, temperature=0.7, stream=False):
        """
        Initialize an Anthropic Claude model.
        
        Args:
            api_key (str): Anthropic API key
            model_name (str): The Claude model version to use
            name (str, optional): Custom identifier
            max_tokens (int, optional): Maximum tokens to generate
            temperature (float, optional): Controls randomness
            stream (bool, optional): Whether to stream the response
        """
        super().__init__(name or f"anthropic_{model_name}")
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stream = stream
        
    def generate(self, prompt):
        """
        Generate text using Anthropic's Claude API.
        
        Args:
            prompt (str): Input prompt
            
        Returns:
            str: Generated text
        """
        try:
            # Split the prompt into system and user parts if possible
            user_message = prompt
            if "Kullanıcı verisi:" in prompt:
                user_message = prompt.split("Kullanıcı verisi:", 1)[1].strip()
            
            # In a real implementation:
            # from anthropic import Anthropic
            
            # client = Anthropic(api_key=self.api_key)
            
            # if self.stream:
            #     # For streaming:
            #     response_text = ""
            #     with client.messages.stream(
            #         model=self.model_name,
            #         max_tokens=self.max_tokens,
            #         messages=[{"role": "user", "content": user_message}],
            #         temperature=self.temperature
            #     ) as stream:
            #         for chunk in stream:
            #             if chunk.content:
            #                 response_text += chunk.content[0].text
            #     return response_text
            # else:
            #     # For non-streaming:
            #     response = client.messages.create(
            #         model=self.model_name,
            #         max_tokens=self.max_tokens,
            #         messages=[{"role": "user", "content": user_message}],
            #         temperature=self.temperature
            #     )
            #     return response.content[0].text
            
            # For this demo, just return a placeholder
            return f"[Anthropic {self.model_name} would generate a response for this prompt]"
            
        except Exception as e:
            return f"Error: Failed to generate with Anthropic: {str(e)}"


class GoogleGeminiModel(LLMModel):
    """Implementation for Google's Gemini models."""
    def __init__(self, api_key, model_name="gemini-1.5-pro-latest", 
                 name=None, temperature=0.7, stream=False):
        """
        Initialize a Google Gemini model.
        
        Args:
            api_key (str): Google AI API key
            model_name (str): Gemini model version
            name (str, optional): Custom identifier
            temperature (float, optional): Controls randomness
            stream (bool, optional): Whether to stream the response
        """
        super().__init__(name or f"gemini_{model_name}")
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.stream = stream
        
    def generate(self, prompt):
        """
        Generate text using Google's Gemini API.
        
        Args:
            prompt (str): Input prompt
            
        Returns:
            str: Generated text
        """
        try:
            # In a real implementation:
            # import google.generativeai as genai
            
            # genai.configure(api_key=self.api_key)
            # model = genai.GenerativeModel(self.model_name)
            
            # generation_config = {
            #     "temperature": self.temperature,
            # }
            
            # if self.stream:
            #     response = model.generate_content(
            #         prompt,
            #         generation_config=generation_config,
            #         stream=True
            #     )
            #     response_text = ""
            #     for chunk in response:
            #         response_text += chunk.text
            #     return response_text
            # else:
            #     response = model.generate_content(
            #         prompt,
            #         generation_config=generation_config
            #     )
            #     return response.text
            
            # For this demo, just return a placeholder
            return f"[Google Gemini {self.model_name} would generate a response for this prompt]"
            
        except Exception as e:
            return f"Error: Failed to generate with Google Gemini: {str(e)}"


class MistralModel(LLMModel):
    """Implementation for Mistral AI models."""
    def __init__(self, api_key, model_name="mistral-large-latest", 
                 name=None, temperature=0.7, max_tokens=1000, safe_prompt=True):
        """
        Initialize a Mistral AI model.
        
        Args:
            api_key (str): Mistral API key
            model_name (str): The model version to use
            name (str, optional): Custom identifier
            temperature (float, optional): Controls randomness
            max_tokens (int, optional): Maximum tokens to generate
            safe_prompt (bool, optional): Whether to enable content filtering
        """
        super().__init__(name or f"mistral_{model_name}")
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.safe_prompt = safe_prompt
        
    def generate(self, prompt):
        """
        Generate text using Mistral AI's API.
        
        Args:
            prompt (str): Input prompt
            
        Returns:
            str: Generated text
        """
        try:
            # In a real implementation:
            # import requests
            
            # headers = {"Authorization": f"Bearer {self.api_key}"}
            # data = {
            #     "model": self.model_name,
            #     "messages": [{"role": "user", "content": prompt}],
            #     "temperature": self.temperature,
            #     "max_tokens": self.max_tokens,
            #     "safe_prompt": self.safe_prompt
            # }
            
            # response = requests.post(
            #     "https://api.mistral.ai/v1/chat/completions",
            #     headers=headers,
            #     json=data
            # )
            
            # response.raise_for_status()  # Raise exception for error status codes
            # return response.json()["choices"][0]["message"]["content"]
            
            # For this demo, just return a placeholder
            return f"[Mistral AI {self.model_name} would generate a response for this prompt]"
            
        except Exception as e:
            return f"Error: Failed to generate with Mistral AI: {str(e)}"


class DeepSeekV2Model(LLMModel):
    """Implementation for DeepSeek-V2 models with 128k context window."""
    def __init__(self, api_key, model_name="deepseek-chat", 
                 name=None, temperature=0.3, max_tokens=1000):
        """
        Initialize a DeepSeek-V2 model.
        
        Args:
            api_key (str): DeepSeek API key
            model_name (str): The model version to use
            name (str, optional): Custom identifier
            temperature (float, optional): Controls randomness
            max_tokens (int, optional): Maximum tokens to generate
        """
        super().__init__(name or f"deepseek_{model_name}")
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    def generate(self, prompt):
        """
        Generate text using DeepSeek's API.
        
        Args:
            prompt (str): Input prompt
            
        Returns:
            str: Generated text
        """
        try:
            # In a real implementation:
            # import requests
            
            # headers = {
            #     "Authorization": f"Bearer {self.api_key}",
            #     "Content-Type": "application/json"
            # }
            
            # data = {
            #     "model": self.model_name,
            #     "messages": [{"role": "user", "content": prompt}],
            #     "temperature": self.temperature,
            #     "max_tokens": self.max_tokens
            # }
            
            # response = requests.post(
            #     "https://api.deepseek.com/v1/chat/completions",
            #     headers=headers,
            #     json=data
            # )
            
            # response.raise_for_status()
            # return response.json()["choices"][0]["message"]["content"]
            
            # For this demo, just return a placeholder
            return f"[DeepSeek-V2 {self.model_name} would generate a response for this prompt]"
            
        except Exception as e:
            return f"Error: Failed to generate with DeepSeek: {str(e)}"


class LLaMAModel(LLMModel):
    """Implementation for LLaMA models."""
    def __init__(self, model_path=None, server_url=None, name="llama"):
        """
        Initialize a LLaMA model.
        
        Args:
            model_path (str, optional): Path to local model weights
            server_url (str, optional): URL of LLaMA serving API if using a remote server
            name (str, optional): Custom identifier
        """
        super().__init__(name)
        self.model_path = model_path
        self.server_url = server_url
        
    def generate(self, prompt):
        """Generate text using a LLaMA model."""
        # In a real implementation, connect to a local instance or API server
        # Example with llama.cpp server:
        # import requests
        # response = requests.post(
        #     f"{self.server_url}/completion",
        #     json={"prompt": prompt, "temperature": 0.7, "max_tokens": 500}
        # )
        # return response.json()["content"]
        
        return f"[LLaMA would generate a response for: {prompt[:50]}...]"


class Phi4Model(LLMModel):
    """Implementation for Microsoft's Phi-4 models."""
    def __init__(self, api_key, model_name="phi-4", name=None):
        """Initialize a Phi-4 model."""
        super().__init__(name or f"phi_{model_name}")
        self.api_key = api_key
        self.model_name = model_name
        
    def generate(self, prompt):
        """Generate text using Phi-4's API."""
        # In a real implementation, use Microsoft's API
        return f"[Phi-4 would generate a response for: {prompt[:50]}...]"


class DummyModel(LLMModel):
    """
    A simple model that returns predetermined responses for testing.
    Useful for development and testing without using actual API calls.
    """
    def __init__(self, name="dummy"):
        """Initialize a dummy model for testing."""
        super().__init__(name)
        
    def generate(self, prompt):
        """Generate a fake response based on prompt contents."""
        # Return different responses based on the task type mentioned in the prompt
        if "Özetleyici:" in prompt or "summary" in prompt.lower():
            content = prompt.split("Kullanıcı verisi:")[1].strip() if "Kullanıcı verisi:" in prompt else prompt
            return f"Özet: {content[:50]}..."
            
        elif "Duygu analizi:" in prompt or "sentiment" in prompt.lower():
            # Return a random sentiment
            import random
            sentiments = ["Pozitif", "Negatif", "Nötr"]
            return f"Duygu analizi sonucu: {random.choice(sentiments)}"
            
        elif "Bilgi çıkarımı:" in prompt or "extraction" in prompt.lower():
            return "Çıkarılan bilgi: [Tarih: 12 Mayıs, Kişi: John Doe, Yer: İstanbul]"
            
        else:
            return f"Dummy model response for: {prompt[:50]}..."


class ModelRegistry:
    """
    Registry for model classes that allows for dynamic registration and instantiation.
    """
    _model_classes = {}  # Maps model type to model class
    
    @classmethod
    def register_model_class(cls, model_type, model_class):
        """
        Register a model class under a specific type name.
        
        Args:
            model_type (str): The type name to register (e.g., 'openai', 'ollama')
            model_class (class): The model class to register
        """
        cls._model_classes[model_type] = model_class
        logger.info(f"Registered model class: {model_type} -> {model_class.__name__}")
    
    @classmethod
    def create_model(cls, model_type, **kwargs):
        """
        Create a model instance based on model type and parameters.
        
        Args:
            model_type (str): The type of model to create
            **kwargs: Arguments to pass to the model constructor
            
        Returns:
            LLMModel: The created model instance
            
        Raises:
            ValueError: If the model type is not registered
        """
        if model_type not in cls._model_classes:
            raise ValueError(f"Model type '{model_type}' not registered")
        
        model_class = cls._model_classes[model_type]
        model = model_class(**kwargs)
        logger.info(f"Created model instance: {model.name} (type: {model_type})")
        return model


class ModelManager:
    """
    Manages a collection of LLM models and provides access to them by name.
    Acts as a registry for all available models in the system.
    """
    def __init__(self):
        """Initialize an empty model registry."""
        self.models = {}  # model_name -> LLMModel instance
        
    def add_model(self, name, model_instance):
        """
        Add a new model to the registry or update an existing one.
        
        Args:
            name (str): Name to register the model under
            model_instance (LLMModel): An instance of a model class
        """
        self.models[name] = model_instance
        logger.info(f"Added model to manager: {name}")
        
    def get_model(self, name):
        """
        Retrieve a model by name.
        
        Args:
            name (str): Name of the model to retrieve
            
        Returns:
            LLMModel or None: The model instance if found, None otherwise
        """
        return self.models.get(name)
        
    def remove_model(self, name):
        """
        Remove a model from the registry.
        
        Args:
            name (str): Name of the model to remove
        """
        if name in self.models:
            del self.models[name]
            logger.info(f"Removed model from manager: {name}")
            
    def list_models(self):
        """
        Get a list of all registered model names.
        
        Returns:
            list: List of model names
        """
        return list(self.models.keys())
    
    def load_from_config(self, config_file):
        """
        Load models from a configuration file.
        
        Args:
            config_file (str): Path to the configuration file (JSON or YAML)
            
        Returns:
            int: Number of models loaded
        """
        try:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                # Default to JSON
                with open(config_file, 'r') as f:
                    config = json.load(f)
            
            # Process the configuration
            models_added = 0
            for model_config in config.get('models', []):
                try:
                    model_type = model_config.pop('type')
                    model_name = model_config.pop('name')
                    
                    # Special handling for API keys - check if they should be loaded from env
                    for key in list(model_config.keys()):
                        if key.endswith('_key') and model_config[key].startswith('env:'):
                            env_var = model_config[key].split(':', 1)[1]
                            model_config[key] = os.environ.get(env_var, '')
                    
                    # Create and add the model
                    model = ModelRegistry.create_model(model_type, **model_config)
                    self.add_model(model_name, model)
                    models_added += 1
                except Exception as e:
                    logger.error(f"Error loading model from config: {str(e)}")
            
            logger.info(f"Loaded {models_added} models from config file: {config_file}")
            return models_added
        except Exception as e:
            logger.error(f"Error loading models from config file: {str(e)}")
            return 0
    
    def discover_ollama_models(self, server_url="http://localhost:11434"):
        """
        Discover and add available Ollama models.
        
        Args:
            server_url (str): URL of the Ollama server
            
        Returns:
            list: Names of discovered models
        """
        logger.info(f"Starting discovery of Ollama models from {server_url}")
        try:
            # Check if Ollama is installed
            is_ollama_available = False
            try:
                logger.info("Checking if Ollama CLI is available")
                result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, check=False)
                is_ollama_available = True
                logger.info(f"Ollama CLI found: {result.stdout.strip()}")
            except FileNotFoundError:
                logger.warning("Ollama CLI not found, will try API directly")
            except Exception as e:
                logger.warning(f"Error checking Ollama CLI: {str(e)}")
            
            # Try to get models from the Ollama server API
            logger.info("Attempting to discover models via Ollama API")
            models = OllamaModel.list_available_models(server_url)
            
            if models:
                logger.info(f"Successfully discovered {len(models)} models via API: {', '.join(models[:5])}{' and more' if len(models) > 5 else ''}")
            elif is_ollama_available:
                # Try using the CLI as a fallback
                logger.info("No models found via API, trying Ollama CLI as fallback")
                try:
                    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    models = [line.split()[0] for line in lines if line.strip()]
                    logger.info(f"Discovered {len(models)} models via CLI: {', '.join(models[:5])}{' and more' if len(models) > 5 else ''}")
                except Exception as e:
                    logger.error(f"Error listing Ollama models via CLI: {str(e)}")
                    logger.debug(f"CLI error details: {traceback.format_exc()}")
            else:
                logger.warning("No Ollama models found and CLI not available")
            
            # Add discovered models to the manager
            added_count = 0
            already_exists_count = 0
            for model_name in models:
                model_id = f"ollama_{model_name}"
                # Only add if not already in the manager
                if model_id not in self.models:
                    logger.debug(f"Adding new Ollama model: {model_name}")
                    model = OllamaModel(model_name=model_name, server_url=server_url)
                    self.add_model(model_id, model)
                    added_count += 1
                else:
                    already_exists_count += 1
                    logger.debug(f"Ollama model already exists: {model_id}")
            
            if added_count > 0:
                logger.info(f"Added {added_count} new Ollama models to the manager")
            if already_exists_count > 0:
                logger.debug(f"{already_exists_count} Ollama models were already registered")
                
            return models
            
        except Exception as e:
            logger.exception(f"Unexpected error during Ollama model discovery: {str(e)}")
            return []


# Register default model classes
ModelRegistry.register_model_class('openai', OpenAIModel)
ModelRegistry.register_model_class('ollama', OllamaModel)
ModelRegistry.register_model_class('anthropic', AnthropicModel)
ModelRegistry.register_model_class('gemini', GoogleGeminiModel)
ModelRegistry.register_model_class('mistral', MistralModel)
ModelRegistry.register_model_class('deepseek', DeepSeekV2Model)
ModelRegistry.register_model_class('llama', LLaMAModel)
ModelRegistry.register_model_class('phi4', Phi4Model)
ModelRegistry.register_model_class('dummy', DummyModel) 