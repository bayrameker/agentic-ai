#!/usr/bin/env python
"""
Script to add a new model to the configuration file.
This demonstrates how to dynamically modify the model configuration.
"""

import argparse
import os
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("add_model")

# Available model types
MODEL_TYPES = {
    "openai": ["api_key", "model_name", "temperature", "max_tokens"],
    "anthropic": ["api_key", "model_name", "temperature", "max_tokens"],
    "gemini": ["api_key", "model_name", "temperature"],
    "mistral": ["api_key", "model_name", "temperature", "max_tokens"],
    "deepseek": ["api_key", "model_name", "temperature", "max_tokens"],
    "phi4": ["api_key", "model_name"],
    "ollama": ["model_name", "temperature", "num_ctx"],
    "dummy": []
}

def add_model_to_config(config_file, model_type, model_name, display_name=None, **kwargs):
    """
    Add a new model to the configuration file.
    
    Args:
        config_file (str): Path to the configuration file
        model_type (str): Type of model (openai, anthropic, ollama, etc.)
        model_name (str): Name or identifier of the model
        display_name (str, optional): Name to register the model as
        **kwargs: Additional model-specific parameters
    
    Returns:
        bool: True if successful, False otherwise
    """
    if model_type not in MODEL_TYPES:
        logger.error(f"Unsupported model type: {model_type}")
        print(f"Supported model types: {', '.join(MODEL_TYPES.keys())}")
        return False
    
    # Prepare model config
    model_config = {
        "type": model_type,
        "name": display_name or f"{model_type}_{model_name}"
    }
    
    # Add model-specific parameters
    for param in MODEL_TYPES[model_type]:
        if param == "api_key" and param in kwargs:
            # Handle API key
            value = kwargs[param]
            if value.startswith("env:"):
                model_config[param] = value
            else:
                # Store as environment variable reference
                env_var = f"{model_type.upper()}_{param.upper()}"
                model_config[param] = f"env:{env_var}"
                # Set the environment variable
                os.environ[env_var] = value
        elif param in kwargs:
            model_config[param] = kwargs[param]
    
    # Set model name if needed
    if "model_name" not in model_config and model_type != "dummy":
        model_config["model_name"] = model_name
    
    # Load existing config or create new
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error reading config file: {str(e)}")
            return False
    else:
        config = {}
    
    # Initialize models list if it doesn't exist
    if 'models' not in config:
        config['models'] = []
    
    # Check for duplicates by name
    existing_names = [m.get('name') for m in config['models']]
    if model_config['name'] in existing_names:
        logger.warning(f"A model with name '{model_config['name']}' already exists. Updating it.")
        for i, model in enumerate(config['models']):
            if model.get('name') == model_config['name']:
                config['models'][i] = model_config
                break
    else:
        # Add new model
        config['models'].append(model_config)
    
    # Save config
    try:
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Successfully added/updated model '{model_config['name']}' in {config_file}")
        return True
    except Exception as e:
        logger.error(f"Error writing config file: {str(e)}")
        return False

def list_models_in_config(config_file):
    """
    List all models in the configuration file.
    
    Args:
        config_file (str): Path to the configuration file
        
    Returns:
        list: List of model configurations
    """
    if not os.path.exists(config_file):
        logger.warning(f"Config file does not exist: {config_file}")
        return []
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f) or {}
        
        models = config.get('models', [])
        return models
    except Exception as e:
        logger.error(f"Error reading config file: {str(e)}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Add a model to the configuration file")
    
    # Required arguments
    parser.add_argument("action", choices=["add", "list"], help="Action to perform")
    
    # Add model arguments
    parser.add_argument("--type", choices=list(MODEL_TYPES.keys()), 
                      help="Type of model to add")
    parser.add_argument("--model", help="Model name/identifier")
    parser.add_argument("--name", help="Display name for the model (defaults to type_model)")
    
    # Optional parameters
    parser.add_argument("--api-key", help="API key or env:VAR_NAME to use environment variable")
    parser.add_argument("--temperature", type=float, help="Temperature setting")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens")
    parser.add_argument("--num-ctx", type=int, help="Context window size (for Ollama)")
    
    # Config file
    parser.add_argument("--config", default="models_config.yaml", 
                      help="Path to configuration file")
    
    args = parser.parse_args()
    
    if args.action == "list":
        models = list_models_in_config(args.config)
        
        if not models:
            print(f"No models found in {args.config}")
        else:
            print(f"Models in {args.config}:")
            for i, model in enumerate(models, 1):
                model_type = model.get("type", "unknown")
                name = model.get("name", "unnamed")
                model_name = model.get("model_name", "unspecified")
                
                print(f"{i}. {name} ({model_type}) - {model_name}")
                # Print additional parameters
                for key, value in model.items():
                    if key not in ["type", "name", "model_name"]:
                        # Hide full API keys for security
                        if key == "api_key" and isinstance(value, str) and not value.startswith("env:"):
                            value = value[:4] + "..." + value[-4:] if len(value) > 8 else "****"
                        print(f"   - {key}: {value}")
    
    elif args.action == "add":
        if not args.type or not args.model:
            parser.error("Both --type and --model are required for the 'add' action")
        
        # Collect model parameters
        kwargs = {}
        if args.api_key:
            kwargs["api_key"] = args.api_key
        if args.temperature:
            kwargs["temperature"] = args.temperature
        if args.max_tokens:
            kwargs["max_tokens"] = args.max_tokens
        if args.num_ctx:
            kwargs["num_ctx"] = args.num_ctx
        
        success = add_model_to_config(
            args.config,
            args.type,
            args.model,
            args.name,
            **kwargs
        )
        
        if success:
            print(f"Model '{args.name or f'{args.type}_{args.model}'}' added successfully")
            print("You can now use this model with:")
            print(f"  python main.py --config {args.config}")
        else:
            print("Failed to add model. See log for details.")

if __name__ == "__main__":
    main() 