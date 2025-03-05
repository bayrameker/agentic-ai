"""
Main module for the Multi-Model LLM Agent system.
Demonstrates how to use the framework with examples.
"""

import argparse
import os
import logging
from models import (
    ModelManager, DummyModel, OpenAIModel, OllamaModel, 
    AnthropicModel, GoogleGeminiModel, MistralModel, DeepSeekV2Model, 
    LLaMAModel, Phi4Model
)
from agent import Agent
from tasks import Task

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

def setup_models(config_file=None, discover_local=True):
    """
    Set up and register models with the model manager.
    
    Args:
        config_file (str, optional): Path to a model configuration file
        discover_local (bool, optional): Whether to discover local models like Ollama
        
    Returns:
        ModelManager: Configured model manager with registered models
    """
    # Create model manager
    model_manager = ModelManager()
    
    # Always register a dummy model for testing (without API keys)
    model_manager.add_model("default_model", DummyModel("default_dummy"))
    
    # Try to load models from configuration file if provided
    if config_file:
        # Check if the config file exists
        if os.path.isfile(config_file):
            logger.info(f"Loading models from configuration file: {config_file}")
            num_models = model_manager.load_from_config(config_file)
            logger.info(f"Loaded {num_models} models from configuration")
        else:
            logger.warning(f"Configuration file not found: {config_file}")

    # If no models were loaded from config or no config was provided,
    # fall back to loading from environment variables
    if len(model_manager.list_models()) <= 1:  # Just the dummy model
        logger.info("Loading models from environment variables")
        # Get API keys from environment variables
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        mistral_api_key = os.environ.get("MISTRAL_API_KEY")
        deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
        
        # Example: OpenAI models
        if openai_api_key:
            model_manager.add_model(
                "gpt3", 
                OpenAIModel(
                    api_key=openai_api_key, 
                    model_name="gpt-3.5-turbo", 
                    temperature=0.7, 
                    max_tokens=1000
                )
            )
            model_manager.add_model(
                "gpt4", 
                OpenAIModel(
                    api_key=openai_api_key, 
                    model_name="gpt-4-turbo-2024-04-09",
                    temperature=0.7, 
                    max_tokens=1500
                )
            )
            model_manager.add_model(
                "gpt4o", 
                OpenAIModel(
                    api_key=openai_api_key, 
                    model_name="gpt-4o",
                    temperature=0.5, 
                    max_tokens=2000
                )
            )
        
        # Example: Anthropic Claude 3 models
        if anthropic_api_key:
            model_manager.add_model(
                "claude_opus", 
                AnthropicModel(
                    api_key=anthropic_api_key, 
                    model_name="claude-3-opus-20240229",
                    max_tokens=2000,
                    temperature=0.7
                )
            )
            model_manager.add_model(
                "claude_sonnet", 
                AnthropicModel(
                    api_key=anthropic_api_key, 
                    model_name="claude-3-sonnet-20240229",
                    max_tokens=1500,
                    temperature=0.7
                )
            )
        
        # Example: Google Gemini models
        if google_api_key:
            model_manager.add_model(
                "gemini", 
                GoogleGeminiModel(
                    api_key=google_api_key, 
                    model_name="gemini-1.5-pro-latest",
                    temperature=0.7
                )
            )
        
        # Example: Mistral AI models
        if mistral_api_key:
            model_manager.add_model(
                "mistral_large", 
                MistralModel(
                    api_key=mistral_api_key, 
                    model_name="mistral-large-latest",
                    temperature=0.7,
                    max_tokens=1000,
                    safe_prompt=True
                )
            )
        
        # Example: DeepSeek V2 models
        if deepseek_api_key:
            model_manager.add_model(
                "deepseek", 
                DeepSeekV2Model(
                    api_key=deepseek_api_key, 
                    model_name="deepseek-chat",
                    temperature=0.3,
                    max_tokens=1000
                )
            )
    
    # Optionally discover and add local models
    if discover_local:
        logger.info("Discovering local Ollama models...")
        ollama_models = model_manager.discover_ollama_models()
        if ollama_models:
            logger.info(f"Found {len(ollama_models)} Ollama models: {', '.join(ollama_models)}")
        else:
            logger.info("No local Ollama models found or Ollama is not running")
    
    return model_manager

def example_direct_task_creation():
    """Example showing how to create tasks directly."""
    # Setup
    model_manager = setup_models(config_file="models_config.yaml")
    agent = Agent(model_manager)
    
    # Create tasks directly
    task1 = Task("summarization", "Bugün hava çok güzel ve ben dışarı çıkıp yürüyüş yapmak istiyorum.")
    task2 = Task("sentiment_analysis", "Bu film gerçekten harika! En sevdiğim yönetmenin eseri.")
    
    # Add tasks to agent
    agent.add_task(task1)
    agent.add_task(task2)
    
    # Run all tasks and print results
    results = agent.run_all_tasks(output_format="text")
    print("\n=== DIRECT TASK CREATION EXAMPLE ===")
    print(results)

def example_request_parsing():
    """Example showing how to parse a user request into tasks."""
    # Setup
    model_manager = setup_models(config_file="models_config.yaml")
    agent = Agent(model_manager)
    
    # Process a natural language request
    user_request = "Lütfen bu metni özetle ve duygu analizini yap: Bugün işe giderken yağmura yakalandım ve ıslandım. Bu yüzden biraz sinirli hissediyorum."
    print("\n=== REQUEST PARSING EXAMPLE ===")
    print(f"User request: {user_request}")
    
    # Let the agent plan tasks from the request
    tasks = agent.plan_tasks_from_request(user_request)
    print(f"Planned {len(tasks)} tasks:")
    for task in tasks:
        print(f"- {task}")
    
    # Run the tasks
    results = agent.run_all_tasks(output_format="text")
    print("\nResults:")
    print(results)

def example_json_output():
    """Example showing JSON output format."""
    # Setup
    model_manager = setup_models(config_file="models_config.yaml")
    agent = Agent(model_manager)
    
    # Add a task
    agent.add_task(Task("information_extraction", "John Doe, doğum tarihi 15 Mayıs 1985, İstanbul'da yaşıyor ve yazılım mühendisi olarak çalışıyor."))
    
    # Get results in JSON format
    results = agent.run_all_tasks(output_format="json")
    print("\n=== JSON OUTPUT EXAMPLE ===")
    print(results)

def example_custom_model_assignment():
    """Example showing how to use different models for different tasks."""
    # Setup
    model_manager = setup_models(config_file="models_config.yaml")
    agent = Agent(model_manager)
    
    # Add some dummy models for demonstration
    model_manager.add_model("special_model", DummyModel("special_dummy"))
    model_manager.add_model("summary_model", DummyModel("summary_specialist"))
    model_manager.add_model("sentiment_model", DummyModel("sentiment_specialist"))
    
    # Set different default models for different task types
    agent.set_default_model("summarization", "summary_model")
    agent.set_default_model("sentiment_analysis", "sentiment_model")
    agent.set_default_model("information_extraction", "special_model")
    
    # Create tasks
    agent.add_task(Task("summarization", "Bu bir uzun metindir ve özetlenmesi gerekiyor."))
    agent.add_task(Task("sentiment_analysis", "Bu ürün beklentilerimi karşılamadı ve oldukça hayal kırıklığına uğradım."))
    agent.add_task(Task("information_extraction", "Toplantı 15 Haziran 2023 tarihinde saat 14:00'te İstanbul ofisinde yapılacak."))
    
    # Run tasks with assigned models
    results = agent.run_all_tasks(output_format="text")
    print("\n=== CUSTOM MODEL ASSIGNMENT EXAMPLE ===")
    print(results)

def example_ollama_local_models():
    """Example showing the use of local models with Ollama."""
    # This example assumes you have Ollama installed and running locally
    print("\n=== OLLAMA LOCAL MODELS EXAMPLE ===")
    print("Note: This example requires Ollama to be installed and running.")
    print("If you don't have Ollama, use can use: 'pip install ollama' or visit https://ollama.ai")
    
    # Setup with Ollama models
    model_manager = ModelManager()
    
    # Try to discover and use real Ollama models
    ollama_models = model_manager.discover_ollama_models()
    
    if ollama_models:
        print(f"Found the following Ollama models: {', '.join(ollama_models)}")
        # Use the first two models we found
        if len(ollama_models) >= 1:
            model1_name = f"ollama_{ollama_models[0]}"
            model2_name = model1_name
            if len(ollama_models) >= 2:
                model2_name = f"ollama_{ollama_models[1]}"
                
            # Use these models
            print(f"Using models: {model1_name} and {model2_name}")
        else:
            # Use dummy models if not enough real models were found
            model_manager.add_model("llama3_local", DummyModel("llama3_local_simulation"))
            model_manager.add_model("mixtral_local", DummyModel("mixtral_local_simulation"))
            model1_name = "llama3_local"
            model2_name = "mixtral_local"
    else:
        print("No Ollama models found. Using dummy models for demonstration.")
        # Use Dummy models for demonstration
        model_manager.add_model("llama3_local", DummyModel("llama3_local_simulation"))
        model_manager.add_model("mixtral_local", DummyModel("mixtral_local_simulation"))
        model1_name = "llama3_local"
        model2_name = "mixtral_local"
    
    # Create agent and set models
    agent = Agent(model_manager)
    agent.set_default_model("summarization", model1_name)
    agent.set_default_model("sentiment_analysis", model2_name)
    
    # Add tasks
    agent.add_task(Task("summarization", "Yapay zeka, insan zekasını taklit eden sistemlerdir. Günümüzde pek çok alanda kullanılmaktadır."))
    agent.add_task(Task("sentiment_analysis", "Bu yeni teknoloji gerçekten hayatımı kolaylaştırdı. Çok memnunum!"))
    
    # Run tasks
    results = agent.run_all_tasks(output_format="json")
    print(results)

def example_dynamic_model_loading():
    """Example demonstrating dynamic model loading from config file."""
    print("\n=== DYNAMIC MODEL LOADING EXAMPLE ===")
    print("Loading models from configuration file...")
    
    # Load models from config file
    model_manager = setup_models(config_file="models_config.yaml", discover_local=True)
    
    # Print available models
    models = model_manager.list_models()
    print(f"Available models ({len(models)}):")
    for model_name in models:
        model = model_manager.get_model(model_name)
        print(f"  - {model_name}: {type(model).__name__}")
    
    # Create an agent and use some of the models
    agent = Agent(model_manager)
    
    # Try to use specific models if available
    for task_type, model_prefix in [
        ("summarization", "ollama_llama3"),
        ("sentiment_analysis", "claude"),
        ("information_extraction", "gpt4"),
        ("translation", "deepseek"),
    ]:
        # Find a model that matches the prefix
        matching_models = [m for m in models if m.startswith(model_prefix)]
        if matching_models:
            agent.set_default_model(task_type, matching_models[0])
            print(f"Assigned {matching_models[0]} to {task_type} tasks")
    
    # Add a test task
    agent.add_task(Task("summarization", "Yapay zeka teknolojileri son yıllarda büyük ilerleme kaydetti. Özellikle dil modellerindeki gelişmeler, doğal dil işlemede insan benzeri yetenekler göstermeye başladı."))
    
    # Run the task
    try:
        results = agent.run_all_tasks(output_format="text")
        print("\nResults:")
        print(results)
        
        # Check if the result contains an error message
        if isinstance(results, str) and "Error: Could not connect to Ollama server" in results:
            print("\n⚠️ OLLAMA CONNECTION ERROR ⚠️")
            print("The system couldn't connect to the Ollama server. Please check that:")
            print("1. Ollama is installed on your system")
            print("2. The Ollama server is running (try 'ollama serve' in a separate terminal)")
            print("3. The server is accessible at http://localhost:11434")
            print("\nTroubleshooting steps:")
            print("- Visit http://localhost:11434 in your browser to check if Ollama is responding")
            print("- Restart the Ollama server")
            print("- Check Ollama logs for any errors")
        elif isinstance(results, str) and "Error:" in results:
            print("\n⚠️ MODEL ERROR ⚠️")
            print("There was an error when running the model. See logs above for details.")
            
    except Exception as e:
        logger.exception("Error running tasks")
        print(f"\nError: {str(e)}")
        print("Check the logs for more details.")

def api_simulation():
    """Simulate how this would work as an API endpoint."""
    # Setup once at server start
    model_manager = setup_models(config_file="models_config.yaml")
    agent = Agent(model_manager)
    
    print("\n=== API SIMULATION ===")
    
    # Example API request 1: Task via natural language
    request1 = {
        "text": "Bu metni özetle: Python programlama dili, yüksek seviyeli, genel amaçlı bir programlama dilidir. Basit ve okunabilir bir sözdizimi vardır.",
        "output_format": "json"
    }
    
    print(f"\nAPI Request 1: {request1}")
    # Process request
    agent.plan_tasks_from_request(request1["text"])
    response1 = agent.run_all_tasks(output_format=request1["output_format"])
    print(f"API Response 1: {response1}")
    
    # Example API request 2: Direct task specification
    request2 = {
        "tasks": [
            {"type": "sentiment_analysis", "content": "Bugün gerçekten harika bir gün!", "model": "default_model"}
        ],
        "output_format": "text"
    }
    
    print(f"\nAPI Request 2: {request2}")
    # Process request
    for task_data in request2["tasks"]:
        agent.add_task(Task(
            task_data["type"], 
            task_data["content"], 
            model=task_data.get("model")
        ))
    response2 = agent.run_all_tasks(output_format=request2["output_format"])
    print(f"API Response 2: {response2}")
    
    # Example API request 3: Multiple models for different tasks
    request3 = {
        "text": "Bu metni özetle ve duygu analizini yap: Toplantıda projemiz övgü aldı ama birkaç düzeltme de istendi.",
        "model_mapping": {
            "summarization": "default_model",
            "sentiment_analysis": "default_model"
        },
        "output_format": "json"
    }
    
    print(f"\nAPI Request 3: {request3}")
    # Process request with model assignments
    agent.plan_tasks_from_request(request3["text"])
    # Set models according to mapping
    for task_type, model_name in request3["model_mapping"].items():
        agent.set_default_model(task_type, model_name)
    response3 = agent.run_all_tasks(output_format=request3["output_format"])
    print(f"API Response 3: {response3}")

def example_ollama_error_simulation():
    """Example demonstrating Ollama error handling."""
    print("\n=== OLLAMA ERROR HANDLING EXAMPLE ===")
    print("Testing error handling when Ollama server is not available...")
    
    # Create a model manager
    model_manager = ModelManager()
    
    # Add a dummy model as the default
    model_manager.add_model("default_model", DummyModel("test_model"))
    
    # Add an Ollama model with incorrect server URL to simulate connection error
    model_manager.add_model(
        "error_model", 
        OllamaModel(
            model_name="llama3:latest",
            server_url="http://non-existent-server:11434",
            name="ollama_error_test"
        )
    )
    
    # Create an agent and use the error model
    agent = Agent(model_manager)
    agent.set_default_model("summarization", "error_model")
    
    # Add a test task
    agent.add_task(Task("summarization", "Bu metni özetle: Python programlama dili çok güzeldir."))
    
    # Run the task and expect an error
    try:
        results = agent.run_all_tasks(output_format="text")
        print("\nResults:")
        print(results)
        
        # Check if the result contains an error message
        if isinstance(results, str) and "Error: Could not connect to Ollama server" in results:
            print("\n⚠️ OLLAMA CONNECTION ERROR ⚠️")
            print("The system couldn't connect to the Ollama server. Please check that:")
            print("1. Ollama is installed on your system")
            print("2. The Ollama server is running (try 'ollama serve' in a separate terminal)")
            print("3. The server is accessible at the specified URL")
            print("\nTroubleshooting steps:")
            print("- Try 'ollama serve' to start the Ollama server")
            print("- Check if the server is running on the correct port")
            print("- Check firewall settings")
        
    except Exception as e:
        logger.exception("Error running tasks")
        print(f"\nError: {str(e)}")
        print("Check the logs for more details.")

def main():
    """
    Entry point for the LLM integration demo.
    """
    parser = argparse.ArgumentParser(description='Run LLM integration examples.')
    parser.add_argument('--example', choices=['direct', 'parsing', 'json', 'models', 'ollama', 'api', 'dynamic', 'error', 'all'], 
                        help='Example to run')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    print("====================================")
    print("MULTI-MODEL LLM AGENT SYSTEM DEMO")
    print("====================================")
    
    # Load models based on configuration file
    if args.config:
        setup_models(config_file=args.config)
    
    # Run requested example
    if args.example == 'direct':
        example_direct_task_creation()
    elif args.example == 'parsing':
        example_request_parsing()
    elif args.example == 'json':
        example_json_output()
    elif args.example == 'models':
        example_custom_model_assignment()
    elif args.example == 'ollama':
        example_ollama_local_models()
    elif args.example == 'api':
        api_simulation()
    elif args.example == 'dynamic':
        example_dynamic_model_loading()
    elif args.example == 'error':
        example_ollama_error_simulation()
    elif args.example == 'all':
        example_direct_task_creation()
        example_request_parsing()
        example_json_output()
        example_custom_model_assignment()
        example_ollama_local_models()
        example_dynamic_model_loading()
        api_simulation()
    else:
        # Default: Run API simulation
        api_simulation()

if __name__ == "__main__":
    main() 