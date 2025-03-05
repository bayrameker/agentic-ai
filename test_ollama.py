import logging
import sys
from models import OllamaModel

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Create an Ollama model instance
model = OllamaModel(
    model_name="llama3.1:8b",  # Using a smaller model for faster response
    name="ollama_test",
    temperature=0.7,
    num_ctx=2048,
    stream=False
)

# Test prompt
prompt = "Summarize the following in one sentence: Artificial intelligence has seen rapid advancements in recent years."

# Generate response
print("\n--- Testing Ollama Model ---")
response = model.generate(prompt)
print("\n--- Response ---")
print(response)
print("\n--- Test Complete ---") 