"""
Agent module for the multi-model LLM system.
Handles task planning, execution, and coordination with different models.
"""

import json
import logging
from tasks import Task
import time

class Agent:
    """
    Agent that processes user requests, breaks them down into tasks,
    and coordinates their execution using appropriate LLM models.
    """
    
    def __init__(self, model_manager):
        """
        Initialize the agent with a model manager.
        
        Args:
            model_manager: ModelManager instance to access LLM models
        """
        self.model_manager = model_manager
        self.tasks_queue = []  # List of pending tasks
        
        # System prompts for different task types
        self.system_prompts = {
            "summarization": "Özetleyici: Aşağıdaki metni özetleyiniz.",
            "sentiment_analysis": "Duygu analizi: Aşağıdaki metnin duygusunu (pozitif, negatif, nötr) belirleyiniz.",
            "information_extraction": "Bilgi çıkarımı: Aşağıdaki metinden önemli bilgileri (tarihler, isimler, yerler vb.) çıkarınız.",
            "translation": "Çevirmen: Aşağıdaki metni Türkçe'den İngilizce'ye çeviriniz.",
            "code_generation": "Kod yazıcı: Aşağıdaki istenen görevi gerçekleştiren kodu yazınız.",
            "question_answering": "Soru yanıtlayıcı: Aşağıdaki soruyu cevaplayınız."
        }
        
        # Default model assignments for different task types
        self.default_model_for_task = {
            "summarization": "default_model",
            "sentiment_analysis": "default_model",
            "information_extraction": "default_model",
            "translation": "default_model",
            "code_generation": "default_model",
            "question_answering": "default_model"
        }
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('Agent')
    
    def add_task(self, task):
        """
        Add a task to the queue.
        
        Args:
            task (Task): The task to add
        """
        self.tasks_queue.append(task)
        self.logger.info(f"Added task to queue: {task}")
        
    def set_system_prompt(self, task_type, prompt):
        """
        Set or update the system prompt for a specific task type.
        
        Args:
            task_type (str): The task type to update
            prompt (str): The new system prompt
        """
        self.system_prompts[task_type] = prompt
        self.logger.info(f"Updated system prompt for task type: {task_type}")
        
    def set_default_model(self, task_type, model_name):
        """
        Set the default model for a task type.
        
        Args:
            task_type (str): The task type
            model_name (str): Name of the model to use by default
        """
        self.default_model_for_task[task_type] = model_name
        self.logger.info(f"Set default model for {task_type} to {model_name}")
        
    def plan_tasks_from_request(self, user_request):
        """
        Analyze a user request and plan appropriate tasks.
        
        Args:
            user_request (str): The user's request text
            
        Returns:
            list: The list of created tasks
        """
        tasks = []
        text_to_process = user_request
        
        # Simple keyword-based task planning
        # In a more advanced implementation, this could use an LLM to analyze the request
        
        # Check if request uses the format "command: content"
        if ':' in user_request:
            command_part, content_part = user_request.split(':', 1)
            text_to_process = content_part.strip()
            
            # Check command part for task keywords
            command_lower = command_part.lower()
            if any(kw in command_lower for kw in ['özet', 'özetle', 'summarize', 'summary']):
                tasks.append(Task("summarization", text_to_process))
                
            if any(kw in command_lower for kw in ['duygu', 'sentiment', 'mood']):
                tasks.append(Task("sentiment_analysis", text_to_process))
                
            if any(kw in command_lower for kw in ['bilgi', 'information', 'extract']):
                tasks.append(Task("information_extraction", text_to_process))
                
            if any(kw in command_lower for kw in ['çevir', 'translate']):
                tasks.append(Task("translation", text_to_process))
                
            if any(kw in command_lower for kw in ['kod', 'code']):
                tasks.append(Task("code_generation", text_to_process))
                
            if any(kw in command_lower for kw in ['soru', 'question', 'cevapla', 'answer']):
                tasks.append(Task("question_answering", text_to_process))
        else:
            # If no explicit format, check the entire request
            request_lower = user_request.lower()
            
            if any(kw in request_lower for kw in ['özet', 'özetle', 'summarize', 'summary']):
                tasks.append(Task("summarization", text_to_process))
                
            if any(kw in request_lower for kw in ['duygu', 'sentiment', 'mood']):
                tasks.append(Task("sentiment_analysis", text_to_process))
                
            if any(kw in request_lower for kw in ['bilgi', 'information', 'extract']):
                tasks.append(Task("information_extraction", text_to_process))
                
            if any(kw in request_lower for kw in ['çevir', 'translate']):
                tasks.append(Task("translation", text_to_process))
                
            if any(kw in request_lower for kw in ['kod', 'code']):
                tasks.append(Task("code_generation", text_to_process))
                
            if any(kw in request_lower for kw in ['soru', 'question', 'cevapla', 'answer']):
                tasks.append(Task("question_answering", text_to_process))
        
        # Add tasks to the queue
        for task in tasks:
            self.add_task(task)
            
        return tasks
    
    def run_task(self, task):
        """
        Execute a single task using the appropriate model.
        
        Args:
            task (Task): The task to execute
            
        Returns:
            dict: The task result information
        """
        # Select the model to use (either specified in task or use default)
        model_name = task.model or self.default_model_for_task.get(task.task_type)
        if not model_name:
            error_msg = f"No model specified for task and no default model for task type: {task.task_type}"
            self.logger.error(error_msg)
            task.status = "failed"
            task.result = error_msg
            return {"task": task.task_type, "status": "failed", "result": error_msg}
        
        # Get the model from the manager
        model = self.model_manager.get_model(model_name)
        if not model:
            error_msg = f"Model '{model_name}' not found"
            self.logger.error(error_msg)
            task.status = "failed"
            task.result = error_msg
            return {"task": task.task_type, "status": "failed", "result": error_msg}
        
        try:
            # Get the appropriate system prompt
            system_prompt = self.system_prompts.get(task.task_type, "")
            if not system_prompt:
                self.logger.warning(f"No system prompt found for task type: {task.task_type}")
            
            # Combine system prompt and user content
            prompt = f"{system_prompt}\nKullanıcı verisi: {task.content}"
            
            # Mark task as running
            task.status = "running"
            self.logger.info(f"Running task {task.task_type} with model {model_name}")
            
            # Generate response using the model
            self.logger.debug(f"Sending prompt to model: {prompt[:100]}...")
            start_time = time.time()
            result = model.generate(prompt)
            elapsed_time = time.time() - start_time
            self.logger.info(f"Task {task.task_type} completed in {elapsed_time:.2f} seconds")
            
            # Check for error messages in the result
            if isinstance(result, str) and result.startswith("Error:"):
                self.logger.error(f"Model returned an error: {result}")
                task.status = "failed"
                task.result = result
                return {"task": task.task_type, "status": "failed", "result": result, "model": model_name}
            
            # Update task
            task.status = "completed"
            task.result = result
            self.logger.info(f"Task completed: {task.task_type}")
            
            # Return task info
            return {"task": task.task_type, "status": "completed", "result": result, "model": model_name}
            
        except Exception as e:
            error_message = f"Error executing task: {str(e)}"
            self.logger.exception(error_message)
            task.status = "failed"
            task.result = f"Error: {str(e)}"
            return {"task": task.task_type, "status": "failed", "result": f"Error: {str(e)}", "model": model_name}
    
    def run_all_tasks(self, output_format="text"):
        """
        Execute all tasks in the queue and format the results.
        
        Args:
            output_format (str): Format for results - "text", "json", or "yaml"
            
        Returns:
            str: Formatted results of all tasks
        """
        results = []
        
        # Process all tasks in the queue
        tasks_to_process = list(self.tasks_queue)  # Make a copy to avoid modification issues
        for task in tasks_to_process:
            result = self.run_task(task)
            results.append(result)
            # Remove completed task from queue
            if task in self.tasks_queue:
                self.tasks_queue.remove(task)
        
        # Format and return the results
        if output_format == "json":
            return json.dumps({"results": results}, ensure_ascii=False, indent=2)
            
        elif output_format == "yaml":
            try:
                import yaml
                return yaml.dump({"results": results}, allow_unicode=True)
            except ImportError:
                self.logger.warning("PyYAML not installed, falling back to JSON")
                return json.dumps({"results": results}, ensure_ascii=False, indent=2)
                
        else:  # Default to text format
            text_lines = []
            for res in results:
                status = f"({res['status']})" if res.get('status') != "completed" else ""
                model_info = f" [Model: {res.get('model', 'unknown')}]"
                text_lines.append(f"{res['task']} {status}{model_info}: {res['result']}")
            return "\n\n".join(text_lines) 