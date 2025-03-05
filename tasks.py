class Task:
    """
    Represents a specific task to be executed by an LLM.
    Each task has a type, content (input), optional model specification, and stores its result.
    """
    def __init__(self, task_type, content, model=None, priority=0):
        """
        Initialize a new task.
        
        Args:
            task_type (str): Type of task (e.g., "summarization", "sentiment_analysis")
            content (str): Input content/text for the task
            model (str, optional): Specific model to use for this task. If None, a default will be used.
            priority (int, optional): Task priority (higher numbers = higher priority)
        """
        self.task_type = task_type      # e.g., "summarization", "sentiment_analysis"
        self.content = content          # input text/content
        self.model = model              # optional: specific model to use
        self.result = None              # will store the task result after execution
        self.priority = priority        # priority level (higher = more important)
        self.status = "pending"         # task status: "pending", "running", "completed", "failed"
        
    def __str__(self):
        """String representation of the task."""
        status_info = f" ({self.status})" if self.status != "pending" else ""
        return f"Task({self.task_type}{status_info}): {self.content[:30]}..." 