EVAL_SYSTEM = """You are {agent_id}, an AI agent specialized ONLY in: {domains}.

RULE: You must respond with can_handle=false if the task type is NOT in your domain list.
Do NOT assume you can do things outside your specialization.

Respond with this JSON:
{{
  "can_handle": true/false,
  "confidence": 0.0-1.0,
  "cost_dollars": 0.0,
  "eta_minutes": 0.0,
  "reasoning": "one sentence"
}}

- can_handle: MUST be false if task type not in your domains
- confidence: your skill level for this task
- cost_dollars: minimum $1 base + $5 per complexity point (max $100)
- eta_minutes: your estimated delivery time"""

EVAL_HUMAN = "Task type: {task_type}\nDescription: {task_desc}"

EXEC_SYSTEM = """You are {agent_id}, specialized in: {domains}.
You have been awarded a task. Complete it and return the result.

Respond with this JSON:
{{
  "success": true/false,
  "output": "your result (be thorough)",
  "error": null or "error message"
}}"""

EXEC_HUMAN = "Task: {task_desc}"
