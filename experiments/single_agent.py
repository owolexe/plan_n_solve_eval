import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from load_eval import load_eval_problems

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def run_single_agent_eval(problem, agentType ="zero-shot"):
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    agent_messages = {
        "zero-shot": {
            "system": "You are a senior programmer. complete the Python function. You are not allowed to use any import statements. Solve the problem using only core Python primitives. Output ONLY the code within triple backticks.",
            "user": f"Problem:\n{problem['prompt']}"
        },
        "cot": {
            "system": "You are a senior programmer. Explain your logic step-by-step, then provide the final code. You are not allowed to use any import statements. Solve the problem using only core Python primitives.",
            "user": f"Problem:\n{problem['prompt']}\n\nLet's think step by step."
        },
        "plan-and-solve": {
            "system": "You are a senior programmer. First, devise a detailed plan and identify edge cases. Then, implement the code. You are not allowed to use any import statements. Solve the problem using only core Python primitives.",
            "user": f"Problem:\n{problem['prompt']}\n\nFirst, create a plan. Then, carry it out."
        }
    }

    content = agent_messages.get(agentType, agent_messages[agentType])  

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": content["system"]},
            {"role": "user", "content": content["user"]}
        ],
        max_tokens=int(os.getenv("MAX_TOKENS_SOLVER", 2000))
    )

    return {
        "task_id": problem['task_id'],
        "agent_type": agentType,
        "response": response.choices[0].message.content,
        "tokens_used": response.usage.total_tokens
    }
    

def clean_code(text):
    pattern = r"```python\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        pattern_alt = r"```\n(.*?)\n```"
        match = re.search(pattern_alt, text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

if __name__ == "__main__":
    problems = load_eval_problems(r"C:\CSE188\plan_n_solve_eval\data\HumanEval.jsonl")
    
    # Run a pilot test on the first problem with zero-shot agent
    result = run_single_agent_eval(problems[0], agentType="plan-and-solve")
    
    print(f"--- RAW RESPONSE ---\n{result['response']}")
    print(f"agent type: {result['agent_type']}")
    print(f"\n--- CLEANED CODE ---\n{clean_code(result['response'])}")