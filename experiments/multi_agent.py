import os
from dotenv import load_dotenv
from openai import OpenAI
from load_eval import load_eval_problems

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_multi_agent_eval(problem):
    prompt_content = problem['prompt']

    planner_message = [
        {"role": "system", "content": "You are a technical architect. Provide a step-by-step logic plan and identify edge cases for the given function prompt. You are not allowed to use any import statements. Do not write code."},
        {"role": "user", "content": f"Problem Prompt:\n{prompt_content}"}
    ]

    plan_response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=planner_message,
        max_tokens=int(os.getenv("MAX_TOKENS_PLANNER", 1000))
    )
    tokens_used = plan_response.usage.total_tokens

    plan = plan_response.choices[0].message.content

    solver_message = [
        {"role": "system", "content": "You are a senior programmer. Implement the Python function based ONLY on the provided plan. You are not allowed to use any import statements. Solve the problem using only core Python primitives. Output ONLY the code within triple backticks."},
        {"role": "user", "content": f"Implementation Plan:\n{plan}"}
    ]

    code_resonse = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=solver_message,
        max_tokens=int(os.getenv("MAX_TOKENS_SOLVER", 1000))
    )
    generated_code = code_resonse.choices[0].message.content
    tokens_used += code_resonse.usage.total_tokens

    return {
        "task_id": problem['task_id'],
        "plan": plan,
        "code": generated_code,
        "tokens_used": tokens_used
    }

if __name__ == "__main__":
    problems = load_eval_problems(r"C:\CSE188\plan_n_solve_eval\data\HumanEval.jsonl")
    
    # Run a pilot test on the first problem
    result = run_multi_agent_eval(problems[0])
    
    print(f"--- PLAN ---\n{result['plan']}")
    print(f"\n--- GENERATED CODE ---\n{result['code']}")