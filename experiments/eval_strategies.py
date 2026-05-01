import pandas as pd
from load_eval import load_eval_problems
from multi_agent import run_multi_agent_eval
from single_agent import run_single_agent_eval, clean_code
from sandbox.sandbox_executor import run_in_sandbox
import time

def main():
    problems = load_eval_problems(r"C:\CSE188\plan_n_solve_eval\data\HumanEval.jsonl")
    
    results = []
    
    #for problem in problems:
    for problem in problems[:50]:  
        task_id = problem['task_id']
        print (f"Evaluating Task ID: {task_id}")
        # Run multi-agent evaluation
        multi_agent_result = run_multi_agent_eval(problem)
        multi_agent_code = clean_code(multi_agent_result['code'])

        # Run single-agent evaluation (cot)
        cot_agent_result = run_single_agent_eval(problem, agentType="cot")
        cot_agent_code = clean_code(cot_agent_result['response'])
        
        # Run single-agent evaluation (plan-and-solve)
        single_agent_result = run_single_agent_eval(problem, agentType="plan-and-solve")
        single_agent_code = clean_code(single_agent_result['response'])
        
        # Evaluate in sandbox
        multi_agent_eval = run_in_sandbox(multi_agent_code, problem['test'])
        cot_agent_eval = run_in_sandbox(cot_agent_code, problem['test'])
        single_agent_eval = run_in_sandbox(single_agent_code, problem['test'])

        print (f"Multi-Agent Eval: {multi_agent_eval['status']}")
        print (f"COT Agent Eval: {cot_agent_eval['status']}")
        print (f"Plan-and-Solve Eval: {single_agent_eval['status']}")
        
        results.append({
            "task_id": task_id,
            "multi_agent_plan": multi_agent_result['plan'],
            "multi_agent_code": multi_agent_code,
            "multi_agent_eval": multi_agent_eval,
            "multi_agent_tokens": multi_agent_result['tokens_used'],
            "cot_agent_code": cot_agent_code,
            "cot_agent_eval": cot_agent_eval,
            "cot_agent_tokens": cot_agent_result['tokens_used'],
            "plan_and_solve_code": single_agent_code,
            "plan_and_solve_eval": single_agent_eval,
            "plan_and_solve_tokens": single_agent_result['tokens_used']
        })
        
        # Sleep to respect rate limits
        time.sleep(1)
    
    df = pd.DataFrame(results)
    df.to_csv("results/evaluation_results.csv", index=False)
    print("Evaluation completed. Results saved to results/evaluation_results.csv")

if __name__ == "__main__":
    main()
    