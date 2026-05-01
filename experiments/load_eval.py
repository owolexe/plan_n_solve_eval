import json

def load_eval_problems(file_path):
    problems = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            problem = json.loads(line.strip())
            problems.append(problem)
    return problems

if __name__ == "__main__":
    data = load_eval_problems(r"C:\CSE188\plan_n_solve_eval\data\HumanEval.jsonl")
    print(f"Loaded {len(data)} problems.")
    print("First problem:", data[0])