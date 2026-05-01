import subprocess
import os
import sys
import tempfile

def run_in_sandbox(generated_code, test_script, timeout=0.5):
    """
    Runs generated code against test assertions in a controlled environment.
    
    Args:
        generated_code (str): The Python code extracted from the LLM.
        test_script (str): The 'test' field from the HumanEval JSON.
        timeout (float): Seconds before killing the process.
        
    Returns:
        dict: Results containing success status and error logs.
    """
    # Create the temporary directory if it doesn't exist
    tmp_dir = os.path.join(os.getcwd(), "sandbox", "tmp_eval")
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Create a unique temp file to avoid Windows file-locking collisions
    fd, temp_path = tempfile.mkstemp(suffix=".py", dir=tmp_dir)
    
    # Combine code and tests
    # Note: HumanEval tests usually look like 'check(function_name)'
    full_content = f"{generated_code}\n\n{test_script}"
    
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(full_content)

        # Run the code using the current Python interpreter
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode == 0:
            return {"status": "passed", "error": None}
        else:
            return {"status": "failed", "error": result.stderr.strip()}

    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"Exceeded {timeout}s"}
    except Exception as e:
        return {"status": "error", "error": str(e)}
    finally:
        # Cleanup: try to remove the file, but don't crash if Windows lock prevents it
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass