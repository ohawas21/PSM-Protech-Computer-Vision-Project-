import subprocess
from datetime import datetime
import os

def generate_commit_log():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "gitlog_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{output_dir}/commit_summary_{timestamp}.log"
    
    try:
        result = subprocess.run(
            ["git", "shortlog", "-s", "-n"],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout
        
        # Only write the raw shortlog output
        with open(output_filename, "w") as f:
            f.write(output.strip())
        
    except subprocess.CalledProcessError as e:
        # If git command fails, still create a file but mention error
        with open(f"{output_dir}/commit_summary_error.log", "w") as f:
            f.write("Error capturing commit log.\n")
            f.write(str(e))

if __name__ == "__main__":
    generate_commit_log()
