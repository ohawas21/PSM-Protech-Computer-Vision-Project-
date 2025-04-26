import subprocess
from datetime import datetime
import os

def generate_commit_log():
    # Generate timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"commit_summary_{timestamp}.log"
    
    try:
        # Run git shortlog to get commits per user
        result = subprocess.run(
            ["git", "shortlog", "-s", "-n"],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout
        
        # Save the result to a log file
        with open(output_filename, "w") as f:
            f.write(output)
        
        print(f"[INFO] Commit summary saved to {output_filename}")
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to run git shortlog: {e}")
        # Optional: write an empty or error log if git fails
        with open("commit_summary_error.log", "w") as f:
            f.write("Error capturing commit log.")

if __name__ == "__main__":
    generate_commit_log()
