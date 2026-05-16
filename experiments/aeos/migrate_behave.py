import os
import shutil

source_dir = r"f:\AI-IN-THE-LOOP\aitl-paper\experiments\aeos"
target_dir = os.path.join(source_dir, "aeos_behave")

# Files to migrate for Paper 3
files_to_copy = [
    "data_loader.py",
    "trainer.py",
    "agent.py",
    "runner.py",
    "coder.py",
    "reviewer.py",
    "runner_critic.py",
    "questionbook.md",
    ".env"
]

def migrate():
    print(f"Creating directory: {target_dir}")
    os.makedirs(target_dir, exist_ok=True)
    
    # Create results folder inside
    os.makedirs(os.path.join(target_dir, "results"), exist_ok=True)
    
    for filename in files_to_copy:
        src = os.path.join(source_dir, filename)
        dst = os.path.join(target_dir, filename)
        
        if os.path.exists(src):
            print(f"Copying {filename}...")
            shutil.copy2(src, dst)
        else:
            print(f"WARNING: Source file {filename} not found!")
            
    print("\nMigration complete! The 'aeos_behave' folder is ready.")
    print("You can now CD into it and run experiments there to keep your root aeos folder clean.")

if __name__ == "__main__":
    migrate()
