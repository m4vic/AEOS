import os
import glob
import shutil

def migrate():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    tabular_dir = os.path.join(results_dir, "tabular2")
    
    os.makedirs(tabular_dir, exist_ok=True)
    
    # Find all json and png files that belong to tabular2
    json_files = glob.glob(os.path.join(results_dir, "*tabular2*.json"))
    png_files = glob.glob(os.path.join(results_dir, "*tabular2*.png"))
    
    # Also find files that have tabular2 in the name but maybe different format
    all_files = set(json_files + png_files)
    
    # Also move plots that don't have tabular2 in name but were from tabular2 runs
    # (Checking if the filename contains 'run' and '.png' and not 'vision')
    other_pngs = glob.glob(os.path.join(results_dir, "*.png"))
    for p in other_pngs:
        if 'vision' not in p and 'text' not in p:
            all_files.add(p)
            
    moved_count = 0
    for f in all_files:
        if os.path.isfile(f):
            filename = os.path.basename(f)
            dest = os.path.join(tabular_dir, filename)
            shutil.move(f, dest)
            print(f"Moved {filename} -> results/tabular2/")
            moved_count += 1
            
    print(f"Migration complete. Moved {moved_count} files.")

if __name__ == "__main__":
    migrate()
