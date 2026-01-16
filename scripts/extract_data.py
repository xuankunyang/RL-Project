import tarfile
import os
import sys

def extract_smart(tar_path, target_keyword="results"):
    print(f"Opening {tar_path}...")
    if not os.path.exists(tar_path):
        print("Error: File not found.")
        return

    with tarfile.open(tar_path, 'r:gz') as tar:
        members = tar.getmembers()
        print(f"Found {len(members)} files in archive.")
        
        extracted_count = 0
        for member in members:
            # member.name is like 'home/user/RL/results/Atari/...'
            # We want to strip everything before 'results/'
            
            parts = member.name.split('/')
            try:
                # Find index of 'results'
                idx = parts.index(target_keyword)
                # New path starts from 'results'
                new_path = os.path.join(*parts[idx:])
                
                # Update member name to extract to relative path
                member.name = new_path
                
                # Extract
                tar.extract(member, path=".")
                extracted_count += 1
            except ValueError:
                print(f"Skipping {member.name} (keyword '{target_keyword}' not found)")
                
        print(f"Successfully extracted {extracted_count} files to ./results/")

if __name__ == "__main__":
    extract_smart("rl_project_all_data.tar.gz")
