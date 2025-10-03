import os
import pyabf
import shutil

root_dir = '/Users/gs075/Documents/HVDriveBackup/Backup/PatchClamp/Data'
dest_dir = '/Users/gs075/Documents/HVDriveBackup/Data_Renamed'
os.makedirs(dest_dir, exist_ok=True)

for filename in os.listdir(root_dir):
    if filename.endswith('.abf'):
        if filename.count('_') <= 3:
            file_path = os.path.join(root_dir, filename)

            # Optionally skip empty files early
            if os.path.getsize(file_path) == 0:
                print(f"Skipped empty file: {filename}")
                continue

            try:
                abf = pyabf.ABF(file_path)
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
                continue

            protocol_name = abf.protocol.replace(' ', '_')
            clean_filename = filename.replace(' ', '_')
            base_filename = clean_filename[:-4]  # remove .abf

            new_filename = f"{base_filename}_{protocol_name}.abf"
            dest_file_path = os.path.join(dest_dir, new_filename)

            if not os.path.exists(dest_file_path):
                shutil.copy2(file_path, dest_file_path)
                print(f"Copied to Backup_Renamed: {filename} -> {new_filename}")
            else:
                print(f"Skipped (already exists): {new_filename}")
