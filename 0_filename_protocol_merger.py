import os
import pyabf

# Directory where .abf files are located
root_dir = '/Volumes/BWH-HVDATA/Individual Folders/Garrett/Patch Clamp/Data'

# Loop through all .abf files in the directory
for filename in os.listdir(root_dir):
    if filename.endswith('.abf'):
        # Check if there are 3 or fewer underscores in the original filename
        if filename.count('_') <= 3:
            file_path = os.path.join(root_dir, filename)
            abf = pyabf.ABF(file_path)
            
            # Get the protocol name from the ABF file
            protocol_name = abf.protocol
            
            # Replace spaces with underscores in the filename and protocol name
            filename = filename.replace(' ', '_')
            protocol_name = protocol_name.replace(' ', '_')
            
            # Remove the .abf extension from the original filename
            base_filename = filename[:-4]  # Remove '.abf'
            
            # Construct the new filename (adding protocol name to the end and .abf at the end)
            new_filename = f"{base_filename}_{protocol_name}.abf"
            new_file_path = os.path.join(root_dir, new_filename)
            # Rename the file
            os.rename(file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")
