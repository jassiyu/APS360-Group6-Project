import os
import shutil
import random

def extract_files_from_all_subdirs(src_folder, dst_dir):
    # Check if the specified source folder exists
    if not os.path.exists(src_folder):
        print(f"The specified source folder '{src_folder}' does not exist.")
        return
    
    # List all subdirectories in the source folder
    for foldername in os.listdir(src_folder):
        folder_path = os.path.join(src_folder, foldername)
        
        # Check if the item is a directory
        if os.path.isdir(folder_path):
            print(f"Processing subfolder: {folder_path}")

            # Traverse through all subdirectories of the current folder
            for root, dirs, files in os.walk(folder_path):
                if len(files) > 0:
                    # Get the relative path of the current subdirectory
                    #relative_path = os.path.relpath(root, src_folder)  # Get relative path from the src_folder
                    base_name = os.path.basename(folder_path)
                    new_folder_path = os.path.join(dst_dir, base_name)  # Create destination path based on the relative path

                    # Create the new folder if it doesn't exist
                    if not os.path.exists(new_folder_path):
                        os.makedirs(new_folder_path)

                    # Randomly select up to 5 files from the current directory
                    selected_files_1 = random.sample(files, min(30, len(files)))

                    # remaining_files = [val_file for val_file in files if val_file not in selected_files_1]
                    # selected_files_val = random.sample(remaining_files, min(30, len(remaining_files)))

                    # Copy each selected file to the new directory
                    for selected_file in selected_files_1:
                        file_path = os.path.join(root, selected_file)
                        shutil.move(file_path, new_folder_path)
                        #print(f'Copied {selected_file} from {root} to {new_folder_path}')

                    # for selected_file in selected_files_val:
                    #     file_path = os.path.join(root, selected_file)
                    #     shutil.copy(file_path, new_folder_path)




# Example usage (use double backslashes `\\` for Windows paths):
# src_directory = r"C:\Users\Joe\Desktop\UOFT\2024Fall\APS 360\project\archive\Train"  # Use raw strings with `r` for Windows paths
# dst_directory = r"C:\Users\Joe\Desktop\UOFT\2024Fall\APS 360\project\archive\truncated_train"     # Destination folder for extracted files                          # Number of files to randomly extract from each folder

##############val##################
##############sample = 30############
# src_directory = r"C:\Users\Joe\Desktop\UOFT\2024Fall\APS 360\project\archive\Train"  # Use raw strings with `r` for Windows paths
# dst_directory = r"C:\Users\Joe\Desktop\UOFT\2024Fall\APS 360\project\archive\truncated_val"     # Destination folder for extracted files

##############test#################
##############sample = 30############
# src_directory = r"C:\Users\Joe\Desktop\UOFT\2024Fall\APS 360\project\archive\Test"  # Use raw strings with `r` for Windows paths
# dst_directory = r"C:\Users\Joe\Desktop\UOFT\2024Fall\APS 360\project\archive\truncated_test"     # Destination folder for extracted files

extract_files_from_all_subdirs(src_directory, dst_directory)
