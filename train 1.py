import tarfile
import random
import os
import shutil

# Step 1: Extract all files from the tar archive to a temporary directory
temp_dir = 'daily_mail_stories_temp'
with tarfile.open('dailymail_stories.tgz', 'r:gz') as file:
    file.extractall(temp_dir)

# Step 2: Navigate to the nested directories and get a list of all `.story` files
nested_dir = os.path.join(temp_dir, 'dailymail/stories')  # Adjust based on the actual directory structure
all_files = [os.path.join(nested_dir, f) for f in os.listdir(nested_dir) if os.path.isfile(os.path.join(nested_dir, f)) and f.endswith('.story')]
random.shuffle(all_files)
subset_files = all_files[:int(0.3 * len(all_files))]

# Step 3: Move the selected files to the desired directory
dest_dir = 'daily_mail_stories'
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

for file in subset_files:
    shutil.move(file, os.path.join(dest_dir, os.path.basename(file)))

# Step 4: Delete the temporary directory
