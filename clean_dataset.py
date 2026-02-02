#!/usr/bin/env python3
"""
Dataset Cleaner - Find and remove corrupted/empty images
"""

import os
from PIL import Image
import shutil

PATH = "/Users/maisievarcoe/Desktop/AI/Coursework/images/Originals"

print("="*60)
print("DATASET CLEANER")
print("="*60)
print(f"Scanning: {PATH}\n")

# Track issues
corrupted_files = []
empty_files = []
hidden_files = []
non_image_files = []
valid_files = 0

# Get all class folders
class_folders = [d for d in os.listdir(PATH) 
                 if os.path.isdir(os.path.join(PATH, d)) and not d.startswith('.')]

print(f"Found {len(class_folders)} class folders: {class_folders}\n")

for class_name in class_folders:
    class_path = os.path.join(PATH, class_name)
    print(f"Checking '{class_name}'...")
    
    files = os.listdir(class_path)
    
    for filename in files:
        filepath = os.path.join(class_path, filename)
        
        # Skip directories
        if os.path.isdir(filepath):
            continue
        
        # Check for hidden files (start with .)
        if filename.startswith('.'):
            hidden_files.append(filepath)
            print(f"  ⚠ Hidden file: {filename}")
            continue
        
        # Check if it's an image extension
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            non_image_files.append(filepath)
            print(f"  ⚠ Non-image file: {filename}")
            continue
        
        # Check if file is empty
        if os.path.getsize(filepath) == 0:
            empty_files.append(filepath)
            print(f"  ⚠ Empty file: {filename}")
            continue
        
        # Try to open the image
        try:
            with Image.open(filepath) as img:
                img.verify()  # Verify it's a valid image
            
            # Re-open to check it can be loaded
            with Image.open(filepath) as img:
                img.load()
            
            valid_files += 1
            
        except Exception as e:
            corrupted_files.append(filepath)
            print(f"  ✗ Corrupted: {filename} ({str(e)[:50]})")
    
    print(f"  ✓ Completed\n")

# Summary
print("="*60)
print("SUMMARY")
print("="*60)
print(f"Valid images:      {valid_files}")
print(f"Hidden files:      {len(hidden_files)}")
print(f"Non-image files:   {len(non_image_files)}")
print(f"Empty files:       {len(empty_files)}")
print(f"Corrupted files:   {len(corrupted_files)}")
print("="*60 + "\n")

# Create backup folder if needed
if hidden_files or non_image_files or empty_files or corrupted_files:
    backup_path = os.path.join(os.path.dirname(PATH), "Problematic_Files")
    os.makedirs(backup_path, exist_ok=True)
    print(f"Backup folder created: {backup_path}\n")
    
    all_problem_files = hidden_files + non_image_files + empty_files + corrupted_files
    
    print("Moving problematic files to backup folder...")
    
    for filepath in all_problem_files:
        try:
            # Get relative path to maintain folder structure
            rel_path = os.path.relpath(filepath, PATH)
            backup_filepath = os.path.join(backup_path, rel_path)
            
            # Create subdirectory if needed
            os.makedirs(os.path.dirname(backup_filepath), exist_ok=True)
            
            # Move file
            shutil.move(filepath, backup_filepath)
            print(f"  Moved: {os.path.basename(filepath)}")
        except Exception as e:
            print(f"  ✗ Error moving {filepath}: {e}")
    
    print("\n✓ All problematic files moved to backup folder")
    print(f"\nYour dataset is now clean! You have {valid_files} valid images.")
    print("You can safely delete the 'Problematic_Files' folder if you don't need those files.\n")
else:
    print("✓ No problematic files found! Your dataset is clean.\n")

print("="*60)
print("FINAL DATASET")
print("="*60)

# Count final images per class
for class_name in class_folders:
    class_path = os.path.join(PATH, class_name)
    image_files = [f for f in os.listdir(class_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
                   and not f.startswith('.')]
    print(f"{class_name}: {len(image_files)} images")

print("="*60)
print("\n✓ Dataset cleaning complete!")
print("You can now run your training script.\n")
