import os
import shutil
import random

# Set your category folders here
CATEGORIES = [
    ('20021010_easy_ham/easy_ham', 'easy_ham'),
    ('20030228_hard_ham/hard_ham', 'hard_ham'),
    ('20030228_spam/spam', 'spam')
]

# Fraction for training set
TRAIN_FRAC = 0.8

for folder, label in CATEGORIES:
    abs_folder = os.path.abspath(folder)
    files = [f for f in os.listdir(abs_folder) if os.path.isfile(os.path.join(abs_folder, f))]
    random.shuffle(files)
    n_train = int(len(files) * TRAIN_FRAC)
    train_files = files[:n_train]
    test_files = files[n_train:]

    # Create train/test subfolders
    train_dir = os.path.join(abs_folder, 'train')
    test_dir = os.path.join(abs_folder, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Copy files
    for fname in train_files:
        shutil.copy2(os.path.join(abs_folder, fname), os.path.join(train_dir, fname))
    for fname in test_files:
        shutil.copy2(os.path.join(abs_folder, fname), os.path.join(test_dir, fname))

    print(f"{label}: {len(train_files)} train, {len(test_files)} test files copied.")

print("Splitting complete.")
