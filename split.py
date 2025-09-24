import os
import random
import shutil

train_folder = 'facial_expressions\\train'

# Step 1: create a list of all image filenames in the train folder
image_filenames = []
for expression in os.listdir(train_folder):
    expression_folder = os.path.join(train_folder, expression)
    if os.path.isdir(expression_folder):
        for filename in os.listdir(expression_folder):
            if filename.endswith('.jpg'):
                image_filenames.append(os.path.join(expression_folder, filename))

# Step 2: shuffle the list of image filenames
random.seed(42)
random.shuffle(image_filenames)
print("Image_filenames", len(image_filenames))

# Step 3: split the shuffled list into train and validation splits
split_index = int(0.8 * len(image_filenames))
train_filenames = image_filenames[:split_index]
print("train_filenames", len(train_filenames))
val_filenames = image_filenames[split_index:]
print("val_filenames", len(val_filenames))

# Step 4: create dictionary for train split
train_dict = {}
for expression in os.listdir(train_folder):
    expression_folder = os.path.join(train_folder, expression)
    if os.path.isdir(expression_folder):
        train_dict[expression] = [f for f in train_filenames if f.startswith(expression_folder)]

# Step 5: create dictionary for validation split
val_dict = {}
for expression in os.listdir(train_folder):
    expression_folder = os.path.join(train_folder, expression)
    if os.path.isdir(expression_folder):
        val_dict[expression] = [f for f in val_filenames if f.startswith(expression_folder)]

# for key in val_dict.keys():
#     print(key)
#     count = 0
#     for value in val_dict[key]:
#         count += 1
#     print(count)

# print('-----------------')
# for key in train_dict.keys():
#     print(key)
#     count = 0
#     for value in train_dict[key]:
#         count += 1
#     print(count)


val_dir = 'facial_expressions/valid'
if not os.path.exists(val_dir):
    os.mkdir(val_dir)

# Loop over the dictionary items and create a subdirectory for each expression
for expression in val_dict.keys():
    expression_dir = os.path.join(val_dir, expression)
    if not os.path.exists(expression_dir):
        os.mkdir(expression_dir)

# Move the files to the appropriate subdirectory
for expression, filenames in val_dict.items():
    for filename in filenames:
        expression_dir = os.path.join(val_dir, expression)
        shutil.move(filename, expression_dir)