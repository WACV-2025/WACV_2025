import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the model
model_VT = torchvision.models.vit_l_32(weights='DEFAULT')
feature_extractor = nn.Sequential(*list(model_VT.children())[:-1])
encoder = feature_extractor[1]

def extractor(img_path):
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )
    try:
        img = Image.open(img_path)
        if img.getbands() == ('L',):
            return

        img = transform(img)
        x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)

        n = x.shape[0]
        x = model_VT._process_input(x)
        batch_class_token = model_VT.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        y = encoder(x)

        y = y[:, 0]
        y = torch.squeeze(y)
        y = torch.flatten(y)
        y = y.data.numpy()

        return y

    except Exception as e:
        print(f"Error extracting features from {img_path}: {e}")
        return

def find_images_in_directory(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Load the data directory where the images are stored
data_dir = 'DeepfakeDetection/Data'
contents = os.listdir(data_dir)
classes = [each for each in contents if os.path.isdir(os.path.join(data_dir, each))]
enum_classes = list(enumerate(classes, 0))

images = []
batch = []
labels = []

# Initialize dataset as an empty list
dataset = []

def processFolders(each):
    global dataset
    print(f"Starting {each[1]} images")
    class_path = os.path.join(data_dir, each[1])
    image_paths = find_images_in_directory(class_path)

    for img_path in image_paths:
        data = []

        file_name_type = 'fake_' + each[1] + '_' + os.path.basename(img_path)
        data.append(file_name_type)
        data.append(str(1))  # For fake images (class 1)
        
        features = extractor(img_path)  # Extract features
        if features is not None:
            temp = np.array([features, data], dtype="O")
            dataset.append(temp)

    print(f"Finished processing {each[1]}")

if __name__ == '__main__':
    import time
    start = time.time()

    with ThreadPoolExecutor(max_workers=1) as executor:
        for _ in executor.map(processFolders, enum_classes):
            pass

    end = time.time()
    print(f"Processing time: {end - start} seconds")

    if dataset:
        dataset = np.array(dataset, dtype="O")
        np_batch = np.stack(dataset[:, 0])
        np_info = np.stack(dataset[:, 1])
        np_labels = np_info[:, 1]
        np_images = np_info[:, 0]

        np_labels_T = np_labels.reshape(-1, 1)
        np_images_T = np_images.reshape(-1, 1)
        np_images_labels = np.hstack((np_images_T, np_labels_T))
        print(np_images_labels)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            np_batch, np_images_labels, test_size=0.1, random_state=0)

        # Convert data to Pandas in order to save as .csv
        data_df_X_train = pd.DataFrame(X_train)
        data_df_y_train = pd.DataFrame(y_train)
        data_df_X_test = pd.DataFrame(X_test)
        data_df_y_test = pd.DataFrame(y_test)

        print(data_df_X_train)

        # Save files as .csv
        data_df_X_train.to_csv('data_df_X_train_deepfake_ffhq_finetuned.csv', mode='a', header=False, index=False)
        data_df_y_train.to_csv('data_df_y_train_deepfake_ffhq_finetuned.csv', mode='a', header=False, index=False)
        data_df_X_test.to_csv('data_df_X_test_deepfake_ffhq_finetuned.csv', mode='a', header=False, index=False)
        data_df_y_test.to_csv('data_df_y_test_deepfake_ffhq_finetuned.csv', mode='a', header=False, index=False)
    else:
        print("No data was processed.")
