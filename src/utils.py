import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

def numpy_to_tensor_img(img):
    img = (255.0 * np.transpose(img, (2,1,0))).astype(np.uint8)
    img = Image.fromarray(img)
    totensor = ToTensor()
    return totensor(img)

def to_tensor_dataset_format(indices, images, labels, class_to_idx):
    dataset_images = []
    dataset_labels = [] 
    for idx in indices:
        class_idx = torch.tensor(class_to_idx[labels[idx]])

        dataset_images.append(numpy_to_tensor_img(images[idx]))
        dataset_labels.append(class_idx)

    dataset_images = torch.stack(dataset_images, dim=0)
    dataset_labels = torch.stack(dataset_labels, dim=0)
    return (dataset_images, dataset_labels)

def forward_pass_with_analytics():
    pass

def idx_to_class(classes_name):
    idx_to_class = {}
    for idx, class_name in enumerate(classes_name):
        idx_to_class[idx] = class_name
    
    return idx_to_class

def class_to_idx(classes_name):
    return {class_:idx for idx, class_ in idx_to_class(classes_name).items()}

# mode is OCC for occurences of labels or PERC for proportion of labels
def dataset_occurence(classes_name, labels, mode="OCC") -> dict:
    assert mode == "PERC" or mode == "OCC"

    dataset_len = len(labels)
    occ = dict()
    
    for class_ in classes_name:
        class_idx = np.where(labels == class_)[0]        
        if mode == "PERC":
            occ[class_] = round((len(class_idx) / dataset_len) * 100, 1)
        elif mode == "OCC":
            occ[class_] = len(class_idx)

    return occ

# Saves an image grid of an instance of each class
def sample_images_dataset(classes_name, labels, images, path):
    tensor_imgs = []
    for class_ in classes_name:
        class_idx = np.where(labels == class_)[0]
        tensor_imgs.append(numpy_to_tensor_img(images[class_idx[0]]))
    
    save_image(torch.stack(tensor_imgs, dim=0), path, nrow=9)
    
    del tensor_imgs

