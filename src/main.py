import pandas as pd
import numpy as np
import os 

from torchvision.transforms import Compose, ToTensor, ToPILImage, RandomChoice,\
            RandomHorizontalFlip, RandomVerticalFlip, Normalize

import torch
from torch import nn, optim
from torch.nn import functional as F

from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from model import WaferNet
from dataset import CustomTensorDataset
from utils import *
from constants import *

if __name__ == "__main__":

    print("Running...")
 
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cpu")

    writer = SummaryWriter(log_dir=EXP_NAME)

    # Loading dataset
    df  = pd.read_pickle("waferImg26x26.pkl")
    images = df.images.values
    labels   = df.labels.values
    labels = np.asarray([str(l[0]) for l in labels])

    classes_name = np.unique(labels)
    print("Class names: {}".format(classes_name))

    print("Dataset size: {}".format(len(labels)))

    class_to_idx = class_to_idx(classes_name)
    idx_to_class = idx_to_class(classes_name)

    # Remove portion of none class, keep 250 samples. Due to none class being 94% of the dataset.
    # TODO: better to keep all data and focus on transforming classes with not enough data
    idx_to_remove = np.random.choice(np.where(labels == "none")[0], 13489-250, replace=False)
    images = np.delete(images, idx_to_remove)
    labels = np.delete(labels, idx_to_remove)
    
    # Class balance and sample of each
    print("Total Dataset Class balance: {}".format(dataset_occurence(classes_name, labels, "PERC"))) # add param in utils to choose for count or proportion
    
    # Save image sample of each class
    sample_images_dataset(classes_name, labels, images, "/Users/georgesnomicos/Documents/Projects/wafer_classifier/sample_classes.jpg")

    # Dataset split
    # TODO: utils method to create datasets or use sklearn k-fold module
    indices = list(range(len(images)))
    split_valid = int(np.floor(PERC_VALID * len(images)))
    split_test = int(np.floor(PERC_TEST * len(images)))
    np.random.shuffle(indices)
    train_indices, valid_indices = indices[split_valid+split_test:], indices[:split_valid]

    print(len(train_indices), len(valid_indices), split_valid, split_valid+split_test)

    # Make sure class 2 (donut) is in training set since very low occurence
    idx_donut_class = np.where(labels == "Donut")
    idx_donut_in_train = np.where(np.array(train_indices) == idx_donut_class[0])
    if not idx_donut_in_train:
        raise BaseException("Label 2 not in the train dataset")

    # Convert numpy to torch tensor (batched)
    # TODO: combine with dataset class
    tensor_dataset_training = to_tensor_dataset_format(train_indices, images, labels, class_to_idx)
    tensor_dataset_valid = to_tensor_dataset_format(valid_indices, images, labels, class_to_idx)

    # Proportion of each class in train
    train_class_occ = dataset_occurence(list(class_to_idx.values()), tensor_dataset_training[1], mode="PERC")
    print("Training Class balance: {}".format(train_class_occ))

    # Train weighted dataset class sampling and dataloaders
    weights = 1. / torch.FloatTensor(list(train_class_occ.values()))
    
    # mean and std from training dataset for normalization
    mean = torch.mean(tensor_dataset_training[0], dim=(0,2,3))
    std = torch.std(tensor_dataset_training[0], dim=(0,2,3)) 
    print("Mean {} and Standard deviation {}".format(mean, std))

    # Transforms
    T = Compose([Normalize(mean, std)]) 
    T_augm = Compose([ToPILImage(),
            RandomChoice([RandomHorizontalFlip(p=1), 
            RandomVerticalFlip(p=1), lambda a: a]),
            ToTensor(),
            Normalize(mean, std)])

    training_data = CustomTensorDataset(tensor_dataset_training, transform=T, transform_augmentation=T_augm)
    train_sampler = WeightedRandomSampler(weights[tensor_dataset_training[1]], len(train_indices))

    valid_data = CustomTensorDataset(tensor_dataset_valid, transform=T)

    train_loader = DataLoader(training_data, batch_size=BATCH_SIZE, sampler=train_sampler)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)

    # Network declaration
    net = WaferNet().to(device)

    # Training
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_R)
    criterion = nn.NLLLoss()

    train_size = len(train_indices)
    valid_size = len(valid_indices)

    for epoch in range(EPOCHS):
        print("Epoch {}/{}".format(epoch+1, EPOCHS))
        print("-" * 10)

        running_loss_train = 0
        running_correct_train = 0
        running_loss_valid = 0
        running_correct_valid = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            out = net(inputs)
            loss = criterion(out, labels)
            _, pred = torch.max(torch.exp(out),1)
            
            loss.backward()
            optimizer.step()

            running_loss_train += loss.item() * inputs.size(0)
            running_correct_train += torch.sum(pred==labels)

        # Validation
        if epoch != 0 or epoch % 1 == 0:
            net.eval()
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                out = net(inputs)
                loss = criterion(out, labels)
                _, pred = torch.max(torch.exp(out),1)
                running_loss_valid += loss.item() * inputs.size(0)
                running_correct_valid += torch.sum(pred==labels)

            epoch_loss_valid = running_loss_valid/valid_size
            epoch_accuracy_valid = running_correct_valid.type(torch.DoubleTensor)/valid_size
            
            print("Validation, Loss:{} Accuracy: {}".format(epoch_loss_valid,epoch_accuracy_valid))
            
            writer.add_scalar('Loss/Valid',epoch_loss_valid, epoch)
            writer.add_scalar('Accuracy/Valid', epoch_accuracy_valid, epoch)
        
        net.train()

        epoch_loss_train = running_loss_train/train_size
        epoch_accuracy_train = running_correct_train.type(torch.DoubleTensor)/train_size
        
        writer.add_scalar('Loss/Train',epoch_loss_train, epoch)
        writer.add_scalar('Accuracy/Train', epoch_accuracy_train, epoch)


        print("Training, Loss:{} Accuracy: {}".format(epoch_loss_train,epoch_accuracy_train))
    
    # Saving parameters network
    torch.save(net.state_dict(), os.path.join(EXP_NAME,"model.pt"))

    # Saving hyperparams to tensorboard
    hyperparam = {"BATCH_SIZE": BATCH_SIZE, "LEARNING_RATE": LEARNING_R, "EPOCHS": EPOCHS}
    writer.add_text("architecture/", str(net))
    writer.add_hparams(hyperparam,{'hparam/accuracy': epoch_accuracy_train, 'config/TestDataIndexStart': split_valid, \
                                'config/TestDataIndexEnd': split_valid+split_test})
    writer.close()