import pandas as pd
import numpy as np
from PIL import Image

from torchvision.utils import save_image
from torchvision.transforms import Compose, ToTensor, ToPILImage, RandomChoice,\
            RandomHorizontalFlip, RandomVerticalFlip, Normalize

import torch
from torch import nn, optim
from torch.nn import functional as F

from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

# from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os 

class WaferNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.pad1 = nn.ConstantPad2d((0,1,0,1),0)
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=(1,1))
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=(1,1))
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=(1,1))
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=(1,1))
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(3*3*256, 512)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 9)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # save_image(x, "test.jpg")
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        # print(x.size())
        x = self.dropout1(F.relu(self.fc1(x.view(-1,3*3*256))))
        x = self.fc2(x)
        
        return self.softmax(x)

# https://stackoverflow.com/questions/55588201/pytorch-transforms-on-tensordataset
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None, transform_augmentation=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        
        self.transform = transform
        self.transform_augm = transform_augmentation

    def __getitem__(self, index):
        x = self.tensors[0][index]

        y = self.tensors[1][index]

        if self.transform_augm and y.item() in [0,1,4,6,7,8]:
            x = self.transform_augm(x)
        elif self.transform: 
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


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

if __name__ == "__main__":

    print("running")
 
    BATCH_SIZE = 64
    LEARNING_R = 0.001
    EPOCHS = 30

    np.random.seed(40)
    torch.manual_seed(40)

    device = torch.device("cpu")

    EXP_NAME = "EXP/Final_model_benchmark_60_ep"

    writer = SummaryWriter(log_dir=EXP_NAME)

    df  = pd.read_pickle("waferImg26x26.pkl")
    images = df.images.values
    labels   = df.labels.values
    labels = np.asarray([str(l[0]) for l in labels])

    classes_name = np.unique(labels)
    print("Class names: {}".format(classes_name))
    class_to_idx = class_to_idx(classes_name)
    idx_to_class = idx_to_class(classes_name)

    # Remove portion of none class, keep 250 samples. Due to none class being 94% of the dataset.
    idx_to_remove = np.random.choice(np.where(labels == "none")[0], 13489-250, replace=False)
    images = np.delete(images, idx_to_remove)
    labels = np.delete(labels, idx_to_remove)
    
    # Class balance and sample of each
    classes_balance = []
    sample_classes = []
    for class_ in classes_name:
        class_idx = np.where(labels == class_)[0]
        classes_balance.append(len(class_idx))

        sample_classes.append(class_idx[0]) 

    print("Total Dataset Class balance: {}".format(np.round((np.array(classes_balance)/len(labels))*100, 2).tolist()))
    
    # Save image sample of each class
    tensor_imgs = []
    for img_idx in sample_classes:
        tensor_imgs.append(numpy_to_tensor_img(images[img_idx]))
    
    save_image(torch.stack(tensor_imgs, dim=0), "/Users/georgesnomicos/Documents/Projects/wafer_classifier/sample_classes.jpg", nrow=9)
    del tensor_imgs

    # Dataset split
    indices = list(range(len(images)))
    split_valid = int(np.floor(0.15 * len(images)))
    split_test = int(np.floor(0.10 * len(images)))
    np.random.shuffle(indices)
    train_indices, valid_indices, test_indices = indices[split_valid+split_test:], indices[:split_valid], indices[split_valid:split_valid+split_test]

    print(len(train_indices), len(valid_indices), len(test_indices))

    # Make sure class 2 (donut) is in training set
    idx_donut_class = np.where(labels == "Donut")
    idx_donut_in_train = np.where(np.array(train_indices) == idx_donut_class[0])
    if not idx_donut_in_train:
        raise BaseException("Label 2 not in the train dataset")


    # Convert numpy to torch tensor (batched)
    tensor_dataset_training = to_tensor_dataset_format(train_indices, images, labels, class_to_idx)
    tensor_dataset_valid = to_tensor_dataset_format(valid_indices, images, labels, class_to_idx)
    tensor_dataset_test = to_tensor_dataset_format(test_indices, images, labels, class_to_idx)

    # Proportion of each class in train
    train_class_balance = [0 for _ in range(9)]
    for idx in train_indices:
        class_idx = class_to_idx[labels[idx]]
        train_class_balance[class_idx] += 1

    print("Training Class balance: {}".format(train_class_balance))

    # Train weighted dataset class sampling and dataloaders
    weights = 1. / torch.FloatTensor(train_class_balance)
    
    # mean and std from training dataset for normalization
    mean = torch.mean(tensor_dataset_training[0], dim=(0,2,3))
    std = torch.std(tensor_dataset_training[0], dim=(0,2,3)) 

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
    test_data = CustomTensorDataset(tensor_dataset_test, transform=T)


    train_loader = DataLoader(training_data, batch_size=BATCH_SIZE, sampler=train_sampler)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # Network declaration:
    net = WaferNet().to(device)

    # Training:
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
    
    y_true = []
    y_pred = []
    running_correct_test = 0
    idx_p = 0
    fail_examples = {}

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        out = net(inputs)

        probas, pred = torch.max(torch.exp(out),1)
        
        y_true.extend(labels.tolist())
        y_pred.extend(pred.tolist())

        running_correct_test += torch.sum(pred==labels)

        accuracy_test = running_correct_test.type(torch.DoubleTensor)/len(test_indices)
        

        for i in range(inputs.size()[0]):
            i_img = len(y_pred)-inputs.size()[0]+i
            if y_pred[i_img] != y_true[i_img]:

                fail_examples[idx_p] = {'img_file_name':"fail_{}.jpeg".format(idx_p),
                        'class':idx_to_class[y_true[i_img]],
                        'predicted':idx_to_class[y_pred[i_img]],
                        'confidence':round(probas[i].item(),3)}
                idx_p += 1
                writer.add_image("Fail_Images/fail{}".format(idx_p),inputs[i,:,:,:].view(3,26,26))
                writer.add_text("Fail_meta/fail{}".format(idx_p), str(fail_examples))

    # cf = confusion_matrix(y_true, y_pred, labels=list(range(9)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # cax = ax.matshow(cf)
    # fig.colorbar(cax)

    # ax.set_xticks(np.arange(len(classes_name)))
    # ax.set_yticks(np.arange(len(classes_name)))
    # ax.set_xticklabels(classes_name,rotation=45)
    # ax.set_yticklabels(classes_name,rotation=0)
    # ax.set_xlabel('True Label')
    # ax.set_ylabel('Predicated Label')
    # for i in range(len(classes_name)):
    #     for j in range(len(classes_name)):
    #         text = ax.text(j, i, cf[i, j],
    #                     ha="center", va="center", color="w")
    # plt.savefig("cf.jpeg")
    # writer.add_figure("Confusion Matrix", fig)

    print("Test, Accuracy: {}".format(accuracy_test))

    # Saving parameters network
    torch.save(net.state_dict(), os.path.join(EXP_NAME,"model.pt"))


    hyperparam = {"BATCH_SIZE": BATCH_SIZE, "LEARNING_RATE": LEARNING_R, "EPOCHS": EPOCHS}
    writer.add_text("architecture/", str(net))
    writer.add_hparams(hyperparam,{'hparam/accuracy': accuracy_test})
    writer.close()