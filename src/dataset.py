from torch.utils.data import Dataset

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