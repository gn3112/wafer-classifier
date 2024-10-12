import torch
from torchvision.transforms import Compose, Normalize
from torch.utils.data import DataLoader

# from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from utils import *
from dataset import CustomTensorDataset
from model import WaferNet

device = torch.device("cpu")

y_true = []
y_pred = []
running_correct_test = 0
idx_p = 0
fail_examples = {}

# Loading dataset
df  = pd.read_pickle("waferImg26x26.pkl")
images = df.images.values
labels   = df.labels.values
labels = np.asarray([str(l[0]) for l in labels])

classes_name = np.unique(labels)

class_to_idx = class_to_idx(classes_name)
idx_to_class = idx_to_class(classes_name)

indices = list(range(len(images)))
test_indices = indices[112:168]

T = Compose([Normalize(torch.tensor([0.2118, 0.5966, 0.1916]) , torch.tensor([0.4086, 0.4906, 0.3936]))])
tensor_dataset_test = to_tensor_dataset_format(test_indices, images, labels, class_to_idx)
test_data = CustomTensorDataset(tensor_dataset_test, transform=T)
test_loader = DataLoader(test_data, batch_size=12)

net = WaferNet().to(device)
net.load_state_dict(torch.load('EXP/Final_model_benchmark_60_ep/model.pt', weights_only=True))
net.eval()

for inputs, labels in test_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    out = net(inputs)

    probas, pred = torch.max(torch.exp(out),1)
    
    y_true.extend(labels.tolist())
    y_pred.extend(pred.tolist())

    running_correct_test += torch.sum(pred==labels)

    accuracy_test = running_correct_test.type(torch.DoubleTensor)/len(test_indices)
    
    # Move this to validation for tensorboard
    for i in range(inputs.size()[0]):
        i_img = len(y_pred)-inputs.size()[0]+i
        if y_pred[i_img] != y_true[i_img]:

            fail_examples[idx_p] = {'img_file_name':"fail_{}.jpeg".format(idx_p),
                    'class':idx_to_class[y_true[i_img]],
                    'predicted':idx_to_class[y_pred[i_img]],
                    'confidence':round(probas[i].item(),3)}
            idx_p += 1
            # writer.add_image("Fail_Images/fail{}".format(idx_p),inputs[i,:,:,:].view(3,26,26))
            # writer.add_text("Fail_meta/fail{}".format(idx_p), str(fail_examples))

# cf = confusion_matrix(y_true, y_pred, labels=list(range(9)))

# fig = plt.figure()
# ax = fig.add_subplot(111)
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
print(fail_examples)