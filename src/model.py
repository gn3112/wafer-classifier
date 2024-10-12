from torch import nn
from torch.nn import functional as F

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