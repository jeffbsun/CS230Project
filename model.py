import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import data_extract


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        # self.conv1 = nn.Conv2d(1, 6, 3)
        # self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(245, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # # If the size is a square you can only specify a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = x.view(-1, self.num_flat_features(x))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def run_model():
    data = data_extract.extract_data()
    print('data shape', data.shape)
    x = data[:, 0:-1]
    y = data[:, -1:]

    net = Net()
    print(net)

    params = list(net.parameters())
    print(len(params))
    print(params[0].size())

    for _ in range(10):
        input = torch.tensor(x).float()
        predicted = net(input)

        criterion = torch.nn.MSELoss()
        actual = torch.tensor(y).float()
        loss = criterion(predicted, actual)

        net.zero_grad()

        optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        print('loss:', loss)


def main():
    print('Starting')
    start_time = time.time()

    run_model()

    end_time = time.time()
    print('time spent: {0:.2f}'.format(end_time - start_time))


if __name__ == '__main__':
    main()