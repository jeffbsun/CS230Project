import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import data_extract

# This is a convolutional model

NUM_EPOCHS = 200

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        input_length = 245
        kernel_size = 7
        max_pool_1_size = 3
        self.cv1 = nn.Conv1d(1, 20, kernel_size)
        self.mx1 = nn.MaxPool1d(max_pool_1_size)

        self.cv2 = nn.Conv1d(20, 1, kernel_size)
        self.mx2 = nn.MaxPool1d(max_pool_1_size)
        # self.cv2 = nn.Conv1d(1, 20, kernel_size)
        # self.mx1 = nn.MaxPool1d(max_pool_1_size)
        size_1 = int((input_length - kernel_size + 1) / max_pool_1_size)
        size_2 = int((size_1 - kernel_size + 1) / max_pool_1_size)
        self.fc2 = nn.Linear(size_2, 84)
        self.fc3 = nn.Linear(84, 1)
        # self.fc1 = nn.Linear(245, 1)

    def forward(self, x):
        # return torch.tanh(self.fc1(x))
        x = self.cv1(x)
        x = self.mx1(x)
        x = self.cv2(x)
        x = self.mx2(x)
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
    y = data[:, -1:] # y is the price at the last time step
    # Make y be in the scale of the second-to-last time step so it becomes a value near 0
    epsilon = 0.0000001
    y = y / (data[:, -2:-1] + epsilon)

    net = Net()
    print(net)

    params = list(net.parameters())
    print(len(params))
    print(params[0].size())

    for _ in range(NUM_EPOCHS):
        input = torch.tensor(x).float().view(x.shape[0], 1, x.shape[1])
        predicted = net(input)

        criterion = torch.nn.MSELoss()
        actual = torch.tensor(y).float().view(y.shape[0], 1, y.shape[1])
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
