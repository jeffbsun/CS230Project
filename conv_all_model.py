import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import data_extract

# This is a convolutional model that uses more than price history - uses all available features

NUM_EPOCHS = 100

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        input_length = 245
        kernel_size = 7
        max_pool_1_size = 3
        self.cv1 = nn.Conv1d(8, 20, kernel_size)
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
        x = torch.relu(self.fc3(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def run_model():
    num_train = 900
    num_dev = 300
    num_test = 300
    num_stocks = num_train + num_dev + num_test
    data = data_extract.extract_all_data(num_stocks)
    print('data shape', data.shape)
    x = data[0:num_train, :, 0:-1]
    y = data[0:num_train, 0:1, -1:] # y is the price at the last time step
    # Make y be in the scale of the second-to-last time step so it becomes a value near 0
    epsilon = 0.0000001
    y = y / (data[0:num_train, 0:1, -2:-1] + epsilon)

    # Create the dev set
    x_dev = data[num_train:num_train+num_dev, :, 0:-1]
    y_dev = data[num_train:num_train+num_dev, 0:1, -1:]
    y_dev = y_dev / (data[num_train:num_train+num_dev, 0:1, -2:-1] + epsilon)

    # Create the test set
    x_test = data[num_train + num_dev : num_stocks, :, 0:-1]
    y_test = data[num_train + num_dev : num_stocks, 0:1, -1:]
    y_test = y_test / (data[num_train + num_dev : num_stocks, 0:1, -2:-1] + epsilon)

    net = Net()
    print(net)

    params = list(net.parameters())
    print(len(params))
    print(params[0].size())

    for _ in range(NUM_EPOCHS):
        input = torch.tensor(x).float()
        predicted = net(input)

        criterion = torch.nn.MSELoss()
        actual = torch.tensor(y).float()

        # print('shape of predicted and actual', predicted.shape, actual.shape, data.shape, x.shape, y.shape)

        loss = criterion(predicted, actual)

        # Get the dev set loss
        predicted_dev = net(torch.tensor(x_dev).float())
        actual_dev = torch.tensor(y_dev).float()
        loss_dev = torch.nn.MSELoss()(predicted_dev, actual_dev)

        # Run an optimizer step
        net.zero_grad()

        optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        print('loss:', loss, 'dev-loss:', loss_dev)

    # Print test set loss
    predicted_test = net(torch.tensor(x_test).float())
    actual_test = torch.tensor(y_test).float()
    loss_test = torch.nn.MSELoss()(predicted_test, actual_test)
    print('test set loss:', loss_test)


def main():
    print('Starting')
    start_time = time.time()

    run_model()

    end_time = time.time()
    print('time spent: {0:.2f}'.format(end_time - start_time))


if __name__ == '__main__':
    main()
