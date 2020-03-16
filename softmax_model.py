import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import data_extract
import numpy as np

# This is a model that outputs not just a predicted price, but rather a softmax of the predicted price probability dist.
# it uses more than price history - uses all available features

NUM_EPOCHS = 500
NUM_Y_BUCKETS = 10

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
        self.fc3 = nn.Linear(84, NUM_Y_BUCKETS)
        # self.sm = nn.Softmax(10)
        # self.fc1 = nn.Linear(245, 1)

    def forward(self, x):
        # return torch.tanh(self.fc1(x))
        x = self.cv1(x)
        x = self.mx1(x)
        x = self.cv2(x)
        x = self.mx2(x)
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = F.softmax(x, dim=2)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def reformat_y(y):
    # Reformats the label y to be a one-hot vector to denote which range/bucket the y value falls into.
    # This is done so that softmax can operate to predict the y bucket
    new_y = []
    bucket_size = 0.01
    bucket_mean = 1.0
    for stock in y:
        y_value = stock[0][0]
        one_hot_index = -1
        bottom_bucket = bucket_mean - bucket_size * ((NUM_Y_BUCKETS / 2) - 1)
        for i in range(NUM_Y_BUCKETS):
            if i == 0:
                if y_value < bottom_bucket:
                    assert(one_hot_index == -1)
                    one_hot_index = i
            elif i == NUM_Y_BUCKETS - 1:
                if y_value >= bucket_mean + bucket_size * ((NUM_Y_BUCKETS / 2) - 1):
                    assert(one_hot_index == -1)
                    one_hot_index = i
            elif bottom_bucket + bucket_size * (i-1) <= y_value < bottom_bucket + bucket_size * i:
                assert(one_hot_index == -1)
                one_hot_index = i
        assert(one_hot_index >= 0)
        one_hot = [0.0] * NUM_Y_BUCKETS
        assert(len(one_hot) == NUM_Y_BUCKETS)
        one_hot[one_hot_index] = 1.0
        new_y.append([one_hot])

    return np.array(new_y)


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

    y = reformat_y(y / (data[0:num_train, 0:1, -2:-1] + epsilon))

    # Create the dev set
    x_dev = data[num_train:num_train+num_dev, :, 0:-1]
    y_dev = data[num_train:num_train+num_dev, 0:1, -1:]
    y_dev = reformat_y(y_dev / (data[num_train:num_train+num_dev, 0:1, -2:-1] + epsilon))

    # Create the test set
    x_test = data[num_train + num_dev : num_stocks, :, 0:-1]
    y_test = data[num_train + num_dev : num_stocks, 0:1, -1:]
    y_test = reformat_y(y_test / (data[num_train + num_dev : num_stocks, 0:1, -2:-1] + epsilon))

    net = Net()
    print(net)

    params = list(net.parameters())
    print(len(params))
    print(params[0].size())

    # print(y)

    for _ in range(NUM_EPOCHS):
        nn_input = torch.tensor(x).float()
        predicted = net(nn_input)

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

    # Show a few predicted results from test set:
    print(predicted_test[:5, :, :])
    print('a few x_test stock price histories')
    np.set_printoptions(threshold=np.inf)
    for i in range(5):
        for j in range(x_test.shape[2]):
            print(x_test[i, 0, j])
        print('\n\n\nstock')


def main():
    print('Starting')
    start_time = time.time()

    run_model()

    end_time = time.time()
    print('time spent: {0:.2f}'.format(end_time - start_time))


if __name__ == '__main__':
    main()
