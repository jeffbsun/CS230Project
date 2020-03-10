import math
import numpy as np
import quandl
import time

NUM_VALUES_TO_READ = 200


# This extracts price, bid, ask, p/e, high, low, volume, and turnover of stocks and outputs an (n, f, t) array
# where n is the number of stocks, f is the number of features (8), and t is the number of days of price history
def extract_all_data(num_values = NUM_VALUES_TO_READ):
    quandl.ApiConfig.api_key = '1sQR4zpQidV69Kx6syR3'

    all_prices = []
    max_code = 87001
    for code_num in range(num_values):
        code = 'HKEX/' + str(code_num).zfill(5)
        try:
            x = quandl.get(code, start_date='2018-01-01', end_date='2018-12-31')
        except:
            continue
        # The first column is the price, so append that
        prices = x.values[:, 0]
        bid = x.values[:, 3]
        ask = x.values[:, 4]
        pe = x.values[:, 5]
        high = x.values[:, 6]
        low = x.values[:, 7]
        volume = x.values[:, 9]
        turnover = x.values[:, 10]

        data_for_stock = [prices, bid, ask, pe, high, low, volume, turnover]
        all_prices.append(data_for_stock)

    print('all_prices len:', len(all_prices))
    max_len = 0
    for stock in all_prices:
        for feature in stock:
            max_len = max(max_len, len(feature))

    print('max len', max_len)
    all_prices_list = []
    for stock in all_prices:
        feature_list = []
        for feature in stock:
            len_diff = max_len - len(feature)
            list = feature.tolist()
            list = [x if x is not None and not math.isnan(x) else 0.0 for x in list]
            for i in range(len_diff):
                list.insert(0, 0.0)
            feature_list.append(list)
        all_prices_list.append(feature_list)

    array = np.array(all_prices_list)
    # randomly shuffle the data in the first dimension (stocks)
    np.random.shuffle(array)
    print('data shape: ', array.shape)
    # array.shape = (len(all_prices), len(all_prices[0]))

    return array


# This extracts only the price history of stocks and outputs an (n, t) array where n is the number of stocks, and
# t is the number of days of price history
def extract_data():
    quandl.ApiConfig.api_key = '1sQR4zpQidV69Kx6syR3'

    all_prices = []
    max_code = 87001
    for code_num in range(NUM_VALUES_TO_READ):
        code = 'HKEX/' + str(code_num).zfill(5)
        try:
            x = quandl.get(code, start_date='2018-01-01', end_date='2018-12-31')
        except:
            continue
        # The first column is the price, so append that
        all_prices.append(x.values[:, 0])

    print(len(all_prices))
    max_len = 0
    for part in all_prices:
        max_len = max(max_len, len(part))

    all_prices_list = []
    for part in all_prices:
        len_diff = max_len - len(part)
        list = part.tolist()
        list = [x if not math.isnan(x) else 0.0 for x in list]
        for i in range(len_diff):
            list.insert(0, 0.0)
        all_prices_list.append(list)

    array = np.array(all_prices_list)
    # array.shape = (len(all_prices), len(all_prices[0]))

    return array


def main():
    print('Starting')
    start_time = time.time()
    extract_all_data()
    end_time = time.time()
    print('time spent: {0:.2f}'.format(end_time - start_time))


if __name__ == '__main__':
    main()