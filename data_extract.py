import numpy as np
import quandl
import time


def extract_data():
    quandl.ApiConfig.api_key = '1sQR4zpQidV69Kx6syR3'

    all_prices = []
    max_code = 87001
    for code_num in range(10):
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
        for i in range(len_diff):
            list.insert(0, 0.0)
        all_prices_list.append(list)

    array = np.array(all_prices_list)
    # array.shape = (len(all_prices), len(all_prices[0]))

    return array


def main():
    print('Starting')
    start_time = time.time()
    extract_data()
    end_time = time.time()
    print('time spent: {0:.2f}'.format(end_time - start_time))


if __name__ == '__main__':
    main()