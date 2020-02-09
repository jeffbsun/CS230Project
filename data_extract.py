import quandl

def extract_data():
    quandl.ApiConfig.api_key = '1sQR4zpQidV69Kx6syR3'

    # max_code = 87001
    # for code_num in range(max_code):
    #     code = 'HKEX/' + str(code_num)
    # quandl.get('HKEX/83079', start_date='2018-01-01', end_date='2018-12-31')
    x = quandl.get('HKEX/83079', start_date='2020-01-01', end_date='2020-01-15')
    print(type(x))
    print(x)


def main():
    extract_data()


if __name__ == '__main__':
    main()