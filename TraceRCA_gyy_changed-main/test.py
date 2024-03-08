import pickle

import pandas as pd


def load_file_to_scv(path, trans_to_path):
    with open(str(path), 'rb') as f:
        data = pickle.load(f)
    data.to_csv(trans_to_path)


dataset_path = r'..\datasets\A\uninjection'


# load_file_to_scv(f'{dataset_path}/admin-order_abort_1011.pkl',
#                  f'{dataset_path}/admin-order_abort_1011_data.scv')

def test_data_frame():
    list = [{'trace_id': 'df6f230fa9712a90799d39c1c45a5777', 'timestamp': [1570798027423252.0, 1570798014175730.0],
             'latency': [605.0, 1989.0], 'http_status': [304.0, 304.0],
             'cpu_use': [0.3780308728795971, 0.3780308728795971], 'mem_use_percent': [0.020359375, 0.020359375],
             'mem_use_amount': [142065664.0, 142065664.0], 'file_write_rate': [0.0, 0.0], 'file_read_rate': [0.0, 0.0],
             'net_send_rate': [16600.32356554635, 16600.32356554635],
             'net_receive_rate': [31423.27482392439, 31423.27482392439],
             'endtime': [1570798027423857.0, 1570798014177719.0],
             's_t': [('ts-ui-dashboard', 'ts-ui-dashboard'), ('istio-ingressgateway', 'ts-ui-dashboard')], 'label': 0},
            {'trace_id': 'cc8da682b10a53aa04c9071ea3fa8ed3',
             'timestamp': [1570798018853124.0, 1570798018032425.0, 1570798017499316.0, 1570798014208121.0],
             'latency': [14632.0, 8525.0, 10011.0, 15948.0], 'http_status': [200.0, 200.0, 200.0, 200.0],
             'cpu_use': [2.406507618850728, 8.861333558021748, 8.861333558021748, 2.406507618850728],
             'mem_use_percent': [0.44608306884765625, 0.4212455749511719, 0.4212455749511719, 0.44608306884765625],
             'mem_use_amount': [1075195904.0, 1027620864.0, 1027620864.0, 1075195904.0],
             'file_write_rate': [0.0, 0.0, 0.0, 0.0], 'file_read_rate': [0.0, 0.0, 0.0, 0.0],
             'net_send_rate': [9928.92480800143, 26908.97892143444, 26908.97892143444, 9928.92480800143],
             'net_receive_rate': [4368.914091802107, 17856.47202510055, 17856.47202510055, 4368.914091802107],
             'endtime': [1570798018867756.0, 1570798018040950.0, 1570798017509327.0, 1570798014224069.0],
             's_t': [('ts-order-other-service', 'ts-order-other-service'), ('ts-station-service', 'ts-station-service'),
                     ('ts-order-other-service', 'ts-station-service'),
                     ('istio-ingressgateway', 'ts-order-other-service')], 'label': 0}]
    data = pd.DataFrame(list[0])

    print(data)


test_data_frame()
