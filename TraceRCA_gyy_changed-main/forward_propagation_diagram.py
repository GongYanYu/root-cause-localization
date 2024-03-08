from collections import defaultdict
from datetime import datetime
from train_config import FEATURE_NAMES


class Invocation:
    """
    调用
    """

    def __init__(self, **kwargs):
        self.trace_id = kwargs.get('trace_id')
        self.source = kwargs.get('source')
        self.target = kwargs.get('target')
        # 延迟
        self.latency = kwargs.get('latency')
        self.http_status = kwargs.get('http_status')
        self.cpu_use = kwargs.get('cpu_use')
        self.mem_use_percent = kwargs.get('mem_use_percent')
        self.mem_use_amount = kwargs.get('mem_use_amount')
        self.file_write_rate = kwargs.get('file_write_rate')
        self.file_read_rate = kwargs.get('file_read_rate')
        self.net_send_rate = kwargs.get('net_send_rate')
        self.net_receive_rate = kwargs.get('net_receive_rate')
        self.endtime = kwargs.get('endtime')
        # 0 is normal  1 is abnormal
        self.label = kwargs.get('label')


class Edge:
    """
    调用边
    """

    def __init__(self, source, target):
        self.source = source
        self.target = target

    def update(self, data_list=[]):
        data_list


class ForwardPropagationDiagram:

    def __init__(self):
        # 图
        self.diagram = defaultdict(list)

    def update(self, inv: Invocation):
        self.diagram[(inv.source, inv.target)].append(inv)
