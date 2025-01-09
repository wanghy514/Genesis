
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
from tensorflow.python.summary import summary_iterator

class TensorBoardEvents:
    def  __init__(self, log_path):
        
        self.data = {}
        self.log_path = log_path        
        self.data = []
        _path = self.log_path
        events = summary_iterator.summary_iterator(_path)
        for e in events:
            for i in range(len(e.summary.value)):
                self.data.append(
                    (e.step, e.summary.value[0].tag, e.summary.value[0].simple_value)
                )
        self.data = sorted(self.data, key=lambda item: item[0])

    def names(self):
        return set(v[1] for v in self.data)

    def get(self, name):

        return {
            'step': [v[0] for v in self.data if v[1] == name],
            'value': [v[2] for v in self.data if v[1] == name],
        }
    
    def _get_attr(self, tag_name, attr_name):
        _path = self.log_path
        events = summary_iterator.summary_iterator(_path)
        for e in events:
            if (len(e.summary.value) > 0):
                assert len(e.summary.value) == 1
                if (e.summary.value[0].tag == tag_name):
                    item = getattr(e.summary.value[0], attr_name)
                    return item


    def get_simple_value(self, tag_name):        
        return self._get_attr(tag_name, 'simple_value')
    

    def get_pr_curve(self, tag_name):
        return self._get_attr(tag_name, 'tensor')

if __name__ == "__main__":

    log_path = "./logs/learn2grasp/events.out.tfevents.1736622235.huayanwang-HP-Laptop-15-dw4xxx.847621.0"
    smoothing_window_size = 1
    tbe = TensorBoardEvents(log_path)
    print (tbe.names())
    curve = tbe.get("Train/mean_reward")
    df = pd.DataFrame({"v": curve["value"]})
    smoothed_curve = np.array(
        df.rolling(smoothing_window_size).mean()["v"]
    )
    plt.plot(curve["step"], smoothed_curve)
    plt.show(block=True)

