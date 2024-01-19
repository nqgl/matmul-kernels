import torch


class TimedFunc(object):
    def __init__(self, func, name=None):
        self.func = func
        self.t = None
        self.name = name
        self.times = []


    def __call__(self, *args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        ret = self.func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        t = start.elapsed_time(end)
        if self.name is not None:
            print(self.name, ": ", t)
        self.times.append(t)
        self.t = t
        return ret
    


