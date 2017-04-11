import threading
from pycuda import driver

class gpuThread(threading.Thread):
    def __init__(self, gpuid, id_num):
        threading.Thread.__init__(self)
        self.ctx  = driver.Device(gpuid).make_context()
        self.device = self.ctx.get_device()
        self.id = id_num

    def run(self):
        print "%s has device %s, api version %s"  \
             % (self.getName(), self.device.name(), self.ctx.get_api_version())
        for i in range(1000):
            print self.id, i
        # Profit!

    def join(self):
        self.ctx.detach()
        threading.Thread.join(self)

driver.init()
ngpus = driver.Device.count()
threads = []
for i in range(ngpus):
    t = gpuThread(i, i)
    t.start()
    threads.append(t)
for t in threads:
    t.join()
