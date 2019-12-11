# python 2 vs 3 stuff...
import sys
if sys.version_info[0] < 3:
    import Queue as queue
else:
    import queue

import threading
# import time
# import numpy as np


# data loading worker
class Worker(threading.Thread):

    def __init__(self, queue, threadid, sampler=None, params=None):
        threading.Thread.__init__(self)

        self.threadid = threadid
        self.sampler = sampler
        self.params = params
        self.no_exit = True

        # the work queue to put self into when data is ready
        self.queue = queue

    def run(self):
        while True:
            if self.no_exit is False:
                # print "worker %d: exiting" % self.threadid
                # done running
                return

            # print "worker %d: getting data" % self.threadid
            # simulate load times
            # time.sleep(1)
            if self.params is None:
                batch = self.sampler()
            else:
                batch = self.sampler(*self.params)
            # print "worker %d: loaded data, waiting to put..." % self.threadid

            # NOTE: the queue class should BLOCK on put if the queue is being
            # used or is full.
            self.queue.put(batch)
            # print "worker %d: finished putting data" % self.threadid


class DataLoaderPool():

    def __init__(self, sampler=None, params=None, max_workers=2, max_queue=5):
        self.max_workers = max_workers
        self.queue = queue.Queue(max_queue)

        self.workers = []
        # create the workers
        for i in range(self.max_workers):
            # print "Creating worker %d" % i
            curr_worker = Worker(self.queue, i, sampler=sampler, params=params)
            curr_worker.daemon = True
            self.workers.append(curr_worker)

            # start filling the queue
            curr_worker.start()

    def get(self):
        # NOTE: the queue class from python is a multi producer multi consumer
        # queue. By default, get should BLOCK while the queue in use or is
        # empty.
        data = self.queue.get()

        return data

    def clear_workers(self):
        """Helper to clear workers."""
        # seems sometimes that workers will cause
        # print "calling destructor"
        # first set the exit flag for each of the workers.
        for worker in self.workers:
            worker.no_exit = False

        # next clear the queue, the workers might be waiting to add data to
        # the queue.
        # print "clearing queue"
        while not self.queue.empty():
            self.queue.get()

        # print "queue empty, joining threads"
        # now join all the workers
        for worker in self.workers:
            worker.join()
        # print "done joining threads"

    def __del__(self):
        self.clear_workers()
