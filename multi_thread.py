from util import *
import threading
import time


class calcThread(threading.Thread):
    def __init__(self, tid, images_test, labels_test):
        threading.Thread.__init__(self)
        self.images_test = images_test
        self.labels_test = labels_test
        self.sum = 0
        self.tid = tid
        self.size = self.images_test.shape[0]
        self.pos = 0
        self.finished = False
        self.predict = 0

    def run(self):
        for i in range(0, self.size):
            self.predict = K_NN(self.images_test[i], images_train, labels_train)
            if self.predict == self.labels_test[i]:
                self.sum = self.sum + 1
            self.pos = self.pos + 1
        self.finished = True

    def get_result(self):
        return self.sum

    def get_process(self):
        return round((self.pos) / self.size, 2)

    def get_percentage(self):
        return round(self.sum / (self.pos + 1), 2)

path = os.curdir
image_train_path = os.path.join(path, "train-images.idx3-ubyte")
label_train_path = os.path.join(path, "train-labels.idx1-ubyte")
image_test_path = os.path.join(path, "t10k-images.idx3-ubyte")
label_test_path = os.path.join(path, "t10k-labels.idx1-ubyte")

images_train = load_image(image_train_path)
images_test = load_image(image_test_path)
labels_train = load_label(label_train_path)
labels_test = load_label(label_test_path)

# 拆分训练集和测试集，分给不同的线程
num_batch = 16
images_train_batches = []
images_test_batches = []
labels_train_batches = []
labels_test_batches = []

batch_size = int(images_test.shape[0] / num_batch)
for i in range(num_batch):
    images_test_batches.append(images_test[i * batch_size:(i + 1) * batch_size])
    labels_test_batches.append(labels_test[i * batch_size:(i + 1) * batch_size])

threads = []

for i in range(num_batch):
    t = calcThread(i, images_test_batches[i], labels_test_batches[i])
    threads.append(t)
    t.start()

finished = False
while not finished:
    finished = True
    outstr = ""
    for t in threads:
        outstr += ('Thread ' + str(t.tid) + '\tAccuracy:' + str(t.get_percentage()) + "\t Progress:" + str(
            t.get_process()) + '\n')
        finished = finished & t.finished
    print(outstr, flush=True)
    time.sleep(3)

for t in threads:
    t.join()

sum =0
for t in threads:
    sum += t.get_result()

print(sum)
