import os
import numpy as np
import struct


# 加载图像文件
def load_image(fn):
    binfile = open(fn, 'rb')
    buffers = binfile.read()
    magic, num, rows, cols = struct.unpack_from('>IIII', buffers, 0)
    bits = num * rows * cols
    images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
    binfile.close()
    images = np.reshape(images, [num, rows * cols])
    return images


# 加载Label文件
def load_label(fn):
    binfile = open(fn, 'rb')
    buffers = binfile.read()
    magic, num = struct.unpack_from('>II', buffers, 0)
    labels = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
    binfile.close()
    labels = np.reshape(labels, [num])
    return labels


def threshold(images):
    return (images >= 127) * 255  # 大于127，则置为1


def selectClass(images, labels, classes):
    index = []
    for i in classes:
        index += np.where(labels == i)[0].tolist()
    return images[np.array(index)], labels[np.array(index)]


def K_NN(image, train_set, train_label, K=10):  # K=1时，就是最近邻算法
    # 将这个768*1的向量复制N份，N是训练集样本数，成为一个N*768的矩阵，并将两个矩阵相减，得到差值
    diff = np.tile(image, (train_set.shape[0], 1)) - train_set
    dist = ((diff ** 2).sum(axis=1)) ** 0.5  # 计算该测试样本和每个训练样本的欧氏距离
    dist_label = np.column_stack((dist, train_label))  # 将Label拼接到欧氏距离向量上
    label_sorted = dist_label[dist_label[:, 0].argsort()][0:K, 1]  # 按第一列排序，取排序后的标签前K个
    return np.argmax(np.bincount(label_sorted.astype(np.int32)))  # 取众数，即为分类的结果


if __name__ == '__main__':
    # 加载训练集和测试集，文件放在根目录下
    path = os.curdir
    image_train_path = os.path.join(path, "train-images.idx3-ubyte")
    label_train_path = os.path.join(path, "train-labels.idx1-ubyte")
    image_test_path = os.path.join(path, "t10k-images.idx3-ubyte")
    label_test_path = os.path.join(path, "t10k-labels.idx1-ubyte")

    images_train = load_image(image_train_path)
    images_test = load_image(image_test_path)
    labels_train = load_label(label_train_path)
    labels_test = load_label(label_test_path)


    # 统计分类正确率
    sum = 0
    print("--------最近邻--------")
    for i in range(0, images_test.shape[0]):
        predict = K_NN(images_test[i], images_train, labels_train)
        if predict == labels_test[i]:
            sum = sum + 1
        if i % 100 == 0 and i != 0:
            print(str(i) + ' / 10000 -- Accuracy: ' + str(sum / i))


    print("--------K近邻（K=10）--------")
    sum = 0
    for i in range(0, images_test.shape[0]):
        predict = K_NN(images_test[i], images_train, labels_train)
        if predict == labels_test[i]:
            sum = sum + 1
        if (i + 1) % 100 == 0:
            print(str(sum) + ' / ' + str(i + 1) + ' -- Accuracy: ' + str(sum / (i + 1)))
