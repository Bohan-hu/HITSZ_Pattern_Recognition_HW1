from util import *
path = os.curdir
image_train_path = os.path.join(path, "train-images.idx3-ubyte")
label_train_path = os.path.join(path, "train-labels.idx1-ubyte")
image_test_path = os.path.join(path, "t10k-images.idx3-ubyte")
label_test_path = os.path.join(path, "t10k-labels.idx1-ubyte")

images_train = load_image(image_train_path)
images_test = load_image(image_test_path)
labels_train = load_label(label_train_path)
labels_test = load_label(label_test_path)

K = 10
print("\n--------"+str(K)+"近邻（K=10）--------")
sum = 0
for i in range(0, images_test.shape[0]):
    predict = K_NN(images_test[i], images_train, labels_train)
    if predict == labels_test[i]:
        sum = sum + 1
    print(str(sum) + ' / ' + str(i + 1) + ' -- Accuracy: ' + str(sum / (i + 1)),end='\r')
print('\n')