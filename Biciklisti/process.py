import cv2
import numpy as np
from sklearn.svm import SVC # SVM klasifikator
from joblib import dump, load
import matplotlib
import matplotlib.pyplot as plt
# prikaz vecih slika
matplotlib.rcParams['figure.figsize'] = 16,12


def train_or_load_model(train_positive_images_paths, train_negative_images_path):

    return load('modelNovi.joblib')

    positive_features = []
    negative_features = []

    labels = []

    nbins = 9  # broj binova
    cell_size = (8, 8)  # broj piksela po celiji
    block_size = (3, 3)  # broj celija po bloku

    dimm = (45, 45)

    hog = cv2.HOGDescriptor(_winSize=(dimm[1] // cell_size[1] * cell_size[1],
                                      dimm[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    for i in range(0, len(train_negative_images_path)):
        print(train_negative_images_path[i])
        image = load_image(train_negative_images_path[i])
        image = cv2.resize(image, dimm)
        negative_features.append(hog.compute(image))
        labels.append(1)

    for i in range(0, len(train_positive_images_paths)):
        print(train_positive_images_paths[i])
        image = load_image(train_positive_images_paths[i])
        image = cv2.resize(image, dimm)
        positive_features.append(hog.compute(image))
        labels.append(0)

    negative_features = np.array(negative_features)
    positive_features = np.array(positive_features)

    x1 = np.vstack((negative_features, positive_features))
    y1 = np.array(labels)
    x1 = reshape_data(x1)

    clf_svm = SVC(kernel='linear', probability=True)
    clf_svm.fit(x1, y1)
    dump(clf_svm, 'modelNovi.joblib')

    return clf_svm

def count_bicycles(image_path, modelsvc):

    nbins = 9  # broj binova
    cell_size = (8, 8)  # broj piksela po celiji
    block_size = (3, 3)  # broj celija po bloku

    dimm = (45, 45)

    hog = cv2.HOGDescriptor(_winSize=(dimm[1] // cell_size[1] * cell_size[1],
                                      dimm[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)


    img = load_image(image_path)
    y=0
    x=0
    cnt=0

    while x+520 <= img.shape[1]:
        while y+330 <= img.shape[0]:
            niz=[]
            img1 = img[y:y+330,x:x+520]
            img1 = cv2.resize(img1, (45, 45))
            niz += [hog.compute(img1)]
            niz = np.array(niz)
            niz = reshape_data(niz)
            if modelsvc.predict(niz) == 0:
                cnt = cnt+1
                img = cv2.rectangle(img, (x, y), (x + 520, y + 330), (0, 255, 0), 3)
                y = y + 100
            y=y+50
        y=0
        x=x+120

    return cnt

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def reshape_data(input_data):
    aa, nx, ny = input_data.shape
    return input_data.reshape((aa, nx*ny))
