import pickle
import numpy as np
from sklearn.svm import SVC
from skimage.feature import hog
from PIL import Image
from sklearn import tree

Xtr=np.loadtxt("TrainData.csv")
Xlabel=np.loadtxt("TrainLabels.csv")

def extractFeatures(Xtr):
    greyimgs = []
    for img in Xtr:
        image = Image.fromarray(img.reshape([28, 28]))
        image = image.convert("L")
        greyimgs.append(image)

    hogi = []
    for image in greyimgs:
        hog_image = hog(image, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), transform_sqrt=True,
                        block_norm="L1")
        hogi.append(hog_image)

    return hogi

def DT(X_train, y_train):
    model = tree.DecisionTreeClassifier(max_depth=5)
    model.fit(X_train, y_train)
    return model


def SVM(X_train, y_train):
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model


def main():
    X = extractFeatures(Xtr)
    model = DT(X, Xlabel)
    pickle.dump(model, open("decision_tree.pkl", 'wb'))

    model2 = SVM(X, Xlabel)
    pickle.dump(model2, open("support_VM.pkl", 'wb'))


if __name__== "__main__":
  main()



