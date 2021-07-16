import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from skimage.feature import hog
from sklearn import metrics
from sklearn.model_selection import cross_val_score, GridSearchCV
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn import tree


Xtr=np.loadtxt("TrainData.csv")
Xlabel=np.loadtxt("TrainLabels.csv")
Xts=np.loadtxt("TestData.csv")

#plt.imshow(Xtr[10].reshape([28,28]))

#plt.show()

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


def DT_CrossVal(X_train, y_train):
    clasifier = tree.DecisionTreeClassifier(max_depth=5)
    all_accuracies = cross_val_score(estimator=clasifier, X=X_train, y=y_train, cv=5)
    print(all_accuracies)
    print(all_accuracies.mean())


def GridSearch(X_train, y_train):
    grid_param = {
        'min_samples_split' : range(10,500,20),
        'criterion': ['gini', 'entropy'],
        'max_depth' : [4,6,8,12]
    }

    gd_sr = GridSearchCV(estimator=tree.DecisionTreeClassifier(),
                         param_grid=grid_param,
                         scoring='accuracy',
                         cv=5,
                         n_jobs=-1)

    gd_sr.fit(X_train, y_train)

    best_parameters = gd_sr.best_params_
    print(best_parameters)

    best_result = gd_sr.best_score_
    print(best_result)


def SVM(X_train, y_train):
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model

def SVM_CrossVal(X_train, y_train):
    clasifier = SVC(kernel='linear')
    all_accuracies = cross_val_score(estimator=clasifier, X=X_train, y=y_train, cv=5)
    print(all_accuracies)
    print(all_accuracies.mean())

def GridSearch2(X_train, y_train):
    grid_param = {
        'kernel':('linear', 'rbf'),
        'C':(1,0.25,0.5,0.75),
        'gamma': (1,2,3,'auto'),
        'decision_function_shape':('ovo','ovr'),
    }

    gd_sr = GridSearchCV(estimator=SVC(),
                         param_grid=grid_param,
                         scoring='accuracy',
                         cv=5,
                         n_jobs=-1)

    gd_sr.fit(X_train, y_train)

    best_parameters = gd_sr.best_params_
    print(best_parameters)

    best_result = gd_sr.best_score_
    print(best_result)




def main():
    X = extractFeatures(Xtr)
    X_train, X_test, y_train, y_test = train_test_split(X, Xlabel, test_size=0.20, random_state=0)
    model = DT(X_train, y_train)
    predict = model.predict(X_test)
    print(metrics.classification_report(y_test, predict))

    DT_CrossVal(X_train, y_train)
    GridSearch(X_train, y_train)


    model2 = SVM(X_train, y_train)
    predict2 = model2.predict(X_test)
    print(accuracy_score(y_test, predict2))

    SVM_CrossVal(X_train, y_train)
    GridSearch2(X_train, y_train)


if __name__== "__main__":
  main()