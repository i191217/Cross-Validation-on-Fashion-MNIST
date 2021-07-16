import pickle
import numpy as np

from skimage.feature import hog
from PIL import Image


Xts=np.loadtxt("TestData.csv")

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



def main():
    Xtest = extractFeatures(Xts)

    model=pickle.load(open('decision_tree.pkl', 'rb'))
    predict= model.predict(Xtest)
    np.savetxt("decision_tree.csv", predict)

    model2 = pickle.load(open('support_VM.pkl', 'rb'))

    predict2 = model2.predict(Xtest)
    np.savetxt("support_VM.csv", predict2)



if __name__== "__main__":
  main()
