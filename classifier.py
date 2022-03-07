import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.plyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisiticRegression
from sklearn.metrics import accuracy_score
from PIL import Image

import PIL.ImageOps
X,y=fetch_openml('mnist_784',version=1,return_X_y=True)


X = np.load('image.npz') ['arr_0']
y = pd.read_csv ("labels.csv") ["labels"]
print (pd.Series (y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "0", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses = len(classes)

x train, x_test, y_train, y_test = train test_split(x, y, random_state=9, train size=3500, test_size=500)
xtrainscaled=xtrain/255.0
xtestscaled=xtest/255.0
clf=LogisiticRegression(solver='saga',multi_class='multinomial').fit(xtrainscaled,ytrain)

def get_prediction(image):
    im_pil=Image.open(image)
    imagebw=impil.convert('L')
    imagebwresize=imagebw.resize((28,28),Image.ANTIALIAS)
    pixelfilter=20
    minpixel=np.percentile(imagebwresize,pixelfilter)
    imageScaled=np.clip(imagebwresize-minpixel,0,255)
    maxpixel=np.max(imagebwresize)
    imagebwresizeinvertedscaled=np.asarray(imageScaled)/maxpixel
    testSample=np.array(imagebwresizeinvertedscaled).reshape(1,784)
    testPredict=clf.predict(testSample)
    return testPredict[0]
