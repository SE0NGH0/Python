from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')
print(mnist.data)

import matplotlib.pyplot as plt
plt.imshow(mnist.data.values[0].reshape(28,28),cmap = 'gray')

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(mnist.data,mnist.target,test_size=0.1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

clf = RandomForestClassifier()
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
print(accuracy_score(y_test,pred))

import glob
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings('ignore')

for path in glob.glob('img/*.png'):
    #print(path)
    img = Image.open(path).convert('L')
    #print(img)
    #plt.imshow(img,cmap='gray')
    data = np.resize(img,(28,28))
    data = 255-data
    plt.imshow(data,cmap='gray')
    data = data.reshape(1,-1)
    pred = clf.predict(data)
    print(pred)
    plt.show()