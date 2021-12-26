import os
import pathlib
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np 
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.metrics import roc_curve
from sklearn.metrics import zero_one_loss
import math 


def process_image(image_path: str):
    img = Image.open(image_path)
    img = ImageOps.grayscale(img)
    img = img.resize(size=(96,96))
    img = np.ravel(img)/255
    return img 



def process_folder(folder: pathlib.PosixPath):
    processed = [] 
    for img in folder.iterdir():
        if img.suffix == ".jpg":
            try : 
                processed.append(process_image(image_path=str(img)))
            except Exception as _:
                continue
    processed = pd.DataFrame(processed)
    processed["class"] = folder.parts[-1]
    
    return processed
    
           
train_buildings = process_folder(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\images\\seg_train\\buildings"))
train_forest = process_folder(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\images\\seg_train\\forest"))                        
train_glacier = process_folder(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\images\\seg_train\\glacier")) 
train_mountain = process_folder(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\images\\seg_train\\mountain"))
train_sea = process_folder(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\images\\seg_train\\sea"))
train_street = process_folder(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\images\\seg_train\\street"))

print("buildings train shape  :",train_buildings.shape)
print("forest train shape  :",train_forest.shape)
print("glacier train shape  :",train_glacier.shape)
print("mountain train shape  :",train_mountain.shape)
print("sea train shape  :",train_sea.shape)
print("street train shape  :",train_street.shape)
print("________________________________")


train_set = pd.concat([train_buildings,train_forest,train_glacier,train_mountain
                       ,train_sea,train_street],axis=0)
with open("train_set.pkl","wb") as f:
    pickle.dump(train_set,f)


test_buildings = process_folder(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\images\\seg_test\\buildings"))
test_forest = process_folder(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\images\\seg_test\\forest"))                        
test_glacier = process_folder(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\images\\seg_test\\glacier")) 
test_mountain = process_folder(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\images\\seg_test\\mountain"))
test_sea = process_folder(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\images\\seg_test\\sea"))
test_street = process_folder(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\images\\seg_test\\street"))

print("buildings test shape  :",test_buildings.shape)
print("forest test shape  :",test_forest.shape)
print("glacier test shape  :",test_glacier.shape)
print("mountain test shape  :",test_mountain.shape)
print("sea test shape  :",test_sea.shape)
print("street test shape  :",test_street.shape)
print("________________________________________")

test_set = pd.concat([test_buildings,test_forest,test_glacier,test_mountain
                      ,test_sea,test_street],axis=0)

with open("test_set.pkl","wb") as f:
    pickle.dump(test_set,f)
 

val_buildings = process_folder(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\images\\seg_val\\buildings"))
val_forest = process_folder(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\images\\seg_val\\forest"))                        
val_glacier = process_folder(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\images\\seg_val\\glacier")) 
val_mountain = process_folder(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\images\\seg_val\\mountain"))
val_sea = process_folder(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\images\\seg_val\\sea"))
val_street = process_folder(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\images\\seg_val\\street"))   
 
val_set = pd.concat([val_buildings,val_forest,val_glacier,val_mountain
                      ,val_sea,val_street],axis=0)

with open("val_set.pkl","wb") as f:
    pickle.dump(val_set,f)
    
print("buildings  validation shape  :",val_buildings.shape)
print("forest validation shape  :",val_forest.shape)
print("glacier  validation shape  :",val_glacier.shape)
print("mountain  validation shape  :",val_mountain.shape)
print("sea  validation shape  :",val_sea.shape)
print("street  validation shape  :",val_street.shape)
print("________________________________________") 
    
train_set = shuffle(train_set).reset_index(drop=True)
test_set = shuffle(test_set).reset_index(drop=True)
val_set = shuffle(val_set).reset_index(drop=True)


print("train set shape :",train_set.shape)
print("test set shape:" ,test_set.shape)
print(" validation set shape",val_set.shape)
print("______________________________________")
  
x_train = train_set.drop("class",axis=1)
y_train = train_set["class"]

x_test = test_set.drop("class",axis=1)
y_test = test_set["class"]


x_val = val_set.drop("class",axis=1)
y_val = val_set["class"]


y_train = tf.keras.utils.to_categorical(y_train.factorize()[0], num_classes=6)
y_test = tf.keras.utils.to_categorical(y_test.factorize()[0], num_classes=6)
y_val = tf.keras.utils.to_categorical(y_val.factorize()[0], num_classes=6)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(9217, activation='relu'),
    tf.keras.layers.Dense(4608, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')]
)

history = model.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=5600,
    validation_data=(x_val, y_val)
) 

plt.plot(history.history['loss'])
plt.title("model loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'],loc = "upper left")
plt.show()


y_pred=model.predict(x_test) 
y_pred=np.argmax(y_pred, axis=1)
Y_test=np.argmax(y_test, axis=1)    
cm = confusion_matrix(Y_test,y_pred)
plt.figure(figsize=(7,5))
sn.heatmap(cm,annot=True)
plt.xlabel("predict")
plt.ylabel("truth")


datadir = "H:\\Data sets\\images\\seg_train"
catagories = ["buildings","forest"]

train_data=[]
def create_train_data():
    for category in catagories:
        path=os.path.join(datadir, category)
        class_num=catagories.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img))
                new_array=cv2.resize(img_array,(96,96))
                train_data.append([new_array,class_num])
            except Exception as e:
                pass
create_train_data()

lenofimage = len(train_data)

X=[]
y=[]

for categories, label in train_data:
    X.append(categories)
    y.append(label)
X= np.array(X).reshape(lenofimage,-1)
X=X/255.0
Y=np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2)

print("________________________________________")
print("train data shape:",X_train.shape)
print("test data shape:",X_test.shape)

model2 = SVC(C=20,kernel="linear",gamma="auto")
model2.fit(X_train,y_train)
print("___________________________________________________")
print("The accuracy of SVC is :",math.ceil(100 * model2.score(X_test,y_test)))

cm2 = confusion_matrix(y_test,model2.predict(X_test))
plt.figure(figsize=(7,5))
sn.heatmap(cm2,annot=True)
plt.xlabel("predict")
plt.ylabel("truth")

fpr2,tpr2,threshold2 = roc_curve(y_test,model2.predict(X_test))
print("Fpr is :",fpr2,"\n tpr is :",tpr2,"\n threshold is :",threshold2)

print("The loss curve is :",zero_one_loss(y_test,model2.predict(X_test),normalize=False))