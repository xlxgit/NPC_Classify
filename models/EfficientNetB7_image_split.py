#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os,shutil
import pandas as pd
import sklearn
import sys
import tensorflow as tf
import time
from tensorflow import keras
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
import pickle

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)
    
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
 
original_dataset_dir = '../data/ABE_mix'
base_dir = './AE_and_BE_kfold'

if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
os.mkdir(base_dir)

train_dir = "AE_and_BE_kfold/train/"
validation_dir = "AE_and_BE_kfold/validation"
print(os.path.exists(train_dir))
print(os.path.exists(validation_dir))



height = 150
width = 150
channels = 3
batch_size = 24
num_classes = 1

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20,) 
    cb=plt.colorbar()
    cb.ax.tick_params(labelsize=18)
    tick_marks = np.arange(len(classes))
    #plt.xticks(tick_marks, classes, rotation=45)
    plt.xticks(tick_marks, classes, fontsize=24,)
    plt.yticks(tick_marks, classes, fontsize=24,)
    thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    for i in range (cm.shape[0]):
        for j in range (cm.shape[1]):
            plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",fontsize=24,
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label', fontsize=24,)
    plt.xlabel('Predicted label', fontsize=24,)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png",dpi=300)
    #plt.show()

def plot_confuse(test_y, y_pred):
    conf_mat = confusion_matrix(y_true=test_y, y_pred=y_pred)
    plt.figure(figsize=(8,7))
    plot_confusion_matrix(conf_mat, range(np.max(test_y)+1))
def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
          previous = smoothed_points[-1]
          smoothed_points.append(previous * factor + point * (1 - factor))
        else:
          smoothed_points.append(point)
    return smoothed_points



"""
for i in range(2):
    x, y = train_generator.next()
    print(x.shape, y.shape)
    print(y)
""" 
    
EfficientNetB7 = keras.applications.EfficientNetB7(include_top = False,
                                       pooling = 'avg',
                                       weights = 'imagenet')
                                       #weights = None)
EfficientNetB7.summary()

for layer in EfficientNetB7.layers[0:-4]:
    layer.trainable = False

model = keras.models.Sequential([
    EfficientNetB7,
    keras.layers.Dense(num_classes, activation = 'sigmoid'),
])
model.compile(loss="binary_crossentropy",
                     #optimizer="sgd", 
                     optimizer="adam", 
                     #optimizer=optimizers.RMSprop(lr=1e-4),
                     metrics=['accuracy'])
model.summary()

#selecting image for kfold
k=20
num_val=480//k
num_image=485
nkfold=50
for nk in range(nkfold):
    trainID,valID = train_test_split(range(num_image),test_size=0.1, )

    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.mkdir(train_dir)
    if os.path.exists(validation_dir):
        shutil.rmtree(validation_dir)
    os.mkdir(validation_dir)

    train_AE_dir = os.path.join(train_dir, 'AE')
    train_BE_dir = os.path.join(train_dir, 'BE')
    if os.path.exists(train_AE_dir):
        shutil.rmtree(train_AE_dir)
    os.mkdir(train_AE_dir)
    if os.path.exists(train_BE_dir):
        shutil.rmtree(train_BE_dir)
    os.mkdir(train_BE_dir)

    validation_AE_dir = os.path.join(validation_dir, 'AE')
    validation_BE_dir = os.path.join(validation_dir, 'BE')
    if os.path.exists(validation_AE_dir):
        shutil.rmtree(validation_AE_dir)
    os.mkdir(validation_AE_dir)
    if os.path.exists(validation_BE_dir):
        shutil.rmtree(validation_BE_dir)
    os.mkdir(validation_BE_dir)

    fnames = ['AE.{}.png'.format(i) for i in trainID]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_AE_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['AE.{}.png'.format(i) for i in valID]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_AE_dir, fname)
        shutil.copyfile(src, dst)
    fnames = ['BE.{}.png'.format(i) for i in trainID]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_BE_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['BE.{}.png'.format(i) for i in valID]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_BE_dir, fname)
        shutil.copyfile(src, dst)

    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function = tf.keras.applications.efficientnet.preprocess_input,
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = 'nearest', 
    )
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                       target_size = (height, width),
                                                       batch_size = batch_size,
                                                       seed = 7,
                                                       shuffle = True,
                                                       class_mode = "binary")
    valid_datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function = tf.keras.applications.efficientnet.preprocess_input)
    valid_generator = valid_datagen.flow_from_directory(validation_dir,
                                                        target_size = (height, width),
                                                        batch_size = batch_size,
                                                        seed = 7,
                                                        shuffle = False,
                                                        class_mode = "binary")
    train_num = train_generator.samples
    valid_num = valid_generator.samples
    print(train_num,valid_num)
    model = keras.models.Sequential([
        EfficientNetB7,
        keras.layers.Dense(num_classes, activation = 'sigmoid'),
    ])
    model.compile(loss="binary_crossentropy",
                         #optimizer="sgd", 
                         optimizer="adam", 
                         #optimizer=optimizers.RMSprop(lr=1e-4),
                         metrics=['accuracy'])
    epochs = 25
    history = model.fit_generator(train_generator,
                                               steps_per_epoch = train_num // batch_size,
                                               epochs = epochs,
                                               validation_data = valid_generator,
                                               validation_steps = valid_num // batch_size )
    
    plt.figure(figsize=(8,6))
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, label='Training acc', linewidth=6)
    plt.plot(epochs, smooth_curve(val_acc), label='Validation acc', linewidth=6)
    plt.title('Training and validation accuracy', fontsize=20)
    plt.ylabel('Accuracy', fontsize=24)
    plt.xlabel('Epoch', fontsize=24)
    plt.tick_params(labelsize=24)
    plt.grid(c='b', linestyle='--', linewidth=1.5) 
    plt.legend(fontsize=20)
    plt.tight_layout()
    
    plt.savefig('acc_classify_test_EfficientNetB7_'+str(nk)+'.png', dpi=300)
    
    plt.figure(figsize=(8,6))
    plt.plot(epochs, loss,  label='Training loss', linewidth=6)
    plt.plot(epochs, smooth_curve(val_loss), label='Validation loss', linewidth=6)
    plt.title('Training and validation loss', fontsize=20)
    plt.ylabel('Loss', fontsize=24)
    plt.xlabel('Epoch', fontsize=24)
    plt.tick_params(labelsize=24)
    plt.grid(c='b', linestyle='--', linewidth=1.5) 
    plt.legend(fontsize=20)
    plt.tight_layout()
    
    plt.savefig('loss_classify_test_EfficientNetB7_'+str(nk)+'.png', dpi=300)
        
    
    
    test_datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function = tf.keras.applications.efficientnet.preprocess_input)
    
    test_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size = (height, width),
                                                        batch_size = batch_size,
                                                        seed = 7,
                                                        shuffle = False,
                                                        class_mode = "binary")
    #test_x,label_y = test_generator.next()
    #print("label_y")
    #print(label_y)
    #predictions = model.predict(test_x)
    
    test_num = test_generator.samples
    label_y = test_generator.classes
    #print("class_labels",label_y)
    print(test_num, len(label_y))
    predictions = model.predict_generator(test_generator, steps = test_num // batch_size +1)
    print(len(predictions))
    #print("y_pred:",predictions)
    #y_pred = np.argmax(predictions, axis=1)
    y_pred = predictions.round()
    print(y_pred)
    
    from sklearn.metrics import confusion_matrix,classification_report
    print(confusion_matrix(label_y, y_pred))
    
    report = classification_report(label_y, y_pred)
    print("report:")
    print(report)
    
    plt.figure() 
    plot_confuse(label_y, y_pred)
    
    confuse_png = "confusion_matrix_EfficientNetB7_"+str(nk)+".png"
    os.rename("confusion_matrix.png", confuse_png)
    
    report0 = classification_report(label_y, y_pred, output_dict=True)
    df0 = pd.DataFrame(report0).transpose()
    resultcsv="confusion_result_EfficientNetB7_"+str(nk)+".csv"
    df0.to_csv(resultcsv, index= True)
 
    label_pred_csv="label_pred_EfficientNetB7_"+str(nk)+".csv"
    df=pd.DataFrame(zip(label_y, y_pred), columns=['test','pred'])
    df.to_csv(label_pred_csv)

    file = open('./history_resnet_'+str(nk)+'.pkl', 'wb')
    pickle.dump(history.history, file)
    file.close()
    del model
