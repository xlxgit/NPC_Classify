#-*-coding = utf-8 -*-
import os,shutil
import tensorflow as tf
from keras import backend as K
import numpy as np
from sklearn.model_selection import train_test_split
import glob
from fnmatch import fnmatchcase as match
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import glob
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support 
import pandas as pd
from keras import callbacks
import pickle
from keras import optimizers
import tensorflow as tf
from tensorflow.keras import optimizers


#set environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

num_cores = 4
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_cores,inter_op_parallelism_threads=num_cores,allow_soft_placement=True, device_count={'CPU': 4})

config.allow_soft_placement=True
config.gpu_options.per_process_gpu_memory_fraction=0.7
config.gpu_options.allow_growth = True

original_dataset_dir = '../data/ABE_2022'

from keras import layers
from keras import models
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
#model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),metrics=['acc'])
model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(lr=0.00020515945271434115),metrics=['acc'])
print(model.summary())


from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
          previous = smoothed_points[-1]
          smoothed_points.append(previous * factor + point * (1 - factor))
        else:
          smoothed_points.append(point)
    return smoothed_points

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

def eval_model(y_true, y_pred, labels):
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)
    tot_p = np.average(p, weights=s)
    tot_r = np.average(r, weights=s)
    tot_f1 = np.average(f1, weights=s)
    tot_s = np.sum(s)
    res1 = pd.DataFrame({
        u'Label': labels,
        u'Precision': p,
        u'Recall': r,
        u'F1': f1,
        u'Support': s
    })
    res2 = pd.DataFrame({
        u'Label': [u'总体'],
        u'Precision': [tot_p],
        u'Recall': [tot_r],
        u'F1': [tot_f1],
        u'Support': [tot_s]
    })
    res2.index = [999]
    res = pd.concat([res1, res2]) 

    conf_mat = pd.DataFrame(confusion_matrix(y_true, y_pred), columns=labels, index=labels)
    return conf_mat, res[[u'Label', u'Precision', u'Recall', u'F1', u'Support']]

def read_img(path):
    cate =[os.path.join(path,k) for k in os.listdir(path)]
    # ['./AE_and_BE_kfold/train/AE','./AE_and_BE_kfold/train/BE']
    print(cate)
    imgs=[]
    labels=[]
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder+'/*'):
            #print('reading the images:%s'%(im))
            img=cv2.imread(im, cv2.IMREAD_COLOR)
            imgs.append(img)
            labels.append(idx)
    return imgs, labels

#tranform image for random forest
def transform(img):
    hist = cv2.calcHist([img], [0,1,2], None, [64]*3, [0,256]*3)
    return hist.ravel()

base_dir = './AE_and_BE_kfold'
num_patient=20

if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
os.mkdir(base_dir)

nkfold=50

for nk in range(nkfold):
    #train_A_idx,test_A_idx = train_test_split(range(num_patient),test_size=0.1, random_state = 12345)
    train_A_idx,test_A_idx = train_test_split(range(num_patient),test_size=0.1, )
    print(test_A_idx)
    trainID_A=[]
    for i in train_A_idx:
        list0=str(i).rjust(2,'0')
        trainID_A+=[j for j in glob.glob(r'../data/ABE_2022/A0*'+list0+'_01_*png')]
    
    testID_A=[]
    for i in test_A_idx:
        list1=str(i).rjust(2,'0')
        testID_A+=[j for j in glob.glob(r'../data/ABE_2022/A0*'+list1+'_01_*png')]
    
    #train_B_idx,test_B_idx = train_test_split(range(num_patient),test_size=0.1, random_state = 123456)
    train_B_idx,test_B_idx = train_test_split(range(num_patient),test_size=0.1, )
    print(test_B_idx)
    trainID_B=[]
    for i in train_B_idx:
        list0=str(i).rjust(2,'0')
        trainID_B+=[j for j in glob.glob(r'../data/ABE_2022/B0*'+list0+'_01_*png')]
    
    testID_B=[]
    for i in test_B_idx:
        list1=str(i).rjust(2,'0')
        testID_B+=[j for j in glob.glob(r'../data/ABE_2022/B0*'+list1+'_01_*png')]
    
    print(testID_A,testID_B)
    
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.mkdir(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.mkdir(test_dir)
    
    train_AE_dir = os.path.join(train_dir, 'AE')
    os.mkdir(train_AE_dir)
    
    train_BE_dir = os.path.join(train_dir, 'BE')
    os.mkdir(train_BE_dir)
    
    
    test_AE_dir = os.path.join(test_dir, 'AE')
    os.mkdir(test_AE_dir)
    
    test_BE_dir = os.path.join(test_dir, 'BE')
    os.mkdir(test_BE_dir)
    
    
    fnames = [i for i in trainID_A]
    for fname in fnames:
        src = os.path.join(fname)
        dst = os.path.join(train_AE_dir)
        shutil.copy(src, dst)
    
    
    fnames = [i for i in testID_A]
    for fname in fnames:
        src = os.path.join(fname)
        dst = os.path.join(test_AE_dir)
        shutil.copy(src, dst)
    
    fnames = [i for i in trainID_B]
    for fname in fnames:
        src = os.path.join(fname)
        dst = os.path.join(train_BE_dir)
        shutil.copy(src, dst)
    
    
    fnames = [i for i in testID_B]
    for fname in fnames:
        src = os.path.join(fname)
        dst = os.path.join(test_BE_dir)
        shutil.copy(src, dst)
    
    train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,)
    
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    batch_size = 24

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True)
    
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)
    
    # steps_per_epoch*batch= trainSize
    # early stopping
    best_weights_filepath = './best_weights_CNN_'+str(nk)+'.hdf5'
    earlyStopping=callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
    saveBestModel = callbacks.ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    num_of_test_samples = len(testID_A) + len(testID_B)
    print(num_of_test_samples // batch_size)
    num_of_train_samples = train_generator.samples

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    #model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),metrics=['acc'])
    model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(lr=0.00020515945271434115),metrics=['acc'])
    history = model.fit_generator(train_generator, steps_per_epoch=num_of_train_samples // batch_size, epochs=25,
        validation_data=test_generator,validation_steps=num_of_test_samples // batch_size,
        callbacks=[earlyStopping, saveBestModel]
        )
    
    modelname='AT2_and_BT2_kfold_CNN_'+str(nk)+'.h5'
    model.save(modelname)

    
    plt.figure(figsize=(8,6))
    acc = history.history['acc']
    print(acc)
    val_acc = history.history['val_acc']
    print(val_acc)
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
    plt.savefig('acc_classify_test_CNN_'+str(nk)+'.png', dpi=300)
    
    plt.figure(figsize=(8,6))
    plt.plot(epochs, loss,  label='Training loss', linewidth=6)
    plt.plot(epochs, smooth_curve(val_loss), label='Validation loss', linewidth=6)
    plt.title('Training and validation loss', fontsize=20)
    plt.ylabel('Loss', fontsize=24)
    plt.xlabel('Epoch', fontsize=24)
    plt.tick_params(labelsize=24)
    plt.legend(fontsize=20)
    plt.grid(c='b', linestyle='--', linewidth=1.5) 
    plt.tight_layout()
    plt.savefig('loss_classify_test_CNN_'+str(nk)+'.png', dpi=300)
    #plt.show()
    
    num_of_test_samples = len(testID_A) + len(testID_B)
    print(num_of_test_samples // batch_size)
    predictions = model.predict_generator(test_generator,  num_of_test_samples // batch_size + 1)
    
    #print("y_pred:",predictions)
    y_pred=predictions.round()
    test_loss, test_acc = model.evaluate_generator(test_generator, steps=6)
    print('test acc:', test_acc)
    
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    print("class_labels",class_labels)
    
    print(confusion_matrix(test_generator.classes, predictions.round()))
    #print("y_pred:",y_pred)
    #print("y_test:",test_generator.classes)
    
    report = classification_report(true_classes, predictions.round(), target_names=class_labels)
    print("CNN report:")
    print(report)
    
    conf_mat, evalues = eval_model(true_classes, y_pred, class_labels)
    print(conf_mat)                         
    print("CNN eval report:")
    print(evalues)
    
    plt.figure() 
    plot_confuse(true_classes, y_pred)
    
    confuse_png = "confusion_matrix_CNN_"+str(nk)+".png"
    os.rename("confusion_matrix.png", confuse_png)
    
    report0 = classification_report(true_classes, y_pred, output_dict=True)
    df0 = pd.DataFrame(report0).transpose()
    resultcsv="confusion_result_CNN_"+str(nk)+".csv"
    df0.to_csv(resultcsv, index= True)
    label_pred_csv="label_pred_CNN_"+str(nk)+".csv" 
    df=pd.DataFrame(zip(true_classes, y_pred), columns=['test','pred'])
    df.to_csv(label_pred_csv) 
    file = open('./history_cnn_'+str(nk)+'.pkl', 'wb')
    pickle.dump(history.history, file)
    file.close()

    del model

    #for random forest
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    
    data_names=[os.path.join(train_dir,k) for k in os.listdir(train_dir)]
    print(data_names)
    
    images,labels = read_img(train_dir)
    print(type(images))
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    print(y)
    
    x = np.row_stack([transform(img) for img in images])
    print(x.shape)
    
    train_x = x[:,:]
    train_y = y
    
    test_images,test_labels = read_img(test_dir)
    test_y = label_encoder.fit_transform(test_labels)
    test_x = np.row_stack([transform(img) for img in test_images])
    
    
    model_rgb_rf = RandomForestClassifier(n_estimators =63, min_samples_split=2, 
                                          max_features=0.15474941416147267, max_depth =25, random_state=None)
    model_rgb_rf.fit(train_x,train_y) 
    
    y_pred = model_rgb_rf.predict(test_x)
    
    conf_mat, evalues = eval_model(test_y, y_pred, label_encoder.classes_)
    print(conf_mat)
    print("RandomForest report:")
    print(evalues)
    scores=model_rgb_rf.score(test_x,test_y)
    print(scores)
    
    
    print(confusion_matrix(test_y, y_pred))
    
    report = classification_report(test_y, y_pred, )
    print(report)
    
    plot_confuse(test_y, y_pred)
    
    confuse_png = "confusion_matrix_RF_"+str(nk)+".png"
    os.rename("confusion_matrix.png", confuse_png)
    
    report = classification_report(test_y, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    resultcsv="confusion_result_RF_"+str(nk)+".csv"
    df.to_csv(resultcsv, index= True)
    label_pred_csv="label_pred_RF_"+str(nk)+".csv" 
    df=pd.DataFrame(zip(test_y, y_pred), columns=['test','pred'])
    df.to_csv(label_pred_csv) 

