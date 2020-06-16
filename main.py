# -*- coding: utf-8 -*-
import os,sys
import time
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import keras
from keras.models import Model, Sequential, load_model
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv1D,SeparableConv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, LSTM, Dense, GRU
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import pickle
import h5py
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

name = 'xception'

#%%
#fr = h5py.File('/media/lab1/D/zbw/radio_5classes_version1.h5','r')   #打开h5文件  
fr = h5py.File('/media/lab1/D/zbw/alldata/data/radio_5classes_little_version3.h5','r')   #打开h5文件  
labels = list(fr.keys())
#with open("/media/lab1/D/zbw/rawradio_length5401_nocutxinxidiantai_start10485761.h5", 'rb') as xd1:  # 这段执行对原始数据进行切片的任务，可在spyder下运行，查看变量
#    Xd = pickle.load(xd1)  # , encoding='latin1'
#labels=Xd.keys()
#snrs = map(lambda j: sorted((set((lambda x: x[j], Xd.keys()))), [1, 0]))
X = []
lbl = []
for label in labels:
#for mod in mods:xcvbnm.
    #for snr in snrs:
    X.append(fr[(label)])
#        for i in range(Xd[(mod, snr)].shape[0]):  lbl.append((mod, snr))
    for i in range(fr[(label)].shape[0]):  lbl.append(label)
X = np.vstack(X)
fr.close()
#%%
#pca = decomposition.PCA(n_components=1000, copy=True, whiten=False) 
#for i in range(X.shape[0]):  
#    newData=pca.fit_transform(X[i,:,:]) 
#%%
np.random.seed(2016)  # 对预处理好的数据进行打包，制作成投入网络训练的格式，并进行one-hot编码
n_examples = X.shape[0]
n_train = n_examples * 0.8  # 对半
train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
test_idx = list(set(range(0, n_examples)) - set(train_idx))  # label
b = np.array(test_idx)
a = len(test_idx)
test_idx1 = np.random.choice(b, size=int(a*0.5), replace=False)
val_idx = list(set(range(0, n_examples)) - set(train_idx)- set(test_idx1))  # label
X_train = X[train_idx]
X_test = X[test_idx1]
X_val = X[val_idx]

def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1

#%%
trainy = list(map(lambda x: labels.index(lbl[x]), train_idx))
Y_train = to_onehot(trainy)
Y_test = to_onehot(list(map(lambda x: labels.index(lbl[x]), test_idx1)))
Y_val = to_onehot(list(map(lambda x: labels.index(lbl[x]), val_idx)))
    
    #in_shp = list(X_train.shape[1:])
in_shp = list(X_train.shape[1:])
print (X_train.shape, in_shp)
classes = labels
# %%
def GroupNorm(x,gama,beta,G,eps=1e-5):
    N,C,H,W=x.shape
    x =tf.reshape(x,[N,G,C // G,H,W])
    mean,var=tf.nn.moments(x,[2,3,4],keep.dims=True)
    x=(x-mean)/tf.sqrt(var+eps)
    x=tf.reshape(x,[N,C,H,W])
    return x*gamma+beta
# %%
from keras import layers
def Sconv_block(input_tensor, kernel_size, filters):
    filters1, filters2 = filters
    kernel_size1,kernel_size2 = kernel_size

    x = SeparableConv2D(filters1, kernel_size1, padding='same',use_bias=False)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(filters2, kernel_size2, padding='same',use_bias=False)(x)
    x = BatchNormalization()(x)
 
#	residual = Conv2D(filters3, kernel_size3, strides=(2, 2), padding='same', use_bias=False)(input_tensor)
#	residual = BatchNormalization()(residual)
#    
#    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
#	x = layers.add([x, residual])
    
#    x = Conv2D(filters1, kernel_size1,padding='same' )(input_tensor)
#    x = BatchNormalization()(x)
#    x = Activation('relu')(x)
#
#    x = Conv2D(filters2, kernel_size2,padding='same' )(x)
#    x = BatchNormalization()(x)
#    x = Activation('relu')(x)
#
#    x = Conv2D(filters3, kernel_size3,padding='same')(x)
#    x = BatchNormalization()(x)

#    x = layers.add([x, input_tensor])
#    x = Activation('relu')(x)
    return x

#%%
from keras import layers
# 这里使用keras的函数式编程 http://keras-cn.readthedocs.io/en/latest/getting_started/functional_API/
input_x_padding = Input(shape=(1,2,5401))
#input_x_padding=embedding()


layer1 = Sconv_block(input_x_padding, [(2,3),(2,3)],[8,8])
layer2 = Sconv_block(input_x_padding, [(2,3),(2,3)],[8,8])
layer3 = Sconv_block(input_x_padding, [(2,3),(2,3)],[8,8])
layer4 = Sconv_block(input_x_padding, [(2,3),(2,3)],[8,8])
layer5 = Sconv_block(input_x_padding, [(2,3),(2,3)],[8,8])
layer6 = Sconv_block(input_x_padding, [(2,3),(2,3)],[8,8])
residual1 = Conv2D(8, (1,1), strides=(1, 1), padding='same', use_bias=False)(input_x_padding)
residual1 = BatchNormalization()(residual1)        
x = layers.add([layer1,layer2,layer3,layer4,layer5,layer6,residual1])

layer_add3 = Activation('relu')(x)
#layer_add4 = layers.add([layer_add3, input_x_padding])
#layer_add4 = Activation('relu')(layer_add4)

#layer1=Reshape((50000,32,1))(layer1)
#layer_add2 = AveragePooling2D(pool_size=(2,1))(layer_add2)
##layer1=Reshape((16,50000,1))(layer1)
#layer4 = conv_block(layer_add2, [(1,1),(2,5),(1,1)],[32,32,2])
#layer5 = conv_block(layer_add2, [(1,1),(2,5),(1,1)],[32,32,2])
#layer6 = conv_block(layer_add2, [(1,1),(2,5),(1,1)],[32,32,2])
#layer_add3 = layers.add([layer4, layer5,layer6])
#layer_add3 = Activation('relu')(layer_add3)
#layer_add4 = layers.add([layer_add3, layer_add2])
#layer_add4 = Activation('relu')(layer_add4)

concat_size = list(np.shape(layer_add3))
input_dim = int(concat_size[-1] * concat_size[-2])
timesteps = int(concat_size[-3])
# concat = np.reshape(concat, (-1,timesteps,input_dim))

input_x = Reshape((timesteps, input_dim))(layer_add3)
# 形如（samples，timesteps，input_dim）的3D张量

#
#gru_out = GRU(64,input_dim=input_dim, input_length=3, activation='relu',return_sequences=True, dropout=0.2,recurrent_dropout=0.0)(input_x)
#layerBN1 = BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(gru_out)

#gru_out2 = GRU(64,input_dim=input_dim, input_length=3,activation='relu',return_sequences=True, dropout=0.2,recurrent_dropout=0.0)(layerBN1)
#layerBN1 = BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(gru_out2)
#max_pool=(MaxPooling2D(pool_size=(4,4)))(layerBN1)
#avg_pool = AveragePooling1D(pool_size=(4,4))(layerBN1)
#layerBN1 = concatenate([avg_pool, max_pool])
#gru_out3 = GRU(128,input_dim=input_dim, input_length=3,activation='relu',return_sequences=True, dropout=0.2,recurrent_dropout=0.0)(layerBN1)
#layerBN1 = BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(gru_out3)
#max_pool=(MaxPooling2D(pool_size=(4,4)))(layerBN1)
#avg_pool = AveragePooling1D(pool_size=(4,4))(layerBN1)
#layerBN1 = concatenate([avg_pool, max_pool])


# 当 输出为250的时候正确里更高
# lstm_out = LSTM(250, input_dim=input_dim, input_length=timesteps)(concat)
#out = TimeDistributed(Dense(128))(gru_out3)
layer_Flatten = Flatten()(input_x)
#layer_dense1 = Dense(128, activation='relu', init='he_normal', name="dense1")(layer_Flatten)
#layer_dropout = Dropout(dr)(layer_dense1)
#layer_BN1 = BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(layer_dense1)
layer_dense2 = Dense(len(classes), init='he_normal', name="dense2")(layer_Flatten)
layer_BN2 = BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(layer_dense2)
layer_softmax = Activation('softmax')(layer_BN2)
output = Reshape([len(classes)])(layer_softmax)
model = Model(inputs=input_x_padding, outputs=output)
#model.load_weights('convmodrecnets_gru85_2.wts.h5', by_name=True)
model.compile(loss='categorical_crossentropy', optimizer='nadam',metrics=['accuracy'])
model.summary()

# %%
# Set up some params
epochs = 30 # number of epochs to train on
batch_size = 16  # training batch size default1024
#%%
#X_train=np.transpose(X_train)
#Y_train=np.transpose(Y_train)
#X_test=np.transpose(X_test)
#Y_test=np.transpose(Y_test)l
#X_train=np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1],X_train.shape[2]))
##Y_train=np.reshape(Y_train, (Y_train.shape[0], Y_train.shape[1]))
#X_test=np.reshape(X_test, (X_test.shape[0],1,X_test.shape[1],X_test.shape[2]))
##Y_test=np.reshape(Y_test, (Y_test.shape[0], Y_test.shape[1],))l

filepath = "/media/lab1/D/zbw/alldata/newdata_model/result/5class-xception/5class_xception_version3.wts.h5"  # 所要保存的文件名字，h5格式，不用写路径，默认在程序执行的文件夹内
model.load_weights(filepath)
#history = model.fit(X_train,
#                    Y_train,
#                    batch_size=batch_size,
#                    epochs=epochs,
##                    xcvbnm.BN_BN_
#                    
#                    verbose=2,
#                    validation_data=(X_test, Y_test),
#                    callbacks=[  # 回调函数，https://keras-cn.readthedocs.io/en/latest/other/callbacks/
#                        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,
#                                                        mode='auto'),
#                        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
#                    ])  # EarlyStopping 当监测值不再改善时，该回调函数将中止训练，如去除本行将执行所有epoch
# Show loss curves
#%%
#filepath = "/media/lab1/D/zbw/alldata/newdata_model/result/only-xception/newdata-5401-150_xception6.wts.h5"
#X_val=np.reshape(X_val, (X_val.shape[0],X_val.shape[1], 1))
#X_train=np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1],X_train.shape[2]))
X_val=np.reshape(X_val, (X_val.shape[0],1,X_val.shape[1],X_val.shape[2]))
#import time
#plt.figure()
#plt.title('Training performance')
#plt.plot(history.epoch, history.history['loss'], label='train loss+error')
#plt.plot(history.epoch, history.history['val_loss'], label='val_error')
#plt.legend()
#plt.savefig('Training performance_12')
model.load_weights(filepath)
#t0 = time.time()
score = model.evaluate(X_val, Y_val, verbose=0, batch_size=batch_size)
#t1 = time.time() 
#t = (t1-t0)/len(val_idx)
#print('time:',t)
print('evaluate_score:', score)
#%%
def plot_confusion_matrix(y_true, y_pred, labels):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    cmap = plt.cm.binary
    cm = confusion_matrix(y_true, y_pred)
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    intFlag = 0 # 标记在图片中对文字是整数型还是浮点型
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        #

        if (intFlag):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=8, va='center', ha='center')

        else:
            c = cm_normalized[y_val][x_val]
            if (c > 0.01):
                #这里是绘制数字，可以对数字大小和颜色进行修改
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
            else:
                plt.text(x_val, y_val, "%d" % (0,), color='red', fontsize=7, va='center', ha='center')
    if(intFlag):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    else:
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title('')
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('Index of True Classes')
    plt.xlabel('Index of Predict Classes')
    plt.savefig('confusion_matrix.jpg', dpi=300)
    plt.show()

test_Y_hat = model.predict(X_val, batch_size=batch_size)
#%%

pre_labels = []
for x in test_Y_hat:
    tmp = np.argmax(x, 0)
    pre_labels.append(tmp)
true_labels = []
for x in Y_val:
    tmp = np.argmax(x, 0)
    true_labels.append(tmp)

oa = accuracy_score(true_labels, pre_labels)
kappa_oa = {}
print('oa_all:', oa)

#%%
conf = np.zeros([len(classes), len(classes)])
confnorm = np.zeros([len(classes), len(classes)])
for i in range(0, X_val.shape[0]):
    j = list(Y_val[i, :]).index(1)
    k = int(np.argmax(test_Y_hat[i, :]))
    conf[j, k] += 1
for i in range(0, len(classes)):
    confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
plot_confusion_matrix(true_labels,pre_labels,labels=classes)
#%%
acc_for_each_class = metrics.precision_score(true_labels, pre_labels, average=None)
average_accuracy = np.mean(acc_for_each_class)
print('aa:', average_accuracy)
#%%
kappa = cohen_kappa_score(pre_labels, true_labels)
print('kappa:', kappa)