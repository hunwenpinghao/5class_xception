# 5classes_xception
该项目是我们为54所做的一个电台信号分类的项目，它主要的功能是实现3个电台信号和2个话音信号的分类，实现环境是keras.
另外，我们还在NVIDIA的JTX-2上实现了部署。


# Description
该工程可以实现5类电台信号的分类，准确率达到了99.6%.
在JTX-2上进行了部署，运行时间比在TITAN-X上慢了进5倍，但是精度没有衰减，对于这种嵌入式设备来说，JTX-2的优势在于它很小巧，仅仅有一本书大小，功耗仅仅有7.5W.
另外，我们也对模型进行了压缩，使得权重仅仅有6.2M大小。


# environment
```
keras
scikit-learn
matplotlib
pickle
h5py
numpy 
```
版本的话，一般较高版本都可以兼容。
