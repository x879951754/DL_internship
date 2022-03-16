### 2022深度学习面经归纳 ###

---

先尝试做一个版本。

#### 图像处理基础 ####

---

**几个池化**

最大池化



平均池化



**几个滤波器**

中值滤波



均值滤波



高斯滤波



Sobel滤波器



#### 神经网络 ####

---

几个经典卷积神经网络模型，及各自的优缺点。这里先提一下卷积的计算公式：
$$
\hat{N} = \frac{N - F + 2*P}{S} + 1
$$
其中N是输入图像大小，F是卷积核大小，P是padding，S是stride，$\hat{N}$是输出图像大小。



**AlexNet**

![img](https://pic3.zhimg.com/80/v2-29c8b75b2cf5248f025fdf12a246801e_720w.jpg)

AlexNet网络一共8层，其中前5层是Conv，后3层是全连接层Fully Connected layer。

(1)ReLU。相比Sigmoid的优势是ReLU能够更快地收敛，因此其训练速度更快。根据Sigmoid函数的特点，其导数在稳定区会非常小，从而权重基本上不会再更新，容易导致梯度消失。

(2)Dropout。Dropout层以一定概率随机地关系当前层中的神经元激活值，不同的神经元组合被关闭，代表了一种不同的结构。所有这些不同的结构使用一个子数据集并行地带权重训练，权重总和为1。在预测时，相当于集成这些模型并取均值。这种结构化的模型正则化技术有利于避免过拟合。另外一个视点是：由于神经元是随机选取的，所以可以减少神经元之间的依赖，确保提取出相互独立的重要特征。

(3)Data Augmentation。这部分作者将每张图片处理为256x256的大小，但网络输入大小为224x224（paper中是224x224）。作者在256x256大小的图像上利用一个224x224的滑动窗口，将滑动窗口中的图像作为输入，这样就能扩大数据集（增加了图片的个数）。



**VGG**

VGG相比AlexNet的一个改进是，采用连续几个3x3大小的卷积核代替AlexNet中的大卷积核（11x11, 5x5）。

(1)去掉了AlexNet中的LRN层（Local Response Normalization，LRN提出活跃的神经元对它周边神经元的影响）。VGG的作者发现LRN的作用并不明显，干脆取消了。

(2)采用更小的3x3卷积核。对于给定的感受野，连续堆叠的小卷积核相比更大的卷积核能保证相同的感受野，但参数量更少。这里举个例子，比如给定感受野size是5x5，大卷积核size为5x5，两个连续的小卷积核size为3x3。这里设定padding为0，stride为1。

使用5x5卷积，输出的图像大小为1x1，参数量为5x5=25。
$$
\hat{N} = \frac{5 - 5 + 2 * 0}{1} + 1\\
\hat{N} = 1
$$
如果使用两个小的3x3卷积：
$$
\hat{N_1} = \frac{5 - 3 + 2 * 0}{1} + 1 \\
\hat{N_1} = 3 \\
\hat{N_2} = \frac{3 - 3 + 2 * 0}{1} + 1 \\
\hat{N_2} = 1
$$
小卷积核的参数总数为2x3x3=18。输出的图像的大小都为1x1，感受野同样是5x5，但是参数量减少了。



(3)池化核减小。AlexNet中的池化核大小为3x3，stride=2。而VGG中池化核大小为2x2，stride为2。

VGG正是利用多个连续的小卷积核替代单个大卷积核，成功地说明了网络结构还是深的效果好。



**GoogLeNet/Inception**

GoogLeNet设计了一种Inception模块结构。Inception是首次提出来的Block（类似后面的ResBlock），它将多个卷积或池化操作组装成一个模块，神经网络以模块为单位再去组装网络结构。

![深度学习|经典网络：GoogLeNet（一）](https://pic2.zhimg.com/v2-dd40daa484adc6bbd93ee1f5ba2277c1_1440w.jpg?source=172ae18b)

在以往的神经网络中，往往每一层只使用一种操作，比如卷积或者池化，并且卷积核大小固定。但实际情况中，对于同一张图片，不同尺度的卷积核的表现效果不同，因为它们的感受野不同（输出图像大小相同，对于不同的卷积核，反过来推感受野）。所以希望让网络自己去选择，Inception模块中并列提供多种卷积核操作，网络在训练过程中通过调节参数自动选择某条路径。这样做提升了网络的表达能力，同时也大大减少了参数量。



**ResNet**

深度网络退化问题：随着网络层数增加到一定程度后，网络的准确率出现饱和，甚至开始下降。ResNet增加了一个恒等映射，将深层网络后面的层作为恒等映射，模型就退化成了一个浅层网络。

ResNet中提出了一种残差结构，如下所示。如果把网络设计成$H(x)=F(x)+x$，则可以学习到残差$F(x)=H(x)-x$，当残差为0时，即学习到的特征为$H(x)=x$，此时堆积层仅仅做了恒等映射，至少网络性能不会下降。

![img](https://pic4.zhimg.com/80/v2-252e6d9979a2a91c2d3033b9b73eb69f_720w.jpg)

ResNet中的两种残差单元如下。一种是输入通道等于输出通道，残差路径特征和快捷路径特征直接相加。另一种是输入通道不等于输出通道，又因为残差路径和快捷路径要相加，因此通道数必须相同，故需要使用1x1卷积调整残差路径上特征的通道数。

![img](https://pic1.zhimg.com/80/v2-0892e5423616c30f69ded61111b111c0_720w.jpg)



**模型压缩与加速**

---

**MobileNet**

先说MobileNetV1。首先假设输入和输出大小相同，都是$D_F$，其中卷积大小为$D_K$，输入有M个通道，输出为N个通道。

普通卷积

<img src="C:\Users\87995\AppData\Roaming\Typora\typora-user-images\image-20220228220056395.png" alt="image-20220228220056395" style="zoom: 67%;" />

: params = $D_K^2 * M * N$，卷积核大小$D_K^2$，输入通道为M，输出通道为N。

: MultiAdd = $D_K^2 * M * N * D_F^2$，上面每次计算一个输出像素点，总共有$D_F^2$个像素点再作相加。

其中params表示参数量，MultiAdd表示乘加操作量，即计算量。

 

先了解一下组卷积，Depthwise Conv是特殊的组卷积

Group Conv

<img src="https://pic2.zhimg.com/80/v2-02ebd72671450190ed6d8a6ad5fd841d_720w.jpg" alt="img" style="zoom:50%;" />

将一个多通道的卷积核分为g（这里是2）组，每组分别对一部分输入特征图作卷积，完事后将两个结果按照通道维度concat。

因此，如果输入和输出相同都为(H, W)，输入通道为c1，输出通道为c2，卷积核大小为(h1, w1)，

如果按照原卷积的计算方法，计算总量为：
$$
h_1 * w_1 * c_1 * c_2 * H * W
$$
那么按照组卷积的计算方法，计算总量为：
$$
(h_1 * w_1 * \frac{c_1}{g} * \frac{c_2}{g} * H * W) * g \\ 
= \frac{h_1 * w_1 * c_1 * c_2 * H * W}{g}
$$


Depthwise Conv

<img src="https://pic3.zhimg.com/80/v2-795c9616bef033b495cbc1c43594aa46_720w.jpg" style="zoom:50%;" />

原论文中是将通道channel和宽高wh进行了分离。depthwise conv用来提取每个通道上的特征，而pointwise conv用来还原最终输出的通道数。

现在仍然对比一开始的普通卷积。普通卷积的计算量为$D_K^2 * M * N * D_F^2$。

下面是depthwise conv + pointwise conv，计算下面的计算量。

![image-20220228221848320](C:\Users\87995\AppData\Roaming\Typora\typora-user-images\image-20220228221848320.png)

depthwise conv的计算量为$D_K^2 * 1 * M * D_F^2$

pointwise conv的计算量为$1 * 1 * M * N * D_F^2$

将两者相加，总计算量为$(D_K^2 + N) * M * D_F^2$

与上面普通卷积作比，为
$$
\frac{Depthwise Conv + Pointwise Conv}{Conv} = \frac{(D_K^2 + N) * M * D_F^2}{D_K^2 * M * N * D_F^2} \\
= \frac{D_K^2 + N}{D_K^2 * N} \\
= \frac{1}{N} + \frac{1}{D_K^2}
$$
其中$N$是输出通道数，$D_K$是卷积核大小。可见计算量减少了不少。



<img src="https://pic3.zhimg.com/80/v2-02a868632a6d8b9d8f4dd923d269e3aa_720w.jpg" alt="img" style="zoom:50%;" />

如图，左边是普通卷积，右侧是Depthwise Conv + Pointwise Conv。



**MobileNetV2**

创新点：

（1）Inverted residuals。通常**Residual Block**（resnet50开始，每个resblock的确是1x1Conv -> 3x3Conv -> 1x1Conv）是先经过一个1x1Conv，把feature map的通道数“压下来”，再经过3x3Conv，最后经过一个1x1Conv，将feature map的通道数再“涨回去”。即先压缩，再扩张。

**而Inverted residuals相反**，先“扩张”，再“压缩”。

（2）Linear bottlenecks。为了避免ReLU对特征的破坏，在Residual Block的Eltwise sum（element wise层和Concat层差不多，Concat是通道拼接，而Eltwise层是 product sum max，乘 和 最大 三种操作）之前的那个1x1Conv不再采用ReLU。



<img src="https://img-blog.csdn.net/20180123092023369?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMTk5NTcxOQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast" style="zoom: 50%;" />

如图：

Depth-wise convolution之前多了一个1x1的“扩张”层，目的是为了提升通道数，获得更多特征；

最后不采用Relu，而是Linear，目的是防止Relu破坏特征。



再看看MobileNetV2的block与ResNet的block： 

<img src="https://img-blog.csdn.net/20180123092059581?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMTk5NTcxOQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast" style="zoom:50%;" />

ResNet Block每次通道数减小为原来的1/4，最后再膨胀为某个通道数。

MobileNetV2 Block每次通道数变为之前的6倍，最后一个1x1Conv再压缩成某个通道数。



**Inverted Residual和ReLU6替换为Linear的原因**

（1）DW卷积提取的特征受到通道数的限制，如果通道数太少，那么DW提取到的特征太少。因此Inverted Residual就是为了一开始将能够提取到的特征增多（论文中是将通道数提升为6倍），这样DW卷积能够提取更多的特征。

（2）ReLU会破坏特征。ReLU激活函数对于负的输入，全部映射为0，而特征在最后输出的时候已经被“压缩”，再经过ReLU，又要损失一部分特征，因此最后一层采用Linear bottleneck。实验结果表明这是正确的。



MobileNetV2改进 https://blog.csdn.net/u011995719/article/details/79135818

Group Conv https://zhuanlan.zhihu.com/p/226448051

详细 https://zhuanlan.zhihu.com/p/45209964



**ShuffleNet**





csdn：https://blog.csdn.net/github_39611196/article/details/89342292



#### Loss函数 ####

---

| 损失函数 | 公式 | 函数图像 | 反向传播 |
| -------- | ---- | -------- | -------- |

**0-1损失函数**

该损失函数预测值与标签相同为0，不同为1。
$$
L(Y,f(x)) = \left\{ \begin{array}{c}
	0, & Y == f(x) \\
	1, & Y != f(X)
\end{array} \right.
$$
(1)0-1损失函数直接对应分类判断错误的个数，但是它是一个非凸函数，不太适用。

(2)感知机就是用的这种损失函数，但是相等这个条件太过苛刻。放宽条件，即满足$|Y-f(X)| < \sigma$时认为相等。
$$
L(Y,f(X)) = \left\{ \begin{array}{c}
	0, & |Y-f(x)| < \sigma \\
	1, & |Y-f(x)| >= \sigma
\end{array} \right.
$$

**绝对值损失函数（L1 loss）**
$$
L(Y,f(X)) = |Y - f(X)|
$$
代表函数：平均绝对误差（Mean Absolute Error）
$$
L = \frac{1}{n} \sum^{n}_{i=1} |y_i - \hat{y_i}|
$$
主要应用于**线性回归**模型的损失函数。



**平方损失函数（L2 loss）**
$$
L(Y,f(X)) = (Y-f(X))^2
$$
代表：均方误差（Mean Square Error）
$$
L = \frac{1}{n} \sum^{n}_{i=1} (y_i - \hat{y_i})^2
$$
主要应用于**线性回归**问题。



**smooth L1 loss**

smooth L1 loss由微软rgb大神提出，Fast RCNN论文提出该方法。

L1损失函数以及求导
$$
L_1 = |x| = \left\{ \begin{array}{c} 
-x, & x < 0 \\
x, & x >= 0
\end{array} \right. \\

L_1^{'} = \left\{ \begin{array}{c} 
-1, & x < 0 \\
1, & x >= 0
\end{array} \right.
$$
L2损失函数以及求导
$$
L_2 = x^2 \\
L_2^{'} = 2 x
$$
smooth L1 loss的函数和导数如下
$$
smooth \ \ L_1 = \left\{ \begin{array}{c} 
0.5x^2, & |x| < 1 \\
|x| - 0.5, & otherwise
\end{array} \right. \\

smooth \ \ L_1^{'} = \left\{ \begin{array}{c} 
-1, & x < -1 \\
-x, & -1 <= x < 0 \\
x, & 0 <= x < 1 \\
1, & 1 <= x
\end{array} \right.
$$
<img src="https://img-blog.csdnimg.cn/20210706174030561.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoZW4xMjM0NTIwbm5u,size_16,color_FFFFFF,t_70" style="zoom:50%;" />

三者比较：

（1）L1 loss求导之后导数为常数。在训练后期，x很小时，如果learning rate不变，损失函数会在稳定值（局部最小值）附近波动，很难收敛到更高的精度；而且可以从图中看出，在x=0时不可微；

（2）L2 loss求导之后为-x或者x，在x很大的时候导数值也非常大，因此在训练刚开始时十分不稳定；

（3）smooth L1 loss取二者精华，去其糟粕。

目标检测中，检测框回归的损失为
$$
L_{loc}(t, t^{gt}) = \sum_{i \in \{x,y,w,h\}} smooth L_1(t_i - t_i^{gt})
$$
$t$表示检测框和真实框的偏移值。



**对数损失函数**
$$
L(Y,P(Y|X)) = -logP(Y|X)
$$
(1)对数损失函数能非常好的表征概率分布，在很多场景尤其是多分类，计算每个类别的概率。

(2)健壮性不强，相比hinge loss，它对噪声更敏感。

(3)对数损失函数应用于**逻辑回归**。



代表函数：交叉熵（Cross Entropy）

$$
Loss_{CE} = - \frac{1}{n} \sum_{x} [y * ln(\hat{y}) + (1 - y) * ln(1 - \hat{y})]
$$
其中x为某个样本，y表示对应标签，$\hat{y}$表示预测值，n表示样本总数。

（1）本质上是一种对数似然函数，可用于二分类和多分类问题中。

二分类问题中的cross entropy（最后的全连接层通过sigmoid输出）。正样本标签为1，负样本为0。
$$
Loss_{BCE} = - \frac{1}{n} \sum_{x} [y * ln(\hat{y}) + (1 - y) * ln(1 - \hat{y})]
$$
多分类问题中的cross entropy（最后的全连接层通过softmax输出）
$$
Loss_{CE} = - \frac{1}{n} \sum_{x} y * ln(\hat{y})
$$
（2）当使用sigmoid作为激活函数的时候，常用交叉熵损失函数，而不用均方差损失函数。因为交叉熵可以完美解决均方差权重更新过慢的问题，具有“误差大的时候，权重更新快；误差小的时候，权重更新慢”的良好性质。



---

![img](https://pic4.zhimg.com/80/v2-ac627eab5f07ead5144cfaaff7f2163b_720w.jpg)

---



**指数损失函数**
$$
L(Y|f(X)) = exp(-y f(X))
$$
对离群点、噪声非常敏感。经常用在AdaBoost算法中。n个样本情况如下：
$$
L(y, f(x)) = \frac{1}{n} \sum^{n}_{i=1} exp(-y_i f(x_i))
$$

**Hinge Loss**
$$
L(y_i, f(x_i)) = max(0, 1 - y_i f(x_i))
$$
（1）hinge loss损失函数表示如果被分类正确，损失为0，否则损失就为$1 - yf(x)$。SVM就是使用这个损失函数。

（2）$f(x)$为预测值，在-1到1之间。$y$是标签，为-1或1。$f(x)$的值在-1到1之间就可以了，并不鼓励$|f(x)| > 1$，即不鼓励分类器过度自信，某个正确分类样本距离分割线超过1并不会有任何奖励，从而使分类器更专注与整体的误差。

（3）健壮性相对较高，对异常点、噪声不敏感。但没有太好的概率解释。



知乎：https://zhuanlan.zhihu.com/p/58883095



**Focal loss**

$$
FL(p_t) = - \alpha_t (1 - p_t)^\gamma log(p_t)
$$





#### 激活函数 ####

---

| 激活函数 | 公式 | 函数图像 | 反向传播 |
| -------- | ---- | -------- | -------- |

激活函数的特点是在神经网络中加入非线性因素。通常激活函数具有以下性质：

1.非线性：保证多层网络不退化成单层线性网络。

2.可微性：在梯度下降中，激活函数需要求导，因此必须可微。或者如ReLU，有限点处不可微，但可替代。

3.非饱和性：饱和性是指函数在某些区间的梯度接近或等于0，使得参数无法继续更新。比如sigmoid的正负无穷两端，ReLU小于0部分。

4.单调性：保证求导后，导数符号不变。但不是必要条件，比如Mish。



**ReLU**
$$
relu(x) = max(0, x)
$$

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.linspace(-5, 5, 200)  # x data (tensor), shape=(100, 1)
y_relu = relu(x)

plt.figure()
plt.grid()
plt.plot(x, y_relu, c='red', label='relu')
plt.ylim((-0.5, 5))
plt.legend(loc='best')
plt.show()
```

![image-20220216152431986](C:\Users\87995\AppData\Roaming\Typora\typora-user-images\image-20220216152431986.png)

relu的反向传播。输入向量为x，经过relu函数后得到向量y，前向传播（forward）得到的误差为e（error），求e对x的梯度。
$$
x = (x_1, x_2, ..., x_n) \\
y = ReLU(x) \\
e = forward(y) \\
$$
求解过程如下：

y对x求导
$$
\frac{d{y_i}}{d{x_i}} = \left \{ \begin{array}{c} 
	0, & x_i <= 0 \\
	1, & x_i > 0
\end{array} \right. \\

m = (\frac{\partial y_1}{\partial x_1}, \frac{\partial y_2}{\partial x_2}, ..., \frac{\partial y_n}{\partial x_n},)
$$
现在我们要计算误差e对x的导，由链式法则我们知道
$$
\frac{\partial e}{\partial x_i} = \frac{\partial e}{\partial y_i} \frac{\partial y_i}{\partial x_i}
$$
而$\frac{\partial y_i}{\partial x_i}$前面已经求出，故
$$
\frac{\partial e}{\partial x_i} = (\frac{\partial e}{\partial y_1} \frac{\partial y_1}{\partial x_1}, \frac{\partial e}{\partial y_2} \frac{\partial y_2}{\partial x_2}, ..., \frac{\partial e}{\partial y_n} \frac{\partial y_n}{\partial x_n}) \\
= (\frac{\partial e}{\partial y_1}, \frac{\partial e}{\partial y_2}, ..., \frac{\partial e}{\partial y_n}) \odot m
$$
m在前面已经求出（$\frac{\partial{y_i}}{\partial{x_i}}$为0或1），$\odot$表示对应元素相乘。



*后面的激活函数在误差反向传播过程中，都可以套用这个公式。稍微修改$\frac{\partial{e}}{\partial{y_i}}$和m即可*。



**Leaky ReLU**
$$
LReLU(x) = max(\alpha x, x), \ \alpha > 0
$$

```python
import numpy as np
import matplotlib.pyplot as plt

def leaky_relu(alpha, x):
    return np.maximum(alpha * x, x)

x = np.linspace(-5, 5, 200)  # x data (tensor), shape=(100, 1)
y_leaky_relu = leaky_relu(0.05, x)

plt.figure()
plt.grid()
plt.plot(x, y_leaky_relu, c='red', label='leaky relu')
plt.ylim((-0.5, 5))
plt.legend(loc='best')
plt.show()
```

![image-20220217224555903](C:\Users\87995\AppData\Roaming\Typora\typora-user-images\image-20220217224555903.png)



**ReLU6**
$$
ReLU6(x) = min(6, max(0, x))
$$

```python
import numpy as np
import matplotlib.pyplot as plt

def relu6(x):
    return np.minimum(np.maximum(0, x), 6)

x = np.linspace(-10, 10, 200)  # x data (tensor), shape=(100, 1)
y_relu6 = relu6(x)

plt.figure()
plt.grid()
plt.plot(x, y_relu6, c='red', label='relu6')
plt.ylim((-0.5, 8))
plt.legend(loc='best')
plt.show()
```

![image-20220217224716011](C:\Users\87995\AppData\Roaming\Typora\typora-user-images\image-20220217224716011.png)



**Sigmoid**
$$
sigmoid(x) = \frac{1}{1 + e^{-x}}
$$

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-5, 5, 200)  # x data (tensor), shape=(100, 1)
y_sigmoid = sigmoid(x)

plt.figure()
plt.grid()
plt.plot(x, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')
plt.show()
```

![image-20220217224821937](C:\Users\87995\AppData\Roaming\Typora\typora-user-images\image-20220217224821937.png)



**tanh**
$$
tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

```python
import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

x = np.linspace(-5, 5, 200)  # x data (tensor), shape=(100, 1)
y_tanh = tanh(x)

plt.figure()
plt.grid()
plt.plot(x, y_tanh, c='red', label='tanh')
plt.ylim((-1.5, 1.5))
plt.legend(loc='best')
plt.show()
```

![image-20220217224949568](C:\Users\87995\AppData\Roaming\Typora\typora-user-images\image-20220217224949568.png)



**Mish**

$$
Mish(x) = x * tanh(ln(1 + e^x))
$$

```python
import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def mish(x):
    return x * tanh(np.log(1 + np.exp(x)))

x = np.linspace(-5, 5, 200)  # x data (tensor), shape=(100, 1)
y_mish = mish(x)

plt.figure()
plt.grid()
plt.plot(x, y_mish, c='red', label='mish')
plt.ylim((-0.5, 5))
plt.legend(loc='best')
plt.show()
```

![image-20220217225053049](C:\Users\87995\AppData\Roaming\Typora\typora-user-images\image-20220217225053049.png)



**Swish**

$$
Swish(x) = x * sigmoid(x)
$$

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def swish(x):
    return x * sigmoid(x)

x = np.linspace(-5, 5, 200)  # x data (tensor), shape=(100, 1)
y_swish = swish(x)

plt.figure()
plt.grid()
plt.plot(x, y_swish, c='red', label='swish')
plt.ylim((-0.5, 5))
plt.legend(loc='best')
plt.show()
```

![image-20220217225136185](C:\Users\87995\AppData\Roaming\Typora\typora-user-images\image-20220217225136185.png)



**Hard-Swish**
$$
HSwish(x) = x * \frac{ReLU6(x + 3)}{6} \\
= x * \frac{min(6, max(0, x) + 3)}{6}
$$

```python
import numpy as np
import matplotlib.pyplot as plt

def relu6(x):
    return np.minimum(np.maximum(0, x), 6)

def hard_swish(x):
    return x * relu6(x + 3) / 6

x = np.linspace(-5, 5, 200)  # x data (tensor), shape=(100, 1)
y_hard_swish = hard_swish(x)

plt.figure()
plt.grid()
plt.plot(x, y_hard_swish, c='red', label='hard_swish')
plt.ylim((-0.5, 5))
plt.legend(loc='best')
plt.show()
```

![image-20220217225612994](C:\Users\87995\AppData\Roaming\Typora\typora-user-images\image-20220217225612994.png)



知乎：https://zhuanlan.zhihu.com/p/107434280



#### 优化器 ####

---

​		一般要求一个函数的最小值，我们会对这个函数求导，然后令$f'(x)=0$。但是临界点不一定是最值，可能是极值，甚至可能不是极值，比如鞍点。

​		梯度下降的定义：如果实值函数$f(x)$在点a处可微且有定义，那么函数$f(x)$在a点沿着梯度相反的方向$-\grad f(a)$下降最多。

​		从函数的局部极小值的**初始估计**出发$x_0$开始，并依次可得到如下序列$x_0, x_1, ...$使得$x_{n+1} \leftarrow x_n - \eta \grad f(x_n), n > 0$。因此可以得到$f(x_0) >= f(x_1) >= f(x_2) >= ...$，如果顺利的话序列$\{ x_n \}$可以收敛到局部极小值，值得注意的是迭代步长$\eta$并不是定值。



**三种梯度下降变体**

Batch Gradient Descent

​		Batch Gradient Descent 通过计算参数$\theta$关于整个训练集的损失函数的梯度。
$$
\theta \leftarrow \theta - \eta \grad J_{\theta}(\theta)
$$
​		但由于我们需要计算整个数据集来执行一次梯度更新，因此 Batch Gradient Descent 可能非常缓慢，并且对于大型数据集（大于内存大小）的训练将会变得十分棘手。此外 Batch Gradient Descent 也不允许我们在线更新模型。



Stochastic Gradiant Descent

​		与 Batch Gradient Descent 相对应的是 Sotchastic Gradient Descent ，其为每个训练样本$x^{(i)}$和标签$y^{(i)}$执行参数更新。
$$
\theta \leftarrow \theta - \eta \grad J_{\theta}(\theta;x^{(i)};y^{(i)})
$$
​		Batch Gradient Descent 对大型数据集有大量的冗余计算，因为它会在每个参数更新前重新计算相似的梯度。而 SGD 针对每个样本进行一次参数更新。由于 SGD 频繁执行更新，且变化很大，这导致目标函数震荡十分剧烈。





Mini-batch Gradient Descent

​		Mini-batch Gradient Descent 同时兼顾了上述两种方法的优势，针对$n$个训练样本的 mini-batch 计算损失进行参数梯度更新。
$$
\theta \leftarrow \theta - \eta \grad J_{\theta}(\theta;x^{(i:i+n)};y^{(i:i+n)})
$$
其优点在于：

- 降低了参数更新的方差，可以更稳定地收敛（与 SGD 相比较）。
- 利用深度学习库对常见大小 mini-batch 的矩阵进行高度优化的特性，可非常高效计算出其梯度。

​		常见 mini-batch 大小在 50 至 256 之间，但会因不同的应用场景而有所不同。训练神经网络时，通常选择 min-batch（**而当使用 mini-batch 时，通常也使用术语 SGD** ）。



**仍然存在问题**

Mini-batch Gradient Descent 虽然具有综合优势，但其并不能保证良好的收敛性，其还有一些需要解决的问题：

- 选择合适的学习率十分困难。学习率太小会导致收敛时间过长，而学习率过大又会阻碍收敛导致损失函数在最小值附近波动甚至发散。
- **学习率表**尝试通过例如调整训练过程中的学习率（例如退火）。即根据预定义的时间表或在各个时期之间的目标函数变化降到阈值以下时降低学习率。但是，这些计划和阈值必须预先定义，因此无法适应数据集的特征。
- 对于参数更新，通常我们采用相同学习率用于所有参数。如果数据集稀疏且模型特性有非常不同的出现频率，我们可能并不想将所有模型特征更新到相同的程度，且对很少出现的特性执行较大的更新。
- 另一个关键挑战在最小化高度非凸损失函数（神经网络中十分常见）如何避免陷入众多的局部最小值。Dauphin 等人认为关键困难在于并不是由于局部最小值，而是在于鞍点。这使得 SGD 很难逃脱，因为在所有维度上梯度都接近于 0 。



这里需要提一下凸函数的性质

- 一元可微函数在某个区间上是凸的，当且仅当它的导数在该区间上单调不减。

- 凸函数的任何极小值也是最小值。严格凸函数最多有一个最小值。

- 凸函数还有一个重要的性质：对于凸函数来说，局部最小值就是全局最小值。



**梯度下降优化算法**

一般通式如下
$$
V_t = \alpha V_{t-1} + \beta g \\
W_t = W_{t-1} - \gamma V_t
$$
其中$\beta, \gamma$可能附带学习率。



1.动量法Momentum

​		Momentum通过将上一个步骤的权重更新向量乘以动量参数$\gamma$，加在当前权重更新向量上，达到了在相关方向上加速 SGD 并抑制振荡。
$$
v_t = \gamma v_{t-1} + \eta \grad J(\theta) \\
\theta = \theta - v_t
$$
通常 momentum 参数$\gamma$设置为 0.9 。



2.AdaGrad算法



3.RMSProp算法（Root Mean Square Prop）

​		如果执行梯度下降的过程中，虽然横轴方向正在推进，但纵轴方向会有大幅度摆动。想减缓摆动方向的学习，即纵轴方向，同时加快，至少不是减缓横轴方向的学习，RMSprop算法可以实现这一点。（RMSProp主要是在调整学习率）

![](https://pic2.zhimg.com/80/v2-d959c25091a32e9b224487e11068caae_720w.jpg?source=1940ef5c)

​		在第t次迭代中，该算法会照常计算当下 mini-batch 的微分，所以要保留这个指数加权平均数，我们用到新符号$S_{dW}$。
$$
S_{dW} = \beta S_{dW} + (1 - \beta)(dW)^2 \\
S_{db} = \beta S_{db} + (1 - \beta)(db)^2 \\

W = W - \frac{\eta}{\sqrt{S_{dW} + \epsilon}} . dW \\
b = b - \frac{\eta}{\sqrt{S_{db} + \epsilon}} . db \\
$$
其中$epsilon > 0$，通常设置为$10^{-6}$，以确保我们不会因除以零或步长过大而受到影响。



4.Adadelta算法



5.Adam算法

​		Adam中动量直接并入了梯度一阶矩（指数加权）的估计。其次，相比于缺少修正因子导致二阶矩估计可能在训练初期具有很高偏置的RMSProp，Adam包括偏置修正，修正从原点初始化的一阶矩（动量项）和（非中心的）二阶矩估计。Adam算法策略可以表示为：

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g^2_t \\
\hat{m_t} = \frac{m_t}{1 - \beta_{1,t}}, \hat{v_t} = \frac{v_t}{1 - \beta_{2,t}} \\
W_{t+1} = W_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \hat{m_t}
$$
​		其中，$m_t, v_t$分别为一阶动量项和二阶动量项；$\beta_1, \beta_2$为动力值大小通常分别取0.9和0.999；$\hat{m_t}, \hat{v_t}$分别为各自的修正值。$W_t$表示t时刻即第t迭代模型的参数，$g_t = \grad J(W_t)$表示t次迭代代价函数关于W的梯度大小；$\epsilon$是一个取值很小的数（一般为1e-8）为了避免分母为0。

​		该方法和RMSProp很像，除了使用的是平滑版的梯度m，而不是原始梯度dx。推荐参数值eps=1e-8，beta1=0.9，beta2=0.999。在实际操作中，推荐Adam作为默认算法，一般比RMSProp要好一点。



#### 工程问题 ####

---

**Q: BN层在训练和测试的异同？**

**A: **Batch Normalization，是深度神经网络训练过程中，使每一层网络的输入保持相近的分布。

不同点：在**训练**过程中，BN是对每一批数据进行归一化，即用的是这个批次的均值和方差。而在**测试**的时候，我们不太会对一个batch的图像进行预测，而是单张图像的预测，因此这里就没有batch的概念，此时使用的均值和方差是全部训练数据的均值和方差（使用移动平均法求得）。

相同点：不管是训练还是测试，二者的计算公式都是一样的。假设$x$是前一层的特征，其中一个batch中有n个样本，分别为$x_1, x_2, ..., x_n$，我们对其进行归一化。
$$
\hat{x_i} = \gamma \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$
其中$\gamma和\beta$分别称作缩放系数和平移系数（weight和bias），它们可以在模型训练中得到。$\mu和\sigma^2$分别是均值和方差。



**Q: BN训练时为什么不用全部训练数据的均值和方差呢？**

**A: **因为这么做容易产生过拟合。原本我只针对每个batch的数据进行归一化到同一个分布，而每个batch的分布肯定会有差别，这个差别能够增加模型的鲁棒性，因此一定程度上能够减小过拟合。正因为如此，训练时采用将全部训练集打乱顺序，并使用一个较大的batch（batch太小无法较好的代表整个训练集的分布，太大就是现在这个问题，过拟合了并且增加了训练过程中的计算）。



**Q: BN和Conv怎么做融合？**

**A: **将BN和Conv进行融合是模型推理加速中的一个trick。对于某一BN层，若某通道均值为$\mu$，方差为$\sigma$，归一化层的权重为$\gamma$，偏移为$\beta$，该通道输入为$x_i$，输出为$\hat{x_i}$：
$$
\hat{x_i} = \gamma \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} x_i + (\beta - \frac{\gamma \mu}{\sqrt{\sigma^2 + \epsilon}})
$$
可以将BN层看成是**输入输出通道数相同，卷积核大小为1\*1的Conv层**。其中，BN层输出的第i个通道对应输入的第i个通道，与其它通道没有关系。若将BN层的操作用卷积层的矩阵相乘来表示，则$W$应该是一个对角矩阵。令BN层的权重为$W_{bn}$，偏置为$b_{bn}$，矩阵形式：
$$
W_{bn} = \left[ \matrix{
	\frac{\gamma_1}{\sqrt{\sigma^2_1 + \epsilon}} & & & \\
	& \frac{\gamma_2}{\sqrt{\sigma^2_2 + \epsilon}} & & \\
	& & \frac{\gamma_3}{\sqrt{\sigma^2_3 + \epsilon}} & \\
	& & & \frac{\gamma_4}{\sqrt{\sigma^2_4 + \epsilon}}
} \right] \\

b_{bn} = \left[ \matrix{
	\beta_1 - \frac{\gamma_1 \mu_1}{\sqrt{\sigma^2_1 + \epsilon}} \\
	\beta_2 - \frac{\gamma_2 \mu_2}{\sqrt{\sigma^2_2 + \epsilon}} \\
	\beta_3 - \frac{\gamma_3 \mu_3}{\sqrt{\sigma^2_3 + \epsilon}} \\
	\beta_4 - \frac{\gamma_4 \mu_4}{\sqrt{\sigma^2_4 + \epsilon}}
} \right]
$$
令前一层的Conv层的输入为$F_0$，该Conv层的输出或下一层BN层的输入为$F_1$，BN层输出为$F_2$，Conv层的W和b分别为$W_{conv}和b_{conv}$，BN层的W和b分别为$W_{bn}和b_{bn}$。
$$
F_1 = W_{conv} F_0 + b_{conv} \\
F_2 = W_{bn} F_1 + b_{bn}
$$
再利用矩阵乘法交换律，有：
$$
F_2 = W_{bn} (W_{conv} F_0 + b_{conv}) + b_{bn} \\
= W_{bn} W_{conv} F_0 + W_{bn} b_{conv} + b_{bn}
$$
在实际计算过程中，是将矩阵展开成一维向量进行点乘运算，所以这里的$W_{conv}, b_{conv}, W_{bn}, b_{bn}$都需要reshape成一维，即$[num_out_channels, -1]$。

**参考代码（Google Colab）**

```python
    import torch
    import torchvision
    
    def fuse(conv, bn):
        fused = torch.nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=True
        )
    
        # setting weights
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
        fused.weight.copy_( torch.mm(w_bn, w_conv).view(fused.weight.size()) )
        
        # setting bias
        if conv.bias is not None:
            b_conv = conv.bias
        else:
            b_conv = torch.zeros( conv.weight.size(0) )
        b_conv = torch.mm(w_bn, b_conv.view(-1, 1)).view(-1)
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
                              torch.sqrt(bn.running_var + bn.eps)
                            )
        fused.bias.copy_( b_conv + b_bn )
    
        return fused
    
    # Testing
    # we need to turn off gradient calculation because we didn't write it
    torch.set_grad_enabled(False)
    x = torch.randn(16, 3, 256, 256)
    resnet18 = torchvision.models.resnet18(pretrained=True)
    # removing all learning variables, etc
    resnet18.eval()
    model = torch.nn.Sequential(
        resnet18.conv1,
        resnet18.bn1
    )
    f1 = model.forward(x)
    fused = fuse(model[0], model[1])
    f2 = fused.forward(x)
    d = (f1 - f2).mean().item()
    print("error:",d)
```

不同版本的代码有些许不同，主要在于Conv层的$b_{bn}$计算时是否设置为0。



**Q: Max Pooling和 Avg Pooling反向传播时的区别？**

**A: **我自己的记忆：max pooling由于在前向推理的过程中，只取了pooling中的最大值，其他的值舍弃了。所以在反向传播时，最大值依然填到它原本的位置，但是其他值由于找不到了，所以在空白位置上填上一个固定值（比如0）。而avg pooling用的是取平均值，所以在反向传播还原feature map时，将当前的特征取平均后填在空白位置上。

这个链接和我的想法一致：https://blog.csdn.net/Jason_yyz/article/details/80003271



**Q: dropout和BN可以同时使用吗？**

**A: **dropout在训练时以一定概率p使某些神经元的函数的输出为0（“删除”该神经元），这些“删除”的神经元权重备份起来；用未“删除”的神经元进行前向推理得到误差，反向传播时对这些保留的神经元进行参数更新，临时“删除”的不更新；恢复被”删除“的神经元。dropout在测试时采用全部神经元进行前向推理（这样的话，那每个神经元的权重参数应该都乘上p，但实际上是训练时保留的神经元权重除以(1-p)，而测试时不作任何操作）。BN的目的是统一各层（准确的说是通道维度上）的均值和方差，这样对于权重初始化就没有以前那么苛刻了，并且可以采用较大的学习率，并支持更多激活函数，增加了泛化能力，使模型更快收敛。

dropout和BN层不同同时使用。dropout之后的feature map改变了数据的标准差（令标准差变大，若数据均值非0时，甚至均值也会改变）。如果与BN一起使用，由于BN在训练时保存了训练集的均值与方差，dropout影响了所有保存的均值与标准差，那么将影响网络的准确性。

解决方案：1.把dropout放到BN层后面就可以，这样就不会产生方差偏移的问题。2.采用高斯dropout（一个均匀分布的dropout，又称Uout），输入数据为正态分布，只需要在dropout后面乘sqrt(0.5)即可恢复原来的标准差。但是对于非0的均值的改变，或者输入非正态分布，依然没有好的解决办法。



**Q: 梯度消失和梯度爆炸？**

**A: **梯度消失的原因：**深层网络**中，采用了**不合适的激活函数**，比如sigmoid；梯度爆炸的原因：**深层网络**中，**权重初始化值太大**。

![img](https://pic4.zhimg.com/80/v2-a49d6d008278e9b45a7c9db4c661319f_720w.jpg)

对于一个神经网络，初始权重为$w$，隐藏层分别为$h_i$，最后的输出为$y$。

权重W更新过程如下，t代表当前这一次迭代。
$$
W_{t+1} = W_t + \eta * \frac{\partial Loss}{\partial W_t}
$$
其中
$$
\frac{\partial Loss}{\partial W} = \frac{\partial Loss}{\partial y} * \frac{\partial y}{\partial h_{t}} * \frac{\partial h_t}{\partial h_{t-1}} * ... * \frac{\partial h_2}{\partial h_{1}} * \frac{\partial h_1}{\partial w}
$$
而我们又知道，每个隐藏层都会接一个激活函数，我们这里直接把$h_i$当做是激活函数的过程了。因此从$h_i$到$h_{i+1}$经过了一个激活函数，$\frac{\partial h_{i+1}}{\partial h_i}$可以看做是激活函数的求导。

讲到这里就容易理解了，如果激活函数求导之后的值<<1，那么连乘之后会是一个趋近于0的值；如果是一个>>1的值，那么连乘之后就是一个趋近于无穷的值。



**为什么 MobileNet、ShuffleNet 在理论上速度很快，工程上并没有特别大的提升？**

​		首先我们要知道，MobileNet相对普通卷积做了什么。MobileNet使用DW卷积和逐点卷积减少了普通卷积的计算量，但是加深了网络的层数。

​		MobileNet在GPU上速率反而没有CPU上的快。这是因为GPU是并行处理大规模数据（矩阵内积）的运算平台，而CPU则倾向于对数据串行计算。因此，假设GPU的显存足够大（无限大），因为每层的计算都可以并行一次处理，此时总计算时间主要取决于**网络的层数**。而对于缺乏并行计算的CPU而言，总运算时间主要取决于**网络的总计算量**。



参考 https://www.zhihu.com/question/343343895



#### 其他理论 ####

---

**Q: 线性回归**

**A: 我们从一元线性回归推导更一般化的线性回归。**$y = w_1 x_1 + w_2 x_2 + w_3 x_3 + ...$

n个样本点$(x_i, y_i)$，对应标签为$\hat{y_i}$，使这n个样本点全部落在一元线性回归方程$y_i = w_1*x_i + w_2$附近，不妨设误差为$\epsilon$，有$\hat{y_i} - y_i = \epsilon$，回归直线满足的条件是：全部样本点与对应的回归估计值的误差平方和最小。
$$
argmin_{w_1, w_2} \sum^{n}_{i=1} = \sum^{n}_{i=1} (\hat{y_i} - w_1 * x_i + w_2)^{2}
$$
现在我们要求$w_1, w_2$，最小二乘估计思路：用直线尽可能的去拟合n个样本点。原问题可以转换为求这个二元函数的最小值。那么我们分别对$w_1, w_2$求偏导，并令导数为0，再将n个样本点带入，能够得到一组$w_1, w_2$拟合这些样本点，并使它们的误差和最小。



换一个角度，从**统计学**的角度去推导。$\hat{y_i} - y_i = \epsilon$，误差可以看做是**随机变量之和**共同作用而产生的，由中心极限定理可知，**随机变量之和的分布**近似服从**正态分布**。因此这里我选择一条直线去拟合所有样本点，那么可以认为是大部分样本点都在直线附近，因此误差大部分集中在零值附近，因此我这里使用均值为0，方差为$\sigma^{2}$的正态分布作为$\epsilon$的先验分布是比较合理的。
$$
p(\epsilon) = \frac{1}{\sqrt{2 \pi} \sigma} e^{-\frac{\epsilon^2}{2 \sigma^2}} \\
\epsilon = \hat{y_i} - y_i \\
p(\epsilon) = \frac{1}{\sqrt{2 \pi} \sigma} e^{-\frac{(\hat{y_i} - y_i)^2}{2 \sigma^2}} \\
$$
再把$y_i = w_1 x_i + w_2$带入，有
$$
p(y_i|x_i; w) = \frac{1}{\sqrt{2 \pi} \sigma} e^{-\frac{(\hat{y_i} - w_1 x_i - w_2)^2}{2 \sigma^2}} \\
$$
将n个样本点带进来，再解这个式子，用到极大似然估计：
$$
argmax_{w_1, w_2} = \prod^{n}_{i=1} \frac{1}{\sqrt{2 \pi \sigma}} e^{-\frac{(\hat{y_i} - w_1 x_i - w_2)^2}{2 \sigma^2}}
$$
两边同时取对数，单调性不变
$$
argmax_{w_1, w_2} = \sum^{n}_{i=1}log \frac{1}{\sqrt{2 \pi \sigma}} e^{-\frac{(\hat{y_i} - w_1 x_i - w_2)^2}{2 \sigma^2}} \\
= \sum^{n}_{i=1} log(e^{-\frac{(\hat{y_i} - w_1 x_i - w_2)^2}{2 \sigma^2}}) - log(\sqrt{2 \pi} \sigma) \\
= \sum^{n}_{i=1} - \frac{(\hat{y_i} - w_1 x_i - w_2)^2}{2 \sigma^2} - log(\sqrt{2 \pi} \sigma)
$$
把常数去掉，单调性不变
$$
argmax_{w_1, w_2} = \sum^{n}_{i=1} - \frac{(\hat{y_i} - w_1 x_i - w_2)^2}{2 \sigma^2} - log(\sqrt{2 \pi} \sigma) \\
= - \sum^{n}_{i=1} (\hat{y_i} - w_1 x_i - w_2)^2 \\
= argmin_{w_1, w_2} (\hat{y_i} - w_1 x_i - w_2)^2
$$
在机器学习算法中，求解最小化损失函数时，可以通过**梯度下降法**来一步步进行迭代求解，从而得到最小化损失函数的模型参数值，梯度下降算法不一定能够找到全局的最优解，有可能是一个局部最优解。然而，如果损失函数是凸函数，那么梯度下降法得到的解就一定是全局最优解。



通俗易懂 https://zhuanlan.zhihu.com/p/72513104

多个角度推导线性回归 https://zhuanlan.zhihu.com/p/36910496



**Q: 逻辑回归**

**A: **逻辑回归虽然被称为回归，但实际上是分类模型，常用于二分类。逻辑回归的前提是：**假设数据服从于这个分布**，利用极大似然估计做参数的估计。

逻辑回归是基于伯努利分布（两点分布）上推导出来的。二项分布是n重伯努利实验成功次数的离散分布。



逻辑斯蒂分布是一种连续型的概率分布，其分布函数如
$$
F(X) = P(X <= x) = \frac{1}{1 + e^{\frac{-(x_ - \mu)}{\gamma}}}
$$
概率密度函数
$$
f(X) = F^{'}(X) = \frac{e^\frac{-(x - \mu)}{\gamma}}{\gamma(1 + e^{\frac{-(x - \mu)}{\gamma}})^2}
$$
设X是连续随机变量，X服从**逻辑斯谛分布**是指X具有上述分布函数和密度函数。



逻辑回归实际上就是在线性回归的最后一层，添加了一个分类概率到输出结果的一个映射关系（通过概率值来判断最终类别），这个映射关系用一个单调可导函数表示出来。而当$\mu=0, \gamma=1$时，逻辑分布函数就是一个sigmoid函数，我们常用sigmoid函数来进行逻辑回归。



**推理**

n个样本点$X = (x_i, y_i), i = 1,2,...$进行二分类。我们先想到采用线性回归$Y = W^T X + b$去拟合这些样本，由于$Y = W^T X + b \in R$，结果是连续的，我们考虑用它来拟合条件概率$P(Y=1|X)$。但是又因为二分类问题，概率只取0和1，故考虑使用广义线性模型。

最理想的是单位阶跃函数：
$$
P(y=1|x) = \left\{\begin{array}{c} 
	0, & y < 0 \\
	0.5, & y = 0 \\
	1, & y > 0
\end{array}\right. \\
y = w^T x + b
$$
但是这个阶跃函数不可微，因此常用另一个函数替代
$$
y = \frac{1}{1 + e^{-(w^T x + b)}}
$$
这个形式就可以看成是一个线性函数外层再套一个sigmoid函数。
$$
w^T x + b = ln\frac{y}{1 - y}
$$
$\frac{y}{1 - y}$这个比值称为几率
$$
w^T x + b = ln \frac{P(y=1|x)}{1 - p(y=1|x)} \\
p(y=1|x) = \frac{1}{1 + e^{-(w^T x + b)}}
$$
**损失函数**

上面那个函数怎么求损失？用似然估计法来求解
$$
L(w) = \prod p(x_i)^{y_i} (1 - p(x_i))^{1 - {y_i}}
$$
然后两边取对数，得到对数似然函数，取负再求极小值
$$
ln L(w) = - \sum^{n}_{i=1} [y_i log p(x_i) + (1 - y_i) log (1 - y_i)]
$$
取整个数据集上的平均
$$
J(w) = - \frac{1}{n} \sum^{n}_{i=1} [y_i log p(x_i) + (1 - y_i) log (1 - y_i)] \\
= - \frac{1}{n} \sum^{n}_{i=1} [y_i ln \frac{p(x_i)}{1 - p(x_i)} + ln(1 - p(x_i))] \\
= - \frac{1}{n} \sum^{n}_{i=1} [y_i(w x_i + b) - ln \frac{1}{1 - p(x_i)}] \\
= - \frac{1}{n} \sum^{n}_{i=1} [y_i(w x_i + b) - ln (1 + \frac{p(x_i)}{1 - p(x_i)})] \\
= - \frac{1}{n} \sum^{n}_{i=1} [y_i(w x_i + b) - ln (1 + e^{w x_i + b})]
$$


**如何对损失函数求解**

通常有两种方法：梯度下降法和牛顿法

**随机梯度下降**，对$w$求偏导
$$
\frac{\partial J(w)}{\partial w} = - \frac{1}{n} \sum^{n}_{i=1} [y_i x_i - \frac{x_i e^{w x_i + b}}{1 + e^{w x_i + b}}] \\
= - \frac{1}{n} \sum^{n}_{i=1} [y_i - \frac{1}{e^{-(w x_i + b)} + 1}] x_i \\
- \frac{1}{n} \sum^{n}_{i=1} [y_i - p(x_i)] x_i
$$
梯度的反传传播
$$
w_{t+1} = w_t + \eta \frac{\partial J(w)}{\partial w}
$$


**牛顿法**，在**现有极小点**估计值的附近对f(x)作**二阶**泰勒展开，进而找到**极小点**的**下一个估计值**。假设$w_k$为当前极小值估计值，那么有：
$$
\varphi(w) = J(w) + J^{'}(w)(w - w_k) + \frac{1}{2!} J^{''}(w)(w - w_k)^2 + R
$$
对$\varphi(w)$对w求导，再将$w=w_k$带入，并令其导数为0，得到
$$
w_{k+1} = w_k - \frac{J^{'}(w_k)}{J^{''}(w_k)}
$$
详细讲解 https://zhuanlan.zhihu.com/p/74874291



逻辑回归对比线性回归，Sigmoid 函数到底起了什么作用？

（1）线性回归是在实数域范围内进行预测，而分类范围则需要在 [0,1]，逻辑回归减少了预测范围；

（2）线性回归在实数域上敏感度一致，而逻辑回归在 0 附近敏感，在远离 0 点位置不敏感，这个的好处就是模型更加关注分类边界，可以增加模型的鲁棒性。



**Q: L1 L2正则化**

**A: **L1正则化，LASSO回归，相当于为模型添加了一个先验知识，w服从0均值的拉普拉斯分布。拉普拉斯概率密度函数（展现的是x的分布，这里x相当于w）
$$
f(x) = -\frac{1}{2 \lambda} e^{- \frac{|x - \mu|}{\lambda}}
$$
它是由两个指数函数组成的，一般$\mu=0$。

<img src="https://images2018.cnblogs.com/blog/890640/201804/890640-20180418120156153-1911436669.png" alt="Figure_1" style="zoom:50%;" />

我们把w替换x，b代替$\lambda$，并取$\mu=0$，有
$$
f(w) = \frac{1}{2 b} e^{- \frac{|w|}{b}}
$$
这就是L1正则项。在逻辑回归中的对数损失函数$J(w)$后加上这个东西。



L2正则化

Ridge回归，为模型添加一个先验知识：w服从0均值高斯分布，高斯分布的概率密度函数（x的分布）：
$$
f(x) = \frac{1}{\sqrt{2 \pi \sigma}} e^{- \frac{(x - \mu)^2}{\sigma^2}}
$$
将w替换x，$\mu=0$，得到
$$
f(w) = \frac{1}{\sqrt{2 \pi \sigma}} e^{- \frac{w^2}{\sigma^2}}
$$
这就是L2正则项。在逻辑回归中的对数损失函数$J(w)$后加上这个东西。



L1 正则化就是在 loss function 后边所加正则项为 L1 范数，加上 L1 范数容易得到稀疏解（0 比较多）。L2 正则化就是 loss function 后边所加正则项为 L2 范数的平方，加上 L2 正则相比于 L1 正则来说，得到的解比较平滑（不是稀疏），但是同样能够保证解中接近于 0（但不是等于 0，所以相对平滑）的维度比较多，降低模型的复杂度。

详细讲解 https://zhuanlan.zhihu.com/p/74874291



**Q: KNN**

**A: **



**Q: 朴素贝叶斯**

**A: **



**神经网络 VS 决策树算法**

​		以决策树为基模型，如随机森林，gradient boosting和xgboot的性能提升主要来自于集成学习。

​		神经网络适用于的场景是图片、视频、文本、音频。有大量特征，每个特征和最终结果都可能有关但又不那么明显（所谓从特征到结果有较大的gap，你用一个像素去判断手写数字几乎就是瞎猜，但是用一个特征判断泰坦尼克号乘客的生存情况准确率很有可能高于50%），和容易获得海量数据的场景。深度神经网络可以从大量特征中提取更高级的特征，海量数据可以减小过拟合。

​		而传统数据挖掘人物相对而言数量较少，特征与结果之间的关系更为明显，数据量相对较低。如果使用神经网络，很有可能造成严重过拟合。对这类任务，一般采用特征工程 + 集成学习的方法。



特征工程

弥补缺失数据

数据类型转换：数据类型转换为可量化的类型，比如颜色，类别等可以用数字代替。

数据缩放：归一化。把不同范围的数据归一化到[0, 1]或者投影到正态分布上。

调整参数：初始化权重，调整学习率（神经网络）。



参考回答 https://www.zhihu.com/question/68130282/answer/260086469



**Q: 决策树**

**A: **比较常用的决策树有ID3，C4.5和CART（Classification And Regression Tree），CART的分类效果一般优于其他决策树。下面介绍具体步骤。

**信息熵增益**

信息熵：假设样本集合D中第k类样本所占的比例为$p_k(k=1,2,...,|\gamma|)$，则D的**信息熵**定义为：
$$
Entropy = Ent(D) = - \sum^{|\gamma|}_{k=1} p_k log_2(p_k)
$$
Ent(D)越小，则D的纯度越高。约定$p_k=0$时，$p_k log_2(p_k)=0$；

（1）当D中只有1类时，此时$log_2(p_k=1)=0, 则Ent(D) = 0$，值最小，纯度最高。

（2）当D中所有类占比相同时，此时$Ent(D)=log_2(|\gamma|)$，值最大，纯度最小。



现在假设属性a有V个不同取值，若使用a来对样本D进行划分，会产生V个分支节点，每个分支节点上的样本在a上的取值都相同（这句话怎么理解？）。记第v个分支节点上样本数为$D_v$，则可计算相应的信息熵，再根据每个节点上的样本占比给分支节点赋予权重$\frac{|D_v|}{|D|}$，可计算出以a作为划分属性所获得的**信息熵增益**：
$$
Gain(D, a) = Ent(D) - \sum^{V}_{v=1} \frac{|D_v|}{|D|} Ent(D_v)
$$
一般而言，信息增益越**大**，则意味着用属性a来进行划分所获得的"纯度提升"越**大**，因此，我们可用信息增益来进行决策树的划分属性选择。

ID3 决策树学习算就是以信息增益为准则来选择划分属性。

如果不理解，可以参考这个例子 https://zhuanlan.zhihu.com/p/26596036

在计算$Ent(D_v)$时，a看做是“矮”，v可以是“嫁或不嫁”这一属性。则计算D中的$p(x='矮')=7/12$，而$D_v$中的$p(y='嫁'|x='矮')=1/7, p(y='不嫁'|x='矮')=6/7$，即条件概率。



**增益率**

​		通过对ID3的学习，可以知道ID3存在一个问题，那就是越细小的分割分类错误率越小，所以ID3会越分越细。这种分割显然只对训练数据有用，对于新的数据没有意义，这就是过拟合。

​		分割太细了，训练数据的分类可以达到0错误率，但是在面对测试数据分错率反倒上升了。决策树是通过分析训练数据，得到数据的统计信息，而不是专为训练数据量身定做。

基于此，著名的C4.5决策树算法不直接使用信息增益，而是使用“增益率”来选择最优划分属性：
$$
Gain_{ratio}(D, a) = \frac{Gain(D, a)}{IV(a)} \\
IV(a) = -\sum^{V}_{v=1} \frac{|D_v|}{|D|} log_2(\frac{|D_v|}{|D|})
$$
其中IV(a)称为a的“固有值”，a的可能取值数目越多，IV(a)的值越大。

显然C4.5为了避免分割太细，对ID3进行了改进。优化项要除以分割太细的代价，这个比值叫做信息增益率，显然分割太细分母增加，信息增益率会降低。

​		需注意的是，增益率准则对可取值数目较少的属性有所偏好，因此， C4.5算法并不是直接选择增益率最大的候选划分属性，而是使用了一个启发式：**先从候选划分属性中找出信息增益高于平均水平的属性，再从中选择增益率最高的。**



**基尼指数**

CART是一个二叉树，也是回归树，同时也是分类树，CART的构成简单明了。

CART只能将一个父节点分为2个子节点。CART用Gini指数来决定如何分裂。

Gini指数：总体内包含的类别越杂乱，Gini指数就越大（跟熵的概念很相似）。
$$
Gini(D) = \sum^{|\gamma|}_{k=1} \sum_{k != k} p_k p_{k'} = 1 - \sum^{|\gamma|}_{k=1} p_k^2
$$
显然，Gini反映了从数据集D中随机抽取两个样本，其类别标记不一样的概率，因此Gini越小，则D的纯度越高。

属性a的基尼指数定义为：
$$
Gini_{index}(D, a) = \sum^{V}_{v=1} \frac{|D_v|}{|D|} Gini(D_v)
$$
选择使上式最**小**的属性a作为划分属性。



**剪枝处理**

**剪枝**是决策树学习算法对付**过拟合**的主要手段。

​		**预剪枝**是指在决策树生成过程中，对每个结点在划分前先进行估计，若当前结点的划分不能带来决策树泛化性能提升，则停止划分并将当前结点标记为叶结点；

​		**后剪枝**则是先从训练集生成一棵完整的决策树，然后自底向上地对**非叶结点**进行考察，若将该结点对应的子树替换为叶结点能带来决策树泛化性能提升，则将该子树替换为叶结点。



**判断决策树泛化性能是否提升**：留出法，即预留一部分数据用作"验证集"以进行性能评估。

例如：在预剪枝中，对于每一个分裂节点，对比分裂前后决策树在验证集上的预测精度，从而决定是否分裂该节点。而在后剪枝中，考察非叶节点，对比剪枝前后决策树在验证集上的预测精度，从而决定是否对其剪枝。



**两种方法对比**：

1）预剪枝使得决策树的很多分支都没有"展开”，不仅降低过拟合风险，而且显著减少训练/测试时间开销；但，有些分支的当前划分虽不能提升泛化性能，但在其基础上进行的后续划分却有可能导致性能显著提高，即预剪枝基于"贪心"本质禁止这些分支展开，给预剪枝决策树带来了欠拟含的风险。

2）后剪枝决策树通常比预剪枝决策树保留了更多的分支。一般情形下，后剪枝决策树的欠拟合风险很小，泛化性能往往优于预剪枝决策树，但后剪枝过程是在生成完全决策树之后进行的，并且要自底向上地对树中的**所有非叶结点**进行逐一考察，因此其训练时间开销比未剪枝决策树和预剪枝决策树都要大得多。



通俗易懂 https://zhuanlan.zhihu.com/p/30059442

详细讲解 https://www.cnblogs.com/liuqing910/p/9121736.html

决策树代码实现 https://shuwoom.com/?p=1452



**Q: boost提升方法**

**A: **boosting思想可以诠释为

（1）在整个数据集上训练模型 h1；

（2）对 h1 表现较差的区域的数据加权，并在这些数据上训练模型 h2；

（3）对 h1 ≠ h2 的区域的数据加权重，并在这些数据上训练模型 h3；

（4）...。

我们可以串行地训练这些模型，而不是并行训练。这是 Boosting 的本质！

![](http://blog.chinaunix.net/attachment/201203/12/8695538_1331555414I2if.jpg)

参考 https://zhuanlan.zhihu.com/p/57689719



**xgboost**





**GBDT**





**Q: 支持向量机**

**A: **

https://zhuanlan.zhihu.com/p/77750026



**Q: nms**

**A: **NMS算法一般是为了解决多个预测框选中同一个目标的情况，只保留一个框并去掉多余框的算法。一般会设定一个阈值nms_thresh=0.5，具体的实现思路如下：

（1）针对**每个类别**中的boxes，先按照score从高到低排序，再选择这类box中**score（置信度）**最大的那一个记为box_best，在原数组中将这个box_best去掉，添加到最终保留boxes的数组中；

（2）将这个box_best与原数组中的其他box进行iou计算，iou大于nms_thresh的box删掉，从原数组中去掉。直到遍历完原数组；

（3）再从原数组中剩余的box中选择score最大的那个作为box_best，从原数组中删除这个box，并保存到最终的保留数组中；

（4）重复以上3个操作，直到原数组为空。



代码实现

```python
def nms(boxes, nms_thresh=0.5):
    '''
    :param boxes (numpy): (x1, y1, x2, y2)
    :param nms_thresh:
    '''
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    
    area = (y2 - y1 + 1) * (x2 - x1 + 1)
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (area[i] + area[order[1:]] - inter)
        
        inds = np.where(ovr <= nms_thresh)[0]
        # inds是一个list，由于order第一个取出去了，所以这里inds都要+1
        order = order[inds + 1]
    return keep
```



**Q: soft-nms**

**A: **经典的nms算法存在一个问题：对于重叠物体无法很好地检测。当图像中存在两个重叠度很高的物体时，它只会保留置信度最高的那个框，而将其他框给剔除。

​		而我们的期望是将重叠的两个物体都检测出来，因此提出了soft-nms。相对于经典nms算法，soft-nms仅仅修改了一行代码。当选取了最大置信度的bbox之后，计算其余bbox与当前置信度最大bbox的iou的过程中，经典nms的做法是直接删除iou大于阈值的bbox，而soft-nms则是使用一个基于iou衰减的函数，降低iou大于阈值的bbox的置信度。iou越大，衰减程度越大。



经典nms和soft-nms的衰减函数的区别，可以通过如下公式来表示：

**经典nms**的置信度衰减公式如下：
$$
s_i = \left\{ \begin{array}{c}
s_i, & iou(M, b_i) < N_t, \\
0, & iou(M, b_i) >= N_t
\end{array} \right.
$$
其中$s_i$是置信度，$N_t$是设置的阈值，$M$是选中的当前置信度最大的bbox，$b_i$是其他bbox。



**soft-nms**的置信度衰减公式：
$$
s_i = \left\{ \begin{array}{c} 
s_i, & iou(M, b_i) < N_t, \\
s_i (1 - iou(M, b_i)), &  iou(M, b_i) >= N_t 
\end{array} \right.
$$
**soft-nms高斯**置信度衰减公式：
$$
s_i = s_i e^{\frac{- iou(M, b_i)^2}{\sigma}}, \ \ \forall b_i \notin D
$$


soft-nms的代码实现（python）

```python
```



nms, soft_nms参考

https://blog.csdn.net/lz867422770/article/details/100019587?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-1.pc_relevant_aa&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-1.pc_relevant_aa&utm_relevant_index=2



#### 目标检测 ####

---

先来一张大图。

![preview](https://pic3.zhimg.com/v2-dd7959839adc00c2803eb69574650a5a_r.jpg)

![](https://pic4.zhimg.com/80/v2-88544afd1a5b01b17f53623a0fda01db_720w.jpg)



**输入端创新**

Mosaic数据增强



cmBN



（SAT自对抗训练）这个没用到算了。



**BackBone创新**

CBL（Conv + Bn + LeakyReLU）改为CBM（Conv + Bn + Mish）



Mish激活函数



Dropblock



**Neck创新**

SPP

<img src="https://pic1.zhimg.com/80/v2-60f3d4a7fb071766ac3c3bf70bb5a6f8_720w.jpg" style="zoom:80%;" />

作者在SPP模块中，使用$k={1x1,5x5,9x9,13x13}$的最大池化的方式，再将不同尺度的特征图进行Concat操作。

SPP模块内部采用不同大小的kernel size和stride实现不同感受野特征输出，有利于待检测图像中目标大小差异较大的情况。



FPN + PAN

和Yolov3的FPN层不同，Yolov4在FPN层的后面还添加了一个**自底向上的特征金字塔。**其中包含两个**PAN结构。**如图中标示。（这里上采样用的是插值，下采样用的卷积）

<img src="https://pic2.zhimg.com/80/v2-a204a672779d1c2bc26777437771cda4_720w.jpg" style="zoom:67%;" />

这样结合操作，FPN层自顶向下传达**强语义特征**，而特征金字塔则自底向上传达**强定位特征**，两两联手，从不同的主干层对不同的检测层进行参数聚合。

<img src="https://pic2.zhimg.com/80/v2-c2f9cb3d71bc3011f6f18adc00db3319_720w.jpg" style="zoom: 67%;" />

原本的PANet网络的**PAN结构**中，两个特征图结合是采用**shortcut（add）**操作，而Yolov4中则采用**concat（通道上）**操作，特征图融合后的尺寸发生了变化。



**Prediction创新**

iou, giou, diou, ciou的区别。

**iou**
$$
IoU = \frac{A \cap B}{A \cup B} \\
L_{IoU} = 1 - IoU
$$
iou的缺点：
（1）作为损失函数，如果两个box没有相交，按照上面这个公式，分子为0，则loss计算出来也为0。这样梯度反向传播，不能进行回归。

（2）不能很好地反应两个box的重合度。比如下面3种情况，它们的iou相同，但显然大的box的重合情况要好于小的box。

<img src="https://pic2.zhimg.com/80/v2-95449558cb098ff9df8c4d31474bd091_720w.jpg" style="zoom: 67%;" />

**giou**

（1）giou和iou一样，只对比值敏感，而对scale也不敏感；

（2）iou的取值是[0, 1]，而giou的取值范围为[-1, 1]。两个box完全重合取到1，两个box无限远取到-1；

（3）与IoU只关注重叠区域不同，**GIoU不仅关注重叠区域，还关注其他的非重合区域**，能更好的反映两者的重合度。

<img src="https://pic1.zhimg.com/80/v2-8b616d48a90ed51a94cfb71bdddabf90_720w.jpg" style="zoom: 25%;" />
$$
GIoU = IoU_{AB} - \frac{|C \backslash (A \cap B)|}{|C|} \\
L_{GIoU} = 1 - GIoU
$$
**diou**

​		DIoU比GIoU和IoU都更加符合目标框回归的机制。将目标与anchor之间的距离，重叠率以及尺度都考虑进来了，使目标框回归变得稳定，不会像前两者一样在训练的时候出现发散的问题。
$$
DIoU = IoU_{AB} - \frac{\rho^2(b, b^{gt})}{c^2} \\
L_{DIoU} = 1 - DIoU
$$
对照下面这幅图，介绍公式中的参数。$b, b^{gt}$分别是预测框和真实框的中心点，$\rho(x)$表示计算两个中心点之间的欧氏距离，$c$表示能同时包含预测框和真实框的大框的对角线距离。

<img src="https://pic3.zhimg.com/80/v2-1e4b54001c4abdf392fe9d4877c83972_720w.jpg" style="zoom: 67%;" />

（1）DIoU的取值也是有正负的。DIoU Loss在与目标重叠时，就是一个负值，可以为边界框提供移动方向。

（2）DIoU loss可以直接最小化两个目标框的距离，因此比GIoU loss收敛快得多。

（3）对于包含两个框在水平方向和垂直方向上这种情况，DIoU损失可以使回归非常快，而GIoU损失几乎退化为IoU损失。

（4）DIoU还可以替换普通的IoU评价策略，应用与NMS中，使得NMS得到的结果更加合理和有效。



**CIoU (Complete IoU)**
$$
惩罚项 \ \ R_{CIoU} = \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v
$$
其中$\alpha$是权重系数，$v$用来衡量长宽比的相似性，定义为$v=\frac{4}{\pi^2} (arctan \frac{w^{gt}}{h^{gt}} - arctan \frac{w}{h})^2$。

完整CIoU的损失函数定义为：
$$
L_{CIoU} = 1 - IoU + R_{CIoU} \\
= 1 - IoU + \frac{\rho^2 (b, b^{gt})}{c^2} + \frac{4}{\pi^2} (arctan \frac{w^{gt}}{h^{gt}} - arctan \frac{w}{h})^2
$$
最后，CIoU loss的梯度类似于DIoU loss，但还要考虑$v$的梯度。在长宽在$[0, 1]$的情况下，$w^2 + h^2$的值通常很小，会导致梯度爆炸，因此在$\frac{1}{w^2 + h^2}$实现时将替换成1。



**IOU_Loss**：主要考虑检测框和目标框重叠面积。

**GIOU_Loss**：在IOU的基础上，解决**边界框不重合**时的问题。

**DIOU_Loss**：在IOU和GIOU的基础上，考虑**边界框中心点距离**的信息。

**CIOU_Loss**：在DIOU的基础上，考虑**边界框宽高比的尺度信息**。

简单 https://blog.csdn.net/chen1234520nnn/article/details/118525518

详细 https://zhuanlan.zhihu.com/p/94799295



**ciou_loss**





**diou_nms**





YOLOv4简化图

![](https://pic2.zhimg.com/80/v2-fdafb0bf53f6ddca68776a3672af5921_720w.jpg)



yolov4创新 https://zhuanlan.zhihu.com/p/136115652

yolov4详细 https://zhuanlan.zhihu.com/p/143747206



---

**YOLOv3**

yolov3步骤讲解 https://zhuanlan.zhihu.com/p/76802514

yolov3详细提问 https://zhuanlan.zhihu.com/p/367395847



0.数据预处理
**输入图像处理**

将原图按照宽高比例缩放至需要的输入尺寸，取$min(w / image_w, h / image_h)$。保证较长边缩放为需要的输入尺寸，较短边按比例缩放不会扭曲。



**训练数据归一化**

yolov3需要的训练数据的标签是根据原图尺寸归一化了的（即真实框的size从绝对值变换为一个相对原图的比例）。这样做是为了避免大边框的影响比小边框更大（yolov3对于小物体的检测误差更大）。做了归一化之后，大小边框都被同等看待，而且训练也容易收敛。



1.网络架构

**backbone**：Darknet53

<img src="https://pic4.zhimg.com/80/v2-8385e8c24d95ababae443bd4db85a33f_720w.jpg" style="zoom: 50%;" />

可以看出Darknet53实际上就是卷积网络+残差层。



**YOLOv3网络全貌**

Yolov3使用Darknet-53作为整个网络的分类骨干部分。在Yolov3论文中并未给出全部网络结构。根据代码，整理数据流图如下（参考多份Yolov3代码，正确性可以保证）。

![](https://pic3.zhimg.com/80/v2-d2596ea39974bcde176d1cf4dc99705e_720w.jpg)

对比另一张图

<img src="https://pic1.zhimg.com/80/v2-41fcbf90757e76578eaf1e6994cb159c_720w.jpg" style="zoom:80%;" />

（1）YOLOv3中只有卷积层（实际上是Conv + Bn + Leaky ReLU，没有全连接层），通过**调节卷积步长**控制输出特征图的尺寸。所以对于输入图片尺寸没有特别限制。

（2）YOLOv3借助了金字塔特征图的思想，小尺寸特征图用于检测大尺寸物体，而大尺寸特征图用于检测小尺寸物体。特征图输出为$N*N*(3*(4+1+80))$，$N$为输出特征图点格数，一共3个anchors，每个框有4维预测框的参数$t_x,t_y,t_w,t_h$，1维预测框置信度，80维物体类别。

（3）YOLOv3总共输出3个特征图。第一、二、三个特征图（从上到下）分别对应与输入图像下采样32倍、16倍、8倍（比如输入是256\*256\*3，输出分别为8\*8\*3、16\*16\*3、32\*32\*3）。从最开始的输入图像（256\*256\*3），经过Darknet53（无全连接层），再经过YOLOBlock生成的特征图被当做两用。一是经过3x3Conv和1x1Conv输出；二是经过1x1Conv之后再进行上采样，再与中间的一层**在通道维度上进行拼接（concate）**，输出中间的一层特征图。最后一层同样做法。（这其实就是FPN的思想）

<img src="https://pic1.zhimg.com/80/v2-48085568c7e30a0a1c6d07f1f418a7a9_720w.jpg" style="zoom: 80%;" />

FPN是自顶向下的，将高层的特征信息通过**上采样**的方式进行传递融合，得到进行预测的特征图。



**FPN（Feature Parymid Network）**

​		由于单stage物体检测算法中，Backbone的最后一个stage的stride通常是32，导致输出的特征图分辨率是输入图片分辨率的1/32，太小，不利于物体检测，因此单stage的物体检测算法，一般会将最后一个stage的MaxPooling去掉或者将stride为2的conv改为stride为1的conv，以增大最后一个分辨率。

​		后来研究发现，单stage物体检测算法中，无法用单一stage的特征图同时有效的表征各个尺度的物体，因此，后来物体检测算法，就逐渐发展为利用不同stage的特征图，形成特征金字塔网络（feature parymid network），表征不同scale的物体，然后再基于特征金字塔做物体检测，也就是进入了FPN时代。

<img src="https://pic1.zhimg.com/80/v2-aeec87a0b2e1494fe109062c946d2368_720w.jpg" alt="img" style="zoom:50%;" />

**1）Backbone生成特征阶段**

​		计算机视觉任务一般都是基于常用预训练的Backbone，生成抽象的语义特征，再进行特定任务微调。物体检测也是如此。

​		Backbone生成的特征，一般按stage划分，分别记作C1、C2、C3、C4、C5、C6、C7等，其中的数字与stage的编号相同，代表的是分辨率减半的次数，如C2代表stage2输出的特征图，分辨率为输入图片的1/4，C5代表，stage5输出的特征图，分辨率为输入图片的1/32。

**2）特征融合阶段**

​		这个是FPN特有的阶段，FPN一般将上一步生成的不同分辨率特征作为输入，输出经过融合后的特征。输出的特征一般以P作为编号标记。如FPN的输入是，C2、C3、C4、C5、C6，经过融合后，输出为P2、P3、P4、P5、P6。这个过程可以用数学公式表达：

![[公式]](https://www.zhihu.com/equation?tex=%5C%5C+P_%7Bi%7D%E3%80%81P_%7Bi%2B1%7D%E3%80%81...%E3%80%81P_%7Bi%2Bn%7D%3Df%28C_%7Bi%7D%E3%80%81C_%7Bi%2B1%7D%E3%80%81...%E3%80%81C_%7Bi%2Bn%7D%29)

**3）检测头输出bounding box**
FPN输出融合后的特征后，就可以输入到检测头做具体的物体检测。



​		物体检测性能提升，一般主要通过数据增强、改进Backbone、改进FPN、改进检测头、改进loss、改进后处理等6个常用手段。其中FPN自从被提出来，先后迭代了不少版本。大致迭代路径如下图：

<img src="https://pic1.zhimg.com/80/v2-c69ff92c9213764f7dcb30f600e2e408_720w.jpg" alt="img" style="zoom:67%;" />

1）是SSD

2）是FPN。其中YOLOv3的FPN与图中前两个有所不同。

![img](https://pic4.zhimg.com/80/v2-b9c4530ad727803540d0b17775b25227_720w.jpg)

3）是PANet

4）是ASFF，BiFPN

FPN参考 https://zhuanlan.zhihu.com/p/148738276



（4）Concat操作和add操作的区别：add操作源于ResNet的思想，$out = f(x) + x$；而Concat操作源于DenseNet，将特征图按照通道维度进行拼接。

（5）上采样（UpSample）：作用是将小尺寸（这里就是指像素小）特征图通过差值的方法，生成大尺度图像。例如使用**最近邻插值**（YOLOv3采用的是最近邻插值）或者**双线性插值**。上采样不改变特征图的通道数。

​		YOLOv3的整个网络，吸取了Resnet、Densenet、FPN的精髓，可以说是融合了目标检测当前业界最有效的全部技巧。



2.YOLOv3输出特征图解码（前向过程）

​		YOLOHead输出3种不同尺度的特征图，比如以输入图像256x256x3为例，第一、二、三层输出分别为8x8x255、16x16x255、32x32x255（注意这里的255还是通道数，而且这个255是人为设计的，为了就是对应上后面的3x85）。YOLOv3设计为每个格子都配置3种不同的anchor，所以最后三个尺度的特征图reshape成8x8x3x(4+1+80)、16x16x3x(4+1+80)、32x32x3x(4+1+80)，主要reshape成这样后面更容易操作。



**先验框**

YOLOv2开始引入anchor机制后，不再与YOLOv1那样直接预测框的坐标，而是预测**偏移值**。通过学习偏移值，就可以通过网络原始定的anchor box坐标经过**线性回归**微调去逐渐靠近GT。

**为什么说是微调？**因为当引入的proposal与GT相差较小时，即IOU很大时，可以认为这中变换是线性的，就可以使用**线性回归（Y=WX）**来建模对窗口进行微调。当Proposal与GT离得较远，那就是复杂的非线性问题了。



YOLOv3沿用了v2中关于先验框的技巧，并且使用k-mean聚类算法对数据集中的标签框进行聚类，得到类别中心点的9个框，作为先验框（wxh）。

注意：先验框只与检测框的w, h有关，与x, y无关。



**检测框（预测框）解码**
$$
b_x = \sigma(t_x) + c_x \\
b_y = \sigma(t_y) + c_y \\
b_w = p_w e^{t_w} \\
b_h = p_h e^{t_h}
$$
网络预测出$(t_x, t_y, t_w, t_h)$后，根据这个公式计算出最终输出的$(b_x, b_y, b_w, b_h)$。



如下图所示，$\sigma(t_x), \sigma(t_y)$是基于矩形框中心点左上角格点坐标的偏移量，$\sigma$是**激活函数**，论文中作者使用**sigmoid**。$(p_w, p_h)$是先验框的宽、高。通过上述公式，计算出实际预测框的宽高$(b_w, b_h)$。

<img src="https://pic3.zhimg.com/80/v2-758b1df9132a9f4b4e0c7def735e9a11_720w.jpg" style="zoom:50%;" />

对比这个图，虚线框才是代码中anchor的位置和大小。

<img src="C:\Users\87995\AppData\Roaming\Typora\typora-user-images\image-20211012152136427.png" alt="image-20211012152136427" style="zoom:50%;" />

其中

$(t_x, t_y)$预测的坐标偏移值，再经过sigmoid变换。

$(t_w, t_h)$尺度缩放。

$(b_x, b_y)$是最终预测出的bbox的中心点坐标。通过计算得出。

$(b_w, b_h)$是最终预测出的bbox的宽和长。通过计算得出。

$(p_w, p_h)$是anchor box经过缩小之后，映射到feature中的宽和高，是通过原始坐标/stride后得到的，其中stride代表每个像素对应原图的大小。

**网络学习的是$(t_x, t_y, t_w, t_h)$这四个offset。**训练时用$(g_x, g_y, g_w, g_h)$替代$(b_x, b_y, b_w, b_h)$计算出t，然后对t作线性回归。



YOLOv2之后 预测框怎么做回归 https://blog.csdn.net/zhicai_liu/article/details/113631706



**Ground Truth的计算**

既然YOLOv3网络预测的是偏移值，那么在计算损失时，也是按照**偏移值**计算损失。现在我们有预测的值，还需要真实标签GT的偏移值，用于计算损失的GT按照以下公式得到：
$$
t_x = G_x - C_x \\
t_y = G_y - C_y \\
t_w = log(G_w / P_w) \\
t_h = log(G_h / P_H)
$$
这个公式其实是由边框预测公式得到的。看最后两个公式，Ground Truth的$t_w，t_h$是log尺度缩放到对数空间了，所以在预测时需要指数回来。这就是答案。



**为什么在计算Ground Truth的tw，th时需要缩放到对数空间？**

tw和th是物体所在边框的长宽和anchor box长宽之间的比率。不直接回归bounding box的长宽，而是**为避免训练带来不稳定的梯度**，将尺度缩放到对数空间。 如果直接预测相对形变tw 和 th，那么要求tw,th>0，因为框的宽高不可能是负数，这样的话是在做一个有不等式条件约束的优化问题，没法直接用SGD来做，所以先取一个对数变换，将其不等式约束去掉就可以了。



**对于某个ground truth的框，究竟是哪个anchor负责匹配它？**

与yolov1一样，对于训练图片中的gt，若其中心点落在某个grid cell内，那么该cell中的3个anchor box负责预测它。具体是哪个anchor预测它，需要在训练中确定，即由那个与gt的iou最大的anchor box预测它，剩余两个anchor box不与该gt框匹配。

与gt框匹配的anchor box计算坐标误差、置信度误差（此时的target为0）以及分类误差，而其他的anchor box只计算置信度误差（此时target为0）。



**检测置信度解码**

置信度在85维中占固定一维，由sigmoid函数解码即可，解码之后的数值区间在[0, 1]中。



**类别解码**

COCO数据集有80个类别，所以类别数在85维输出中占了80维，每一维独立代表一个类别的置信度。YOLOv3使用多个（Logistic分类器）sigmoid代替了YOLOv2中的softmax，取消了类别之间的互斥，可以使网络更加灵活。



3.训练策略与损失函数（反向过程）

**训练策略**

确定正负样本。

（1）预测框一共分为3中情况，正样本、负样本、忽略样本。

（2）先说明一下，从YOLOHead输出的特征图一共有3种，每个大小的特征图上有3中不同大小的anchor。以8x8，16x16，32x32这三种特征图来算，一共有8 × 8 × 3 + 16 × 16 × 3 + 32 × 32 × 3 = 4032个anchor。

正样本：在训练的时候，每取一个gt，就与这4032个anchor作iou，iou最大的anchor作为正样本。如果第一个gt已经匹配了一个anchor了，再拿出一个gt，与剩下4031个anchor作iou，同样选择最大的那个作为正样本。

负样本：与全部gt的iou都小于0.5的anchor，则为负样本。

忽略样本：除去正样本外，与任何一个gt的iou大于0.5的样本，为忽略样本。



**loss函数**

正样本的损失：预测框损失（通过gt计算出的t_x, t_y, t_w, t_h）、置信度损失（置信度标签为1）、类别损失（类别标签对应类别为1，其他类别为0）。

负样本的损失：置信度标签为0，只有置信度损失。

忽略样本：不产生任何损失。



定位损失：

![](https://pic1.zhimg.com/80/v2-6e2d955d2822127c0b1f72f6080c1dd0_720w.png)

置信度损失：

![](https://pic2.zhimg.com/80/v2-c1ea1dd997ba1a3dce1e90262064e96d_720w.png)

分类损失：

![](https://pic4.zhimg.com/80/v2-bc512cd729141c50d9f136934b78b78f_720w.png)



![[公式]](https://www.zhihu.com/equation?tex=%5Clambda_%7Bcoord%7D+%3D+2+-+w%2Ah) ,**加大对小框的损失**，更好的检测出小目标。
S²为grid cell数，即13²/26²/52²。
B为anchor数，即3。
![[公式]](https://www.zhihu.com/equation?tex=1_%7Bij%7D%5E%7Bobj%7D) 判断第i个网格中的第j个box是否负责这个object。
![[公式]](https://www.zhihu.com/equation?tex=1_%7Bij%7D%5E%7Bnoobj%7D) 为不含object的box的confidence预测。



**为什么会有忽略样本？**

​		忽略样例是Yolov3中的点睛之笔。由于Yolov3使用了多尺度特征图，不同尺度的特征图之间会有重合检测部分。比如有一个真实物体，在训练时被分配到的检测框是特征图1的第三个box，IOU达0.98，此时恰好特征图2的第一个box与该ground truth的IOU达0.95，也检测到了该ground truth，如果此时给其置信度强行打0的标签，网络学习效果会不理想。

​		如果给全部的忽略样例置信度标签打0，那么最终的loss函数会变成$Loss_{obj}和Loss_{noobj}$的拉扯，不管两个loss数值的权重怎么调整，或者网络预测趋向于大多数预测为负例，或者趋向于大多数预测为正例。而加入了忽略样例之后，网络才可以学习区分正负例。



**优化器**

作者在文中没有提及优化器，Adam，SGD等都可以用，github上Yolov3项目中，大多使用Adam优化器。



**语义分割**

**UNet网络结构**

​		UNet发表于2015年，属于FCN的一种变体。UNet的初衷是为了解决生物医学图像方面的问题，由于效果确实很好后来也被广泛的应用在语义分割的各个方向，比如卫星图像分割，工业瑕疵检测等。

​		UNet跟FCN一样都是Encoder-Decoder结构，结构简单但很有效。Encoder负责提取特征，你可以将自己熟悉的各种特征提取网络放在这个位置。由于在医学方面，样本收集较为困难，作者为了解决这个问题，应用了图像增强的方法，在数据集有限的情况下获得了不错的精度。

<img src="https://pic3.zhimg.com/80/v2-39073bacc426f0e464b53336c83e19da_720w.jpg" alt="img" style="zoom:80%;" />

UNet的网络下采样（提取特征）有5层（不要管为什么是5层，而不是说4层或6层，问就是炼丹）。

（1）蓝色和白色方框表示特征；

（2）蓝色箭头表示用3x3卷积提取特征；

（3）灰色箭头表示 skip-connection ，用于特征融合；

（4）红色箭头表示池化pooling，用于降维；

（5）绿色箭头表示上采样upsample，用于恢复维度；

（6）在右上角有个青色的箭头，表示输出结果。



**Encoder**

​		Encoder有卷积操作和下采样操作组成，文中所用的卷积结构统一为3x3的卷积核，padding为0，stride为1。计算一下，从刚输入的图像 572x572x1 ，用N=572验证下一层是否是570。
$$
N' = \frac{N - F + 2*P}{S} + 1 \\
N' = \frac{572 - 3 + 2*0}{1} + 1 \\
N' = 570
$$
每次卷积操作都是

```python
nn.Sequential(nn.Conv2d(in_channels, out_channels, 3),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True))
```

一个Conv + BN + ReLU，之后再用一个size=2，stride=2的max pooling。

<img src="https://pic2.zhimg.com/80/v2-20ca8abca1f5c73691b3c53e1ca1b68d_720w.jpg" alt="img" style="zoom:50%;" />

```python
nn.MaxPool2d(kernel_size=2, stride=2)
```

上面的步骤重复5次，最后一次没有max pooling，直接得到feature map送入Decoder。可以看到UNet网络最下面一行，28x28蓝色框中没有打虚线，即没有进行最大池化，直接上采样，然后进行通道拼接。



**Decoder**

​		feature map 经过 Decoder 恢复原始分辨率，该过程除了卷积比较关键的步骤就是 upsampling 与 skip-connection。Upsampling 上采样常用的方式有两种：1.FCN中介绍的**反卷积**；2.**插值**。





**反卷积**









**双线性插值（bilinear）**

（1）将一个图像放大$\alpha$倍，依次遍历放大图像每个像素点坐标$(x', y')$，通过$\lfloor \frac{x'}{\alpha}, \frac{y'}{\alpha} \rfloor$计算该点在原图中的最近4个点的坐标--$(x, y), (x+1, y), (x, y+1), (x+1, y+1)$这4个点。

<img src="https://img.geek-docs.com/opencv/opencv-examples/bli_fig.png" alt="img" style="zoom: 50%;" />

按照图中，将一个2x2大小的图，放大2倍，变为4x4大小的图片。依次遍历4x4大小图片中的每个点的坐标，比如第一个点$(0, 0)$，通过$\lfloor \frac{x'}{\alpha}, \frac{y'}{\alpha} \rfloor$计算结果仍然是$(0, 0)$。然后在原图中找最近的4个点，坐标分别为$(0, 0), (0, 1), (1, 0), (1, 1)$，在原图中分别取这4个位置上的点的值作线性插值。

（2）如果取权值，我这里通俗的讲，就是(1 - 距离)。

（3）插值为$I(x', y') = (1 - d_x)(1 - d_y)I(x, y) + d_x(1 - d_y)I(x + 1, y) + (1 - d_x)d_yI(x, y + 1) + d_x d_y I(x + 1, y + 1)$，用计算的值填补放大图片上坐标为$(x', y')$的点。



**双线性插值**代码实现

```python
```



**介绍一下YOLOv3**

（1）darknet53作为backbone，其中使用了残差结构，与resnet101或者resnet152精度差不多；

（2）抛弃了pooling池化操作，使用步长为2的卷积进行下采样；

（3）特征融合方面，为了加强小目标的检测，引入了类似于FPN的多尺度特征融合（只不过FPN中是elementwise add，而YOLOv3中的FPN是按通道concat），经过上采样后与前面的层进行concat，使得浅层特征和深层特征的融合，使得YOLOv3在小目标的精度上有了很大的提升。

（4）YOLOv3核心思想：将输入图像经过backbone，neck，提取特征后，通过head输出为SxS的网格，物体中心落在哪一个网格内，那么这个网格就负责预测该物体。包括坐标，置信度，类别。



**介绍Faster RCNN**

<img src="https://pic4.zhimg.com/80/v2-e64a99b38f411c337f538eb5f093bdf3_720w.jpg" alt="img" style="zoom: 80%;" />

Faster RCNN是基于候选区域的双阶段检测器，总共分为4个部分：

（1）Conv layers（VGG16）。作为一种CNN网络目标检测方法，Faster RCNN首先使用一组基础的conv + relu + pooling层提取image的feature maps。该feature maps被共享用于后续RPN层和全连接层。

Conv层：conv + relu + pooling。其中conv是kernel=3x3，stride=1，padding=1。

pooling层：kernel_size=2，stride=2，padding=0。



（2）Region Proposal Networks。RPN网络用于生成region proposals。该层通过softmax判断anchors属于positive或者negative，在利用bounding box regression修正anchors获得精确的proposals。

<img src="https://pic3.zhimg.com/80/v2-1908feeaba591d28bee3c4a754cca282_720w.jpg" alt="img" style="zoom:67%;" />



RPN网络分为两条线：

上面一条：通过softmax分类anchor获得positive和negative分类（有无object）；

下面一条：计算对于anchors的bounding box regression的偏移量，以获得精确的proposal。

最后一个Proposal负责综合positive anchors和对应的bounding box regression偏移量，获取最终的proposals，同时剔除太小和超出边界的proposals。其实整个网络到了Proposal Layer这里，就完成了相当于目标定位的功能。最后对剩余的positive anchors进行NMS。



**其实RPN最终就是在原图尺度上，设置了密密麻麻的候选Anchor。然后用cnn去判断哪些Anchor是里面有目标的positive anchor，哪些是没目标的negative anchor。所以，仅仅是个二分类而已！**



RPN中anchors的生成

其中每行的4个值$(x_1, y_1, x_2, y_2)$表矩形左上和右下角点坐标。9个矩形共有**3种形状**，长宽比为大约为$width:height \in \{1:1, 1:2, 2:1\}$。



（3）Roi Pooling。该层收集输入的feature maps和proposals，综合这些信息后提取proposal feature maps，送入后续全连接层判定目标类别。

（4）Classification。利用proposal feature maps计算proposal的类别，同时再次bounding box regression获得检测框的最终的精确位置。

<img src="https://pic3.zhimg.com/80/v2-c0172be282021a1029f7b72b51079ffe_720w.jpg" alt="img" style="zoom:67%;" />

Faster RCNN https://zhuanlan.zhihu.com/p/31426458



#### Attention ####

---

理解self-attention

https://www.zhihu.com/question/325839123

某位知友解释：

解释地太清楚啦。我画蛇添足地做个比喻帮助理解：
假如一个男生B，面对许多个潜在交往对象B1，B2，B3...，他想知道自己谁跟自己最匹配，应该把最多的注意力放在哪一个上。那么他需要这么做：
1、他要把自己的实际条件用某种方法表示出来，这就是Value；
2、他要定一个自己期望对象的标准，就是Query；
3、别人也有期望对象标准的，他要给出一个供别人参考的数据，当然不能直接用自己真实的条件，总要包装一下，这就是Key；
4、他用自己的标准去跟每一个人的Key比对一下（Q*K），当然也可以跟自己比对，然后用softmax求出权重，就知道自己的注意力应该放在谁身上了，有可能是自己哦。



**SENet**

<img src="https://pic4.zhimg.com/80/v2-77affb3d6037ab0fa4f564f30c38031b_720w.jpg" alt="img" style="zoom: 67%;" />

x为输入进来的特征，$F_{tr}(.,\theta)$卷积层提取x特征，将通道数从$c_1$转变为$c_2$。将此时的特征图称为$u$。



**Squeeze**

$F_{sq}(.)$就是S操作。对特征图u，大小为HxWxC_2。对每个通道进行global avgpooling，得到1x1xC_2的特征图。



**Excitation**

这个模块简单点就是全连接1 + ReLU + 全连接2 + Sigmoid。

第一个全连接层，将通道数从$c_2$变为$\frac{c_2}{r}$，r为缩放比例；

然后ReLU激活；

再经过一个全连接层，将通道数从$\frac{c_2}{r}$变为$c_2$，还原通道；

最后经过一个Sigmoid层，得到各个通道的激活值（0-1）。



**Scale**

得到通道注意力权重要想能够与原特征图x相乘，size必须相同。所以这里将Excitation之后的注意力权重（是一个长度为$c_2$一维向量，因为刚通过全连接层），先view成1x1xC_2，然后再放大为u的宽高，通道数相同，进行相乘。



```python
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
```



**CBAM**

​		论文（2018年）提出了一种轻量的注意力模块( CBAM，Convolutional Block Attention Module )，可以在通道和空间维度上进行 Attention 。论文在 ResNet 和 MobileNet 等经典结构上添加了 CBAM 模块并进行对比分析，同时也进行了可视化，发现 CBAM 更关注识别目标物体，这也使得 CBAM 具有更好的解释性。

<img src="https://img-blog.csdnimg.cn/20210310210027842.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JvYWRkZA==,size_16,color_FFFFFF,t_70#pic_center" style="zoom:67%;" />

分为两个部分介绍

**Channel Attetion Module (CAM)**

<img src="https://img-blog.csdnimg.cn/20210310210545901.png#pic_center" style="zoom: 67%;" />

对输入特征图F（HxWxC）用maxpooling和avgpooling分别进行池化，得到两个1x1xC的特征图。再通过一个两层的MLP，第一层的神经元个数为C/r（r为比例），第二层神经元个数为C，这两层神经网络是共享权重的。将MLP输出的特征进行element wise sum操作，再通过一个sigmoid激活操作，得到Channel Attention Module。将CAM和一开始的特征图F作element wise 乘操作，生成下一个SAM输入。



**Spatial Attention Module (SAM)**

<img src="https://img-blog.csdnimg.cn/20210310210627977.png#pic_center" style="zoom:67%;" />

将CAM输出的结果作为SAM的输入。假设输入特征图F为HxWxC，同样用一个maxpooling和avgpooling（我猜pooling层的大小应该为1x1xC），按照通道进行池化，得到两个池化特征图，分别是最大池化核平均池化的结果，大小均为HxWx1。然后将它们按通道拼接，生成一个HxWx2的特征图，再对这个特征图用7x7卷积做操作，生成大小为HxWx1的特征图（这里代码中通道7x7卷积之后H和W还能保持不变，我猜是padding的原因。然后一看论文，padding="same"，即padding=3，我计算一下$\frac{H - 7 + 2 * 3}{1} + 1 = H$，的确没变）。最后再经过一个sigmoid层，生成Spatial Attention Module，将SAM和输入的F相乘，得到最终特征。



原文链接：https://blog.csdn.net/Roaddd/article/details/114646354



**ECANet**





#### 模型集成 ####

---

Bagging是Bootstrap Aggregating的英文缩写，不要误认为bagging是一种算法。Bagging和Boosting都是ensemble learning中的学习框架，代表着不同的思想。

- boosting派系，它的特点是各个弱学习器之间有依赖关系。
- bagging流派，它的特点是各个弱学习器之间没有依赖关系，可以并行拟合。



**bagging**

​		随机采样(bootsrap)就是从我们的训练集里面采集固定个数的样本，但是每采集一个样本后，都将样本放回。也就是说，之前采集到的样本在放回后有可能继续被采集到。对于我们的Bagging算法，一般会随机采集的样本数和训练集样本数一样，都为m。这样得到的采样集和训练集样本的个数相同，但是样本内容不同。如果我们对有m个样本训练集做T次的随机采样，则由于随机性，T个采样集各不相同。

`注意到这和GBDT的子采样是不同的。GBDT的子采样是无放回采样，而Bagging的子采样是放回采样。`

​		对于一个样本，它在某一次含m个样本的训练集的随机采样中，每次被采集到的概率是$\frac{1}{m} $，不被采集到的概率为1-$\frac { 1 }{ m } $，如果m次都没被采样到的概率为$({1 - \frac{1}{m})}^{m}$，当$m\rightarrow \infty  $时，$lim_{m \rightarrow \infty }{(1 - \frac{1}{m})^m}$约定于$ \frac{1}{e}$=0.368，也就是说，在bagging的每轮随机采样中，训练集中大约有36.8%的数据没有被采样集采集中。自助采样会改变数据的初始分布导致引入估计偏差。

​		对于这部分大约36.8%的没有被采样到的数据，我们常常称之为袋外数据(Out Of Bag, 简称OOB)。这些数据没有参与训练集模型的拟合，因此可以用来检测模型的泛化能力。

​		bagging对于弱学习器没有限制，这和Adaboost一样。但是最常用的一般也是决策树和神经网络。

​		bagging的集合策略也比较简单，对于分类问题，通常使用简单投票法，得到最多票数的类别或者类别之一为最终的模型输出。对于回归问题，通常使用简单平均法，对T个弱学习器得到的回归结果进行算术平均得到最终的模型输出。

​		由于Bagging算法每次都进行采样来训练模型，因此泛化能力很强，对于降低模型的方差很有作用。当然对于训练集的拟合程度就会差一些，也就是模型的偏倚会大一些。



相对于Boosting系列的Adaboost和GBDT，bagging算法要简单的多。这里对bagging**总结**。

假设输入为样本集D=$\{(x_1,y_1),(x_2,y_2),...,(x_m,y_m)\}$，弱学习器算法，弱分类器迭代次数为T。

（1）对t=1, 2, ..., T。对训练集进行第t次随机采样，每次采样m次，得到包含m个样本的采样集$D_t$。用采样集$D_t$训练第t个弱学习器$G_t(x)$。

（2）如果是分类算法，则T个弱学习分类器投出最多票数的类别或类别之一为最终类别。如果是回归算法，T个弱学习器得到的回归结果进行算数平均得到的值作为最终的模型输出。



**随机森林（Random Forest）**

RF是Bagging算法的进化版，它的思想仍然是bagging，但是进行了独有的改进。



首先，RF使用了CART决策树作为弱学习器，这让我们想到了梯度提升树GBDT。

第二，在使用决策树的基础上，RF对决策树的建立做了改进，对于普通的决策树，会在节点上所有的n个样本特征中选择一个最优的特征来做决策树的左右子树划分，但是RF通过随机选择节点上的一部分样本特征，这个数字小于n，假设为$n_{sub}$，然后在这些随机选择的$n_{sub}$个样本特征中，选择一个最优的特征来做决策树的左右子树划分。这样进一步增强了模型的泛化能力。

如果$n_{sub}=n$，则此时RF的CART决策树和普通的CART决策树没有区别。$n_{sub}$越小，则模型越健壮，当然此时对于训练集的拟合程度会变差。也就是说$n_{sub}$越小，模型的方差会减小，但是偏差会增大。在实际案例中，一般会通过交叉验证调参获取一个合适的$n_{sub}$值。



除了上面两点，RF和普通的bagging算法没有什么不同，下面简单**总结**下RF的算法。

输入为样本集$D={(x_1,y1),(x_2,y_2),...(x_m,y_m)}$ ，弱分类器迭代次数T。输出为最终的强分类器f(x)。

（1）对于t=1, 2, ...,T。对训练集进行第t次随机采样，每次采样m次，得到包含m个样本的采样集$D_t$。

（2）用采样集$D_t$训练第t个决策树模型$G_t(x)$。在训练决策树模型的节点的时候，在**当前节点**上所有的样本特征中**选择一部分**样本特征，在这些随机选择的部分样本特征中选择一个最优的特征来做决策树的左右子树划分。

（3）如果是分类算法，则T个弱学习分类器投出最多票数的类别或类别之一为最终类别。如果是回归算法，T个弱学习器得到的回归结果进行算数平均得到的值作为最终的模型输出。



参考 https://blog.csdn.net/leadai/article/details/79907417



