### B超项目 ###

---

图像分辨率：指图像中存储的信息量，是每英寸中的像素点个数，即像素密度（像素个数/英寸）。

屏幕分辨率：屏幕分辨率就是屏幕上显示的像素个数,一般是以（水平像素数×垂直像素数）表示。常说的2k屏，4k屏就是这个意思。

单通道图片：俗称灰度图，图片由二维矩阵构成，每个像素点用一个值表示颜色，它的像素值在0到255之间，0是黑色，255是白色，中间值是一些不同等级的灰色。

<img src="C:\Users\87995\AppData\Roaming\Typora\typora-user-images\image-20220224211544676.png" alt="image-20220224211544676" style="zoom: 67%;" />

黑白图片：二值图像（黑白图像）：每个像素点只有两种可能，0和1，0代表黑色，1代表白色。数据类型通常为1个二进制位。

![image-20220224211500856](C:\Users\87995\AppData\Roaming\Typora\typora-user-images\image-20220224211500856.png)

三通道图片：可以是彩色图，可以是灰度模式的图像。三通道分别指RGB(红，绿，蓝)通道。将通道红绿蓝三通道比作三个手电筒，那么RGB的值就是三个手电筒的灯光亮度。

如果R,G,B三个通道的亮度一致，即R=G=B，那么这样的图片就是灰度模式的图片。如果这三个值不相等，那么就是彩色图片。因此三通道的彩色照片变成灰度模式的图的方式就是，将R，G，B三个通道值改成一样。下面的图片的R,G,B三个通道值一致。

<img src="C:\Users\87995\AppData\Roaming\Typora\typora-user-images\image-20220224211739022.png" alt="image-20220224211739022" style="zoom:80%;" />

#### 数据预处理 ####

---

**数据集分布有什么特点**

图像的分布归根结底是像素值服从某种分布。



在B超项目中，数据集分布特点：

1.采集的B超图片大小参差不齐；

2.灰度RGB三通道图片；

3.分为左右两部分，两部分分别位于人体左右对称部位，如果一边是劳损区域，另一边就是非劳损区域（这个本来用途是用作医生临床检查的辅助）；

4.第3点中的对称，并不是出传统意义上的轴对称，是平移对称；

5.每张B超图片四周都有黑边。

6.每半边图像中绝大多数包含的真实框在3~5个之间。



**数据集分布不均衡**

1.从数据角度：

扩充数据集。

（1）欠采样。通过减小丰富类的样本数量来平衡数据集，当数据量足够多时采用此方法。通过保存所有稀有类样本，并在丰富类别中随机选择与稀有类别样本相等数量的样本，可以检索平衡的新数据集以进一步建模。

（2）过采样。当样本数量不足时采用过采样，它尝试通过增加稀有样本数量来平衡数据集。通过使用重复、自举或合成少数类的过采样方法（比如SMOTE）来生成新的稀有样本。

注意到欠采样和过采样这两种方法相比而言，都没有绝对的优势。这两种方法的应用取决于它适用的用例和数据集本身。另外将过采样和欠采样结合起来使用也是成功的。

过采样后的数据集中包含大量的重复样本，训练出来的模型可能会过拟合；而欠采样由于筛除了一些数据样本，导致模型只学到了一部分知识，模型欠拟合。



（3）对数据集进行重采样。



（4）人造数据SMOTE（Synthetic Minority Over-sampling Technique 人工少数类过采样算法）。

首先要定一个feature空间（这一点在深度学习时就让人费解了，feature空间还没提出来，该如何定呢）；

对每一个minority的类样本，在feature空间里，找到K个最紧邻；

对每一个最紧邻到目标样本计算一个方向vector，然后乘以（0,1）之间的一个比例，然后叠加到样本的各个feature维度上。这样就产生了一个新样本。

SMOTE https://blog.csdn.net/seavan811/article/details/46879783

图解 https://blog.csdn.net/haoji007/article/details/106166305/



2.从算法角度：

改变[分类算法](https://so.csdn.net/so/search?q=分类算法&spm=1001.2101.3001.7020)。



3.从评价指标角度：

（1）谨慎选择AUC作为评价指标。对于数据极端不平衡时，可以观察观察不同算法在同一份数据下的训练结果的precision和recall，这样做有两个好处，一是可以了解不同算法对于数据的敏感程度，二是可以明确采取哪种评价指标更合适。针对机器学习中的数据不平衡问题，建议更多PR(Precision-Recall曲线)，而非ROC曲线，具体原因画图即可得知，如果采用ROC曲线来作为评价指标，很容易因为AUC值高而忽略实际对少两样本的效果其实并不理想的情况。

（2）不要只看Accuracy。Accuracy可以说是最模糊的一个指标了，因为这个指标高可能压根就不能代表业务的效果好，在实际生产中，我们可能更关注precision/recall/mAP等具体的指标，具体侧重那个指标，得结合实际情况看。



*目标检测中的不平衡问题

Focal Loss



详细 https://blog.csdn.net/pnnngchg/article/details/85728231

整理 https://www.cnblogs.com/charlotte77/p/10455900.html



接下来说说做的项目数据不平衡的问题。
		项目中的数据是二类的，两种类型的数据相差两个数量级，训练集中一个是八万左右，另一个是一百多万。测试集中两者数量无法获知。
		不采用上面介绍的方法进行数据平衡的情况下（训练集按9:1分为训练集和验证集），最佳阈值在0.3左右，各模型的accuracy在90%左右，f1值在0.65左右。训练集上也查不多，只比验证集低一点。

​		为了处理不平衡的问题，我先用了最简单的方法，就是同时欠采样和过采样。为了将正例和负例平衡到统一的数量级，对负例以0.8的比例降采样，对正例以3倍数量过采样。在将训练集按9:1分为训练集和验证集。这时候的验证集上的结果有了很大的提升，f1值直接提升到了0.9左右，阈值向右靠到了0.45。但是在测试集上，效果却大大下降了（具体是多少我忘记记了。。但差的挺大的）。

​		跟同学讨论了一下，觉得可能是**测试集的数据与原始训练集的数据分布应该是一样的**，就是**都是不平衡的一个分布**，如果在训练集更改了数据的分布，那么训练出来的模型拟合的就是平衡的数据分布，在去预测测试集的不平衡数据的时候，就会有较大差距了。如果是这样的话，不对数据进行平衡处理才是正确的做法。。所以后面我就没有再对数据用上面的方法进行平衡处理了。。



​	**结论：**在选择采样法事需要注意一个问题，如果你的实际数据是数据不平衡的，在训练模型时发现效果不好，于是采取了采样法平衡的数据的比例再来进行训练，然后去测试数据上预测，这个时候算法的效果是否会有偏差呢？此时你的训练样本的分布与测试样本的分布已经发生了改变，这样做反而会产生不好的效果。**在实际情况中，我们尽可能的需要保持训练和测试的样本的概率分布是一致的**，如果测试样本的分布是不平衡的，那么训练样本尽可能与测试样本的分布保持一致，哪怕拿到手的是已经清洗和做过预处理后的平衡的数据。



**数据增强**

1.一定程度内的随机旋转、平移、缩放、裁剪、填充、左右上下翻转。

2.对图像中的像素添加噪声扰动。

3.颜色变换。在图像的RGB颜色空间上添加增量。

4.改变图像的亮度、清晰度、对比度、锐度等。

5.除此之外，还有采样算法SMOTE，生成对抗网络GAN等都可以进行图像扩充。

图像增强 https://blog.csdn.net/Celibrity/article/details/106297733



**B超项目中的数据预处理**

对图片的处理：

1.去除图片四周黑边。opencv中是(h, w)。具体做法是从左往右找到w的1/4处，从上往下遍历，找到像素值大于某个阈值的所有点的索引，取第1个点的h索引；同理，从上往下找到h的3/4处，在从左往右遍历，找到像素值大于某个阈值的所有点的索引，取第一个点的w索引。以这两个索引位置作为边界，剔除黑边，同时适当平移标签。

2.从中间竖直分为左右两半，只取包劳损区域的半边，其标签也适当进行平移。



对标签的处理：

1.两种格式的标签，第一种红色矩形框的处理方法。将原灰度图像3通道RGB转换色彩空间到YCrCb，分离Y, Cr, Cb三个通道并取其中的Cr通道（为什么要取Cr这个通道呢？Y, Cr, Cb分别代表亮度、色度和饱和度。Cr反映了RGB输入信号红色部分与RGB信号亮度值之间的差异。而Cb反映的是RGB输入信号蓝色部分与RGB信号亮度值之间的差异。）再在Cr通道上进行轮廓检测（这里可以加上一点，对每个轮廓计算面积，面积太小的去除），然后对这些轮廓取最小外接矩形。最终将图像名、图像大小、标签信息等按照Pascal Voc的标签格式封装成一个xml的Annotation。

2.两种格式的标签，第二种json格式处理方法。这类格式的标签是由于医生使用的是labelme工具得到的标签，一般是图像分割格式的标签。这里我们需要把json格式转换成xml格式，用于目标检测。json文件中有一个表示形状的键shape_type，包括矩形、椭圆和扇形。矩形我们直接取左上和右下作为xml格式的标签，而对于椭圆和扇形，我们用最小外接矩形的左上角和右下角作为最终的标签。



#### 检测网络 ####

---

B超项目中的图像识别部分由两个网络组成，分别是一个目标检测网络和一个二分类网络。目标检测网络用来初步提取B超图像中的明显的劳损区域，而分类网络用来进一步对这些劳损区域进行过滤。



**5折交叉验证**

在训练模型之前，将数据集划分成**训练集**和**测试集**

再对训练集进行5折交叉验证，将其中4份作为**训练集1,2,3,4**，另一份作为**测试集1,2,3,4**（这里可以叫验证集）。



设置n批不同超参数（每一批超参数包括学习率，训练轮次等）

选择其中一批超参数，用训练集1,2,3,4，训练4个模型，并在测试集1,2,3,4上进行评估（准确率和损失值）。这4个模型在**测试集1,2,3,4**上的平均误差作为此**超参数**模型下的泛化误差。

这样我们就得到了n个不同模型（超参数不同，模型就不同），我们选择最优的那批超参数，作为最终的模型参数，在**训练集**上进行训练，并在**测试集**上进行评估。

https://zhuanlan.zhihu.com/p/83841282?from_voters_page=true



B超项目中，数据集总共有600+张图片，将每张图片左右分开后，取有劳损区域的那一边作为数据集。这样划分后大概有1100+张图片作为目标检测的数据集。

然后采用5折交叉验证，过程如上，训练一个目标检测网络，用来提取劳损区域。

---

在训练模型的过程中，yolov3训练代码是采用的mxnet官网上的代码，根据需要，将数据集生成了一个.lst的文件格式，然后通过官网代码，根据.lst文件生成对应的.rec和.idx文件。之后就可以用来训练模型了。

---



**YOLOv3**

这一块我在“深度学习面经”中写了一遍，这里我再回顾一遍。



**偏差和方差**

https://www.zhihu.com/question/20448464



**Focal Loss**
$$
FL(p_t) = - \alpha_t (1 - p_t)^{\gamma} log(p_t)
$$
其中$\alpha_t$是类别权重，如果正样本比负样本=5:95，那么这个$\alpha$设为0.95。$\alpha=0.5$时，相当于关闭该功能。

$(1-p_t)^{\gamma}$是难度权重，$p_t$表示分为这个类别的概率，则概率越大，它所做出的的loss的贡献应该越小，所以应该打压这个loss值。$\gamma=0$时，关闭该功能，相当于不考虑样本分类的难易程度，一视同仁。

通俗易懂 https://blog.csdn.net/weixin_43913077/article/details/120360332

论文角度 https://zhuanlan.zhihu.com/p/49981234



**YOLOv3采用focal loss 效果拙见**

论文里，yolov3里一个gt只分配给一个最高iou的predict anchor，不是最高，如果iou高于阈值，就忽略这个预测，对于小于阈值的predict anchor只有objectness loss，这个方法筛除了很多预测框，而objectness loss只是很小的一部分。
本身yolov3只对ahchor和groud truth最高iou做loss，其本身筛选掉了大部分背景框，所以不存在不均衡的问题了。
yolov3在计算loss的时候，统计的是那些已经和ground truth匹配上的loss，其实这就是解决正负样本不均衡的一个办法，所以focal loss在yolo上作用没那么大。
原文链接：https://blog.csdn.net/qq_34795071/article/details/89536836



#### 过滤网络 ####

---

**准备数据集**

图像：该分类网络的数据集是在前一个网络的基础上进行预测，将目标检测预测到的框中的内容作为第一通道，该框在对称区域上进行滑动，找到**相似区域**作为第二通道，将两者相减作为第三通道。三个通道合成一张图片作为过滤网络的数据集。



标签：目标检测的预测的框与目标检测标签进行iou计算，大于某一阈值则为正样本，标签设置为1；否则为负样本，标签设置为-1。



**特征提取**

对目标检测的预测框进行一定范围的扩张（上下左右），并将预测框部分涂黑，只留出周围扩张区域，这样得到掩膜作为模板区域。而参考区域上的滑动窗口也同样是这样一个掩膜。

![](https://img-blog.csdnimg.cn/20190419232017555.png)

bitwise_and https://blog.csdn.net/u014303647/article/details/89409813



通过模板区域在参考图像上找相似区域的时候，原理是利用一个滑动窗口在指定范围内进行滑动。利用HOG提取模板区域的特征和窗口中的特征，然后计算二值之间的距离。



下面几个算法可以直接调库 skimage。这里先要讲一个Sobel滤波器

```python
'''
纵向
[1   2   1]
[0   0   0]
[-1 -2  -1]

横向
[1   0   -1]
[2   0   -2]
[1   0   -1]
'''

import cv2 as cv
import numpy as np


def BGR2GRAY(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    # 转换成单通道灰度图
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
    gray = gray.astype(np.int8)

    # img[:, :, 0] = gray
    # img[:, :, 1] = gray
    # img[:, :, 2] = gray

    return gray


def sobel_filter(gray, ksize=3):
    H, W = gray.shape[:2]
    pad = ksize // 2
    out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float32)
    out[pad:pad + H, pad:pad + W] = gray.copy()
    tmp = out.copy()

    out_v = out.copy()
    out_h = out.copy()

    kv = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    kh = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]

    for y in range(H):
        for x in range(W):
            out_v[pad + y, pad + x] = np.sum(kv * tmp[y:y + ksize, x:x + ksize])
            out_h[pad + y, pad + x] = np.sum(kh * tmp[y:y + ksize, x:x + ksize])

    out_v = np.clip(out_v, 0, 255)
    out_h = np.clip(out_h, 0, 255)

    out_v = out_v[pad:pad + H, pad:pad + W].astype(np.uint8)
    out_h = out_h[pad:pad + H, pad:pad + W].astype(np.uint8)

    return out_v, out_h


img = cv.imread('../imori.jpg')
gray = BGR2GRAY(img)  # cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('gray', gray)
# cv.waitKey(0)
# cv.destroyAllWindows()

out_v, out_h = sobel_filter(gray)
cv.imshow('out_v', out_v)
cv.imshow('out_h', out_h)
cv.waitKey(0)
cv.destroyAllWindows()
```



**HOG**

（1）先用Sobel Filter计算出$g_x, g_y$，然后可以算出当前像素点的梯度$g = \sqrt{g_x^2 + g_y^2}$，和方向$tan \theta = \frac{g_y}{g_x} -> \theta = arctan \frac{g_y}{g_x}$；

（2）将8x8个像素点设置为一个cell，由于每个像素点有2个信息：梯度和方向，所以一个cell共有8x8x2=128个信息，一个cell就包含这么多信息，要是用于后面的计算，计算量爆炸。并且对像素值敏感，抗噪声能力不强。所以我们设置cell这个东西出来。

（3）我们将$\theta$从0到180（注意这里不是360，将正好相反的两个方向设置为同一个），分为9个区间，在每个像素点的$\theta$落入的区间内，填入相应的梯度值，这样我们就将8x8x2=128个信息转换为了9个信息。

（4）将连续的2x2个cell看作是一个block，按照stride=一个cell单位，在feature map上滑动。每次滑动把这4个cell展成一行，36维向量，计算L2范数，并归一化。这个值作为当前block的特征。

参考 https://zhuanlan.zhihu.com/p/85829145



**pHash**

感知哈希算法，通过离散余弦变换（DCT，DFT才是离散傅里叶变换）降低图片频率，相比aHash（一个区域内的像素点作均值）有较好的鲁棒性。

（1）将图片缩放为32x32；

（2）灰度化处理；

（3）计算DCT，并选取左上角8x8的矩阵。

（DCT是一种特殊的**傅立叶变换**，将图片从像素域变换为频率域，并且DCT矩阵从左上角到右下角代表越来越高频率的系数，但是除左上角外，其他地方的系数为0或接近0，因此只保留左上角的低频区域。）

（4）计算DCT均值。

（5）将每个DCT值与均值DCT比较，大于或等于平均值，记为1，小于平均值记为0，由此生成二进制数组。

（6）图片配对，计算汉明距离。



**lbp**

(Local Binary Pattern，局部二值模式)，用于局部纹理特征提取。

基本的LBP特征算子：

原始的LBP算子定义为在3x3的窗口内，以窗口中心像素为阈值，将相邻的8个像素的灰度值与其进行比较，若周围像素值大于等于中心像素值，则该像素点的位置被标记为1，否则为0。这样3x3邻域内的8个点经比较可产生8位二进制数（通常转换为十进制数即LBP码，共256种），即得到该窗口中心像素点的LBP值，并用这个值来反映该区域的纹理信息。需要注意的是，LBP值是按照顺时针方向组成的二进制数。

原文链接 https://blog.csdn.net/lk3030/article/details/84034963



**Canny**

同样提取模板区域和相似区域的特征，这里我们提取两者的边缘特征，然后计算二者之间的距离。

cv2.Canny(src, threshold1, threshold2)



**计算距离**

MSE（均方差）



cosine（余弦相似度）

两个向量$A=(x_1, x_2), B=(y_1, y_2)$
$$
cos(\theta) = \frac{A * B}{|A| * |B|} \\
cos(\theta) = \frac{x_1 y_1 + x_2 y_2}{\sqrt{x_1^2 + x_2^2} * \sqrt{y_1^2 + y_2^2}}
$$


欧氏距离能够体现个体数值特征的绝对差异，所以更多的用于需要从维度的数值大小中体现差异的分析，如使用用户行为指标分析用户价值的相似度或差异。

余弦距离更多的是从方向上区分差异，而对绝对的数值不敏感，更多的用于使用用户对内容评分来区分兴趣的相似度和差异，同时修正了用户间可能存在的度量标准不统一的问题（因为余弦距离对绝对数值不敏感）。



欧式距离

L2范数



jaccard

交并比



phash（感知哈希算法计算距离）

```python
# 实际上就是计算汉明距离
n = 0
for i in range(len(hash1)):
    if hash1[i] != hash2[i]:
        n += 1
return n
```



**搭建神经网络**

改进的孪生神经网络：https://blog.csdn.net/weixin_43723393/article/details/117470344

这里先思考孪生神经网络的思路。

首先，参考区域一定是无劳损区域

（1）如果模板区域是正样本。它与参考区域相似 -> 可以说明它不是劳损区域；它与参考区域不相似 -> 什么都不能说明。

（2）如果模板区域是负样本。它与参考区域相似 -> 可以说明它不是劳损区域；它与参考区域不相似 -> 什么都不能说明。

只要与参考区域相似，那么就可以排除模板区域不是劳损区域。但是我们用的是HingeLoss，标签与预测对的上损失才为0，标签与损失对不上则有损失。这里参考区域一定是负样本，这样训练的网络只能判断负样本。



于是孪生神经网络在此处并没有什么意义。不如干脆用分类网络来决定，输入一个三通道图片

（1）它符合某些特征，通过神经网络输出，与标签的损失小，那就判断为正样本；

（2）否则是负样本。

这里为什么一定要输入一个3通道图像呢？融合特征，将非劳损区域的特征融合到一起，能学习到更多有用的特征信息，有助于提高网络的识别率。



backbone是ResNet18_v2

ResNet-v2重新设计了一种残差网络基本单元（unit）就是将激活函数（先BN再ReLU）移到权值层之前，形成一种“预激活（pre-activation）”的方式，如(b)，而不是常规的“后激活（post-activation）”方式，并且预激活的单元中的所有权值层的输入都是归一化的信号，如(a)。这使得网络更易于训练并且泛化性能也得到提升。

<img src="https://pic2.zhimg.com/80/v2-5b1954163952049e13a75403c5f4b161_720w.jpg" style="zoom:80%;" />

稍微修改网络：

（1）去掉最后的全连接层

（2）添加一个输出通道为256的全连接层；

（3）添加一个输出通道为64的全连接层；

（4）添加一个输出通道为1的全连接层；

（5）最后再接一个tanh激活函数；



输入图像是一个3通道图片，大小为(128,128,3)。这里为什么是128x128？





**数据增强**

```
trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
)
```

我自己的回答：

（1）ToTensor()先将(H, W, C)转换通道为(C, H, W)，然后转成float类型，最后每个像素再除以255，进行归一化到[0.0, 1.0]之间；

（2）ToTensor()已经[0.0, 1.0]了，怎么还要-mean再/std呢？Normalize是对数据按通道再进行标准化，[0.0, 1.0]只改变了范围，没有改变分布；而normalize是为了让数据服从正态分布。

（3）mean和std的这个值怎么来的？ImageNet数据集上跑下来的，规定的这个值。



别人的解答：数据如果分布在(0,1)之间，可能实际的bias，就是神经网络输入的b会比较大，而模型初始化时b=0，这样会导致神经网络收敛比较慢，经过Normalize后，可以加快模型的收敛速度。
因为对RGB图片而言，数据范围是[0-255]的，需要先经过ToTensor除以255归一化到[0,1]之后，再通过Normalize计算过后，将数据归一化到[-1,1]。

参考 https://blog.csdn.net/qq_38765642/article/details/109779370



**损失函数**

HingeLoss
$$
HingeLoss(x) = max(0, 1 - y f(x))
$$
经过该分类网络输出的值只有1或者-1。

当为标签为1时：

（1）若输出为1，分类正确，损失为0；

（2）若输出为-1，分类错误，损失为2；

而当标签为-1时：

（1）若输出为1，分类错误，损失为2；

（2）若输出为-1，分类正确，损失为0。



**学习率策略**

（1）固定步长衰减；

有时我们希望学习率每隔一定步数（或者epoch）就减少为原来的gamma分之一，使用固定步长衰减依旧先定义优化器，再给优化器绑定StepLR对象：

```python
optimizer_StepLR = torch.optim.SGD(net.parameters(), lr=0.1)
StepLR = torch.optim.lr_scheduler.StepLR(optimizer_StepLR, step_size=step_size, gamma=0.65)
```

<img src="https://pic1.zhimg.com/80/v2-a1c38e6c8e26ad3e953d1ebb67d7243c_720w.jpg" alt="img" style="zoom: 67%;" />

（2）多步长衰减；

上述固定步长的衰减的虽然能够按照固定的区间长度进行学习率更新，但是有时我们希望**不同的区间采用不同的更新频率**（或者说有的区间更新学习率，有的区间不更新学习率），这就需要使用MultiStepLR来实现动态区间长度控制：

```python
optimizer_MultiStepLR = torch.optim.SGD(net.parameters(), lr=0.1)
torch.optim.lr_scheduler.MultiStepLR(optimizer_MultiStepLR, milestones=[200, 300, 320, 340, 200], gamma=0.8)
```

<img src="https://pic1.zhimg.com/80/v2-4752ff055e6c6daafc1bc74c9b367090_720w.jpg" alt="img" style="zoom:67%;" />

从图中可以看出，学习率在区间[200, 400]内快速的下降，这就是milestones参数所控制的，在milestones以外的区间学习率始终保持不变。



（3）指数衰减；

学习率按照指数的形式衰减是比较常用的策略，我们首先需要确定需要针对哪个优化器执行学习率动态调整策略，也就是首先定义一个优化器：

```python
optimizer_ExpLR = torch.optim.SGD(net.parameters(), lr=0.1)
```

定义好优化器以后，就可以给这个优化器绑定一个指数衰减学习率控制器：

```python
ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer_ExpLR, gamma=0.98)
```

其中参数gamma表示衰减的底数，选择不同的gamma值可以获得幅度不同的衰减曲线，如下：

<img src="https://pic3.zhimg.com/80/v2-d990582cda2fc2aa88ae91d5aa17a6b6_720w.jpg" alt="img" style="zoom:67%;" />

（4）余弦退火衰减。

严格的说，余弦退火策略不应该算是学习率衰减策略，因为它使得学习率按照周期变化，其定义方式如下：

```python
optimizer_CosineLR = torch.optim.SGD(net.parameters(), lr=0.1)
CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_CosineLR, T_max=150, eta_min=0)
```

<img src="https://pic2.zhimg.com/80/v2-bb255df05eb665cc6530845bde637bc9_720w.jpg" alt="img" style="zoom:67%;" />



参考 https://zhuanlan.zhihu.com/p/93624972



**模型训练和评估**

在下面一节并讲。



#### 评估指标 ####

---

一级指标：混淆矩阵

TP：被正确划分的正样本。

FP：错误预测为正样本。实际为负样本。

TN：正确预测为负样本。

FN：错误预测为负样本。实际为正样本。

![](https://img-blog.csdn.net/20180531150151899?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L09yYW5nZV9TcG90dHlfQ2F0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



二级指标：precision，recall

**accuracy**

准确度 = 划分正确 / 总样本
$$
accuracy = (TP + TN) / (TP + FP + TN + FN)
$$


**precision**

精确率 = 正确预测为正样本 / 预测为正样本中
$$
precision = TP / (TP + FP)
$$


**recall**

召回率 = 正确预测为正样本 / 实际为正样本中
$$
recall = TP / (TP + FN)
$$
为什么你的recall高而precision低，怎么改进？

看公式，应该是FN（错误预测为负样本，实际为正样本）小，即没有检测出来；而FP（错误预测为正样本，实际为负样本，即预测错误）大。画个图如下：

<img src="https://pic2.zhimg.com/80/v2-aca24666e0a3988178d45c1bcf8691b9_720w.jpg" alt="img" style="zoom: 67%;" />

通过第二行两个图看出，是由于正样本太少，而负样本太多的缘故，正负样本比例不平衡。



三级指标：F-beta score，ROC，AUC

**F-beta score**

F1-score

F1分数（F1-score）是分类问题的一个衡量指标。一些多分类问题的机器学习竞赛，常常将F1-score作为最终测评的方法。它是精确率和召回率的调和平均数，最大为1，最小为0。
$$
F_1 = \frac{2}{\frac{1}{precision} + \frac{1}{recall}} \\
F_1 = \frac{2 * precision * recall}{precision + recall}
$$



此外还有F2分数和F0.5分数。F1分数认为召回率和精确率同等重要，F2分数认为召回率的重要程度是精确率的2倍，而F0.5分数认为召回率的重要程度是精确率的一半。
$$
F_{\beta} = (1 + \beta^2) * \frac{precision * recall}{(\beta^2 * precision) + recall}
$$

**PRC（PR曲线）**

将所有结果列成一个表格，然后每次去一行数据计算precision和recall。再在坐标轴上描出每个点的坐标。y轴是precision，x轴是recall，画出的曲线图。

<img src="https://img-blog.csdnimg.cn/2019032414544420.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zMTg2NjE3Nw==,size_16,color_FFFFFF,t_70" alt="img" style="zoom:67%;" />

**ROC（曲线） -> AUC（曲线下面积）**

先计算TPR（True Positive Rate），也就是recall
$$
TPR = \frac{TP}{TP + FN}
$$
再计算FPR（False Positive Rate），所有负样本中预测错误的负样本。计算公式如下
$$
FPT = \frac{FP}{FP + TN}
$$
画出**ROC**后，再取曲线下方面积，得到的就是AUC（Area Under the Curve）。



**AUC (Area under Curve)：ROC曲线下的面积，介于0.1和1之间，作为数值可以直观的评价分类器的好坏，值越大越好。**
 AUC = 1，是完美分类器，采用这个预测模型时，存在至少一个阈值能得出完美预测。绝大多数预测的场合，不存在完美分类器。
 0.5 < AUC < 1，优于随机猜测。这个分类器（模型）妥善设定阈值的话，能有预测价值。
 AUC = 0.5，跟随机猜测一样（例：丢铜板），模型没有预测价值。



**mAP**

mean Average Precision，目标检测模型的评估指标。

首先得直到
$$
precision = \frac{TP}{TP + FP} = \frac{TP}{all \ detections} \\
recall = \frac{TP}{TP + TN} = \frac{TP}{all \ ground \ truth}
$$
假设我们有 7 张图片（Images1-Image7），这些图片有 15 个目标（绿色的框，GT 的数量，上文提及的 `all ground truths`）以及 24 个预测边框（红色的框，A-Y 编号表示，并且有一个置信度值）

<img src="https://pic1.zhimg.com/80/v2-793336302ec813c5498cdea348255191_720w.jpg?source=1940ef5c" style="zoom:67%;" />

（1）先求出每个框，包括它的置信度，预测为正样本或者负样本（TP or FP）；

<img src="https://pica.zhimg.com/80/v2-af3db57f93bd7e5c0786273bdaa78251_720w.jpg?source=1940ef5c" style="zoom: 67%;" />

（2）按照置信度从高到低对每个框进行排序。

<img src="https://pic3.zhimg.com/80/v2-855b1e83d69700445924bcb81f0e0c91_720w.jpg?source=1940ef5c" style="zoom: 67%;" />

（3）每次遍历表中的一个点，计算当前每个点的precision和recall。

比如Detections为R时，累计TP（ACC TP）为1，累计FP（ACC FP）为0，累计TP + TN（也就是all ground truth）为15。计算出当前$precision=\frac{TP}{TP + FP}=\frac{1}{1+0}=1, recall=\frac{TP}{TP + TN}=\frac{1}{15}=0.0666$。

继续，Detections为Y时，累计TP为1，累计FP为1，累计TP + TN为15，那么$precision=\frac{1}{1+1}=0.5, recall=\frac{1}{15}=0.0666$。

第三个，Detections为J时，累计TP为2，累计FP为1，累计TP + TN为15。$precision=\frac{2}{2+1}=0.6666, recall=\frac{2}{15}=0.1333$。

...

（4）然后根据每个点的坐标，绘制出PR曲线（AP就是PR曲线下面积），如下

<img src="https://pic3.zhimg.com/80/v2-fd0bd7bcfefd47a1450593cfcde4b2d8_720w.jpg?source=1940ef5c" style="zoom:67%;" />

（5）要计算P-R下方的面积，一般使用的是插值的方法。取 11 个点 **[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]** 的插值所得。

<img src="https://pic3.zhimg.com/80/v2-16275079aee866864a238113607cd051_720w.jpg?source=1940ef5c" style="zoom:67%;" />

计算一个AP的方法如下
$$
AP = \frac{1}{11} \sum_{r \in \{0,0.1,0.2,...,1\}} p_{interp}(r) \\
AP = \frac{1}{11} (1 + 0.6666 + 0.4285 + 0.4285 + 0.4285 + 0 + 0 + 0 + 0 + 0) \\
AP = 0.2684
$$
要计算mAP，就把所有类别的 AP 计算出来，然后求取平均即可。



具体做法 https://www.zhihu.com/question/53405779/answer/993913699

代码 https://zhuanlan.zhihu.com/p/70667071



两种mAP计算方法

<img src="https://img-blog.csdn.net/20170105154429999?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaHlzdGVyaWMzMTQ=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast" style="zoom:50%;" />

第一种方法
$$
\sum^{N}_{k=1} p(k)\Delta(r)
$$
第二种方法，插值AP（Interpolated average precision）
$$
max_{\hat{k} >= k} p(\hat{k}) \\
\sum^{N}_{k=1} max_{\hat{k} >= k} p(\hat{k}) \Delta r(k)
$$


2010年之后的做法在计算AP的时候有所不同。10年之前是取11个点，10年之后是逐点计算AP。



#### 其他 ####

---

**线程和进程的关系和区别**

https://www.php.cn/faq/478711.html



**python中的比较：is和==**

（1）is 比较的是**两个实例对象是不是完全相同，它们是不是同一个对象，占用的内存地址是否相同**。莱布尼茨说过：“世界上没有两片完全相同的叶子”，这个is正是这样的比较，比较是不是同一片叶子（即比较的id是否相同，这id类似于人的身份证标识）。

（2）== 比较的是**两个对象的内容是否相等，即内存地址可以不一样，内容一样就可以了**。这里比较的并非是同一片叶子，可能叶子的种类或者脉络相同就可以了。默认会调用对象的 \_\_eq\_\_()方法。

1、is 比较两个对象的 id 值是否相等，是否指向同一个内存地址；
2、== 比较的是两个对象的内容是否相等，值是否相等；
3、小整数对象[-5,256]在全局解释器范围内被放入缓存供重复使用；
4、is 运算符比 == 效率高，在变量和None进行比较时，应该使用 is。



**讲讲用户态和内核态**

unix/linux体系架构

<img src="https://images2015.cnblogs.com/blog/431521/201605/431521-20160523163606881-813374140.png" style="zoom: 33%;" />

​		由于需要限制不同程序之间的访问能力，防止他们获取别的程序的内存数据，或者获取外围设备的数据，并发送到网络，cpu划分出两个权限等级--用户态和内核态。

内核态：cpu可以访问内存的所有数据，包括外围设备，例如硬盘，网卡。cpu也可以将自己从一个程序切换到另一个程序。

用户态：只能受限的访问内存，且不允许访问外围设备，占用cpu的能力被剥夺，cpu资源可以被其他程序获取。



用户态和内核态之间的切换

​		所有的用户程序都是运行在用户态的，但是有时候程序确实需要做一些内核态的事情，例如从硬盘读取数据，或者从键盘获取数据等。而唯一可以做这些事情的就是操作系统，所以此程序就需要先申请操作系统请求以程序的名义来执行这些操作。

​		这时需要一个这样的机制：用户态程序切换到内核态，但是不能控制在内核态中执行的指令。这种机制叫做系统调用，在cpu中的实现称之为陷阱指令（Trap Instruction）。

工作流程如下：

（1）

（2）

（3）

（4）

（5）



有三种情况会将用户态到内核态的转换：

（1）系统调用。这是用户态进程主动要求切换到内核态的一种方式，用户态进程通过系统调用申请使用操作系统提供的服务程序完成工作。而系统调用的核心机制还是使用了操作系统为用户特别开放的一个中断来实现，如linux的int 80h中断。

（2）异常事件。当CPU正在执行运行在用户态的程序时，突然发生某些预先不可知的异常事件，这个时候就会触发从当前用户态执行的进程转向内核态执行的异常事件。典型的如缺页异常。

（3）外围设备的中断。当外围设备完成用户的请求操作后，会向CPU发出中断信号，此时CPU就会暂停执行下一个即将要执行的命令，转去执行中断信号对应的处理程序。如果先前执行的指令是在用户态下，那么自然就发生从用户态到内核态的转换。



详细参考 https://www.cnblogs.com/maxigang/p/9041080.html



**怎么判断用多进程还是多线程**

​		进程其实就是我们经常说的执行程序，多个进程指的是有多个程序同时占用cpu资源。比如你开启浏览器，并打开音乐，登录qq，这些就是多个进程。

<img src="C:\Users\87995\AppData\Roaming\Typora\typora-user-images\image-20220303155846849.png" alt="image-20220303155846849" style="zoom:67%;" />

经典一句话：进程是资源分配的最小单位，线程是cpu调度的最小单位。

多线程的优点：（1）数据方便共享（不需要进程间的通信）；（2）占用系统内存小；（3）提高cpu利用率。

多线程的缺点：（1）调试困难；（2）防止读写竞争，锁机制；（2）编程复杂。

多进程的优点：（1）一个进程挂掉不会影响其他的进程；（2）编程简单。

多进程的缺点：（1）耗资源。

所以：

（1）经常需要创建销毁，优先采用线程；

（2）需要进行大量计算，所谓大量计算即消耗cpu，频繁切换，这种情况选择使用多线程。最常见的有图像处理，深度计算；

（3）强相关的处理用线程，弱相关的处理用进程；

（4）多核分布用线程，多机分布用进程；

如果多进程和多线程都能够满足要求，那么选择最熟悉、最拿手的那个。需要提醒的是：虽然有这么多的选择原则，但实际应用中基本上都是“进程 + 线程”的结合方式。

详细 https://www.cnblogs.com/x_wukong/p/10106214.html





**tcp三次握手和四次挥手**





**讲讲tcp可靠性体现在哪些方面**





**讲讲python协程**

在异步IO中经常出现这个问题。

​		协程，又称微线程，纤程，英文名Coroutine。协程的作用是在执行函数A时可以随时中断去执行函数B，然后中断函数B继续执行函数A（可以自由切换）。但这一过程并不是函数调用，这一整个过程看似像多线程，然而协程只有一个线程执行。

协程的优势：

（1）执行效率极高。因为子程序切换（函数）不是线程切换，由程序自身控制，没有切换线程的开销。所以与多线程相比，线程的数量越多，协程性能的优势越明显。

（2）不需要多线程的锁机制。因为只有一个线程，也不存在同时写变量冲突，在控制共享资源时也不需要加锁。



**python运行中间的pyc文件**

​		编程语言大概能分为两种类型：**解释性语言**和**编译性语言**。解释性语言大概就是指程序执行的时候，执行一句，翻译一句给机器；编译性语言大概在程序运行之前，预编译翻译为机器语言。

​		java和python属于中间那种，先将程序预编译为一种人类语言和机器语言之间的语言，然后在运行时再继续编译。比如java中，有这么一条语句javac hello.java（生成.class文件）这样的预编译过程。

​		而python就没有这个过程，python运行时需要加载其依赖的模块，所以python在第一次运行的时候会生成一个.pyc文件，把模块的编译版缓存在\_\_pycache\_\_目录中（为了加速模块的加载速度），在第二次运行时就会直接运行pyc文件。.pyc产生后一般不会变动，除非py源文件也变化了，python会对比编译版本与源码的修改日期，查看它是否已过期，是否要重新编译。



**计算机体系结构来讲，程序运行的快指的是什么**





**了解pytorch框架本身吗？看过源码吗，了解底层实现吗？**



参考 [https://blog.csdn.net/weixin_33725722/article/details/88107432?utm_term=pytorch%E5%BA%95%E5%B1%82%E6%BA%90%E7%A0%81&utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~sobaiduweb~default-1-88107432&spm=3001.4430](https://blog.csdn.net/weixin_33725722/article/details/88107432?utm_term=pytorch底层源码&utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~sobaiduweb~default-1-88107432&spm=3001.4430)





**什么是np问题**



