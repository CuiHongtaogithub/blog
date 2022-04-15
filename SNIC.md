## superpixels（超像素）

第一次听说这个超像素很容易理解错误，以为是在普通的像素基础上继续像微观细分，如果这样理解就恰好理解反了，其实超像素是一系列像素的集合，这些像素具有类似的颜色、纹理等特征，距离也比较近。用超像素对一张图片进行分割的结果见下图，其中每个白色线条区域内的像素集合就是一个超像素。需要注意的是，超像素很可能把同一个物体的不同部分分成多个超像素。2^2^=4



![img](https://pic3.zhimg.com/v2-14c6d2ea46a0297cfb846c4ccc0eda4e_b.webp)



超像素最早的定义来自2003年 Xiaofeng Ren等人的一篇论文《Learning a Classification Model for Segmentation》。

其中超像素中比较常用的一种方法是SLIC（simple linear iterative clustering），是Achanta 等人2010年提出的一种思想简单、实现方便的算法，将彩色图像转化为CIELAB颜色空间和XY坐标下的5维特征向量，然后对5维特征向量构造距离度量标准，对图像像素进行局部聚类的过程。SLIC算法能生成紧凑、近似均匀的超像素，在运算速度，物体轮廓保持、超像素形状方面具有较高的综合评价，比较符合人们期望的分割效果。



![img](https://pic4.zhimg.com/80/v2-199000a0006707c07cd0d8487f13c92f_720w.jpg)

**通常期望超像素具有以下特性：**

1. **紧密区域边界粘附和  紧凑；限制邻接程度。**（*被用紧性因子m表示，是用户提供的。其中m表示空间和像素颜色的相对重要性的度量。当m大时，空间邻近性更重要，并且所得到的超像素更紧凑（即它们具有更低的面积与周长比）。当m小时，所得到的超像素更紧密地粘附到图像边界，但是具有较小的规则尺寸和形状。当使用CIELAB色彩空间时，m可以在[1,40]的范围内。*）
3. **包含一个类似像素的小集群。**
4.  **均匀；大小大致相同的簇。**
5.  **高的计算效率。**



## **Semantic Segmentation（语义分割）**

语义分割还是比较常见的，就是把图像中每个像素赋予一个类别标签（比如汽车、建筑、地面、天空等），比如下图就把图像分为了草地（浅绿）、人（红色）、树木（深绿）、天空（蓝色）等标签，用不同的颜色来表示。

不过这种分割方式存在一些问题，比如如果一个像素被标记为红色，那就代表这个像素所在的位置是一个人，但是如果有两个都是红色的像素，这种方式无法判断它们是属于同一个人还是不同的人。也就是说语义分割只能判断类别，无法区分个体。



![img](https://pic2.zhimg.com/80/v2-a0e8be79238485e6867f23caeeb97825_720w.jpg)



但很多时候我们更需要个体信息，想要区分出个体怎么办呢？继续往下看吧

## **Instance Segmentation（实例分割）**

实例分割方式有点类似于物体检测，不过[物体检测^1^](https://zhuanlan.zhihu.com/p/50996404?utm_source=qq&utm_medium=social&utm_oi=1342065170911408128)一般输出的是 bounding box，实例分割输出的是一个mask。

实例分割和上面的语义分割也不同，它不需要对每个像素进行标记，它只需要找到感兴趣物体的边缘轮廓就行，比如下图中的人就是感兴趣的物体。该图的分割方法采用了一种称为[Mask R-CNN](https://zhuanlan.zhihu.com/p/65321082?utm_source=qq&utm_medium=social&utm_oi=1342065170911408128)**（注：CNN是卷积神经网络的简称，当图片被输入类似神经系统时候，神经网络感受信息不是图片，而是以像元为单位的数据。而卷积是 他在处理图片时不是以一个像素为单位，而是以较大的一个小的图像块为单位处理，从而神经网络能够得到这一小块总体特征，如上图得到图片的轮廓。经过多次的对上一次卷积，逐渐总结出鼻子嘴巴等信息。从而得到对图像的总体特征，是一张人脸）**的方法。我们可以看到每个人都是不同的颜色的轮廓，因此我们可以区分出单个个体。



![img](https://pic4.zhimg.com/80/v2-dbb56a65bcb6c7eedfd833445fdf9ecf_720w.jpg)



## **Panoptic Segmentation（全景分割）** 

最后说说全景分割，它是语义分割和实例分割的结合。如下图所示，每个像素都被分为一类，如果一种类别里有多个实例，会用不同的颜色进行区分，我们可以知道哪个像素属于哪个类中的哪个实例。比如下图中黄色和红色都属于人这一个类别里，但是分别属于不同的实例（人），因此我们可以通过mask的颜色很容易分辨出不同的实例。



![img](https://pic3.zhimg.com/80/v2-a3bde4183e7f0c20fdb1e01e4ab572a6_720w.jpg)

## SLIC算法

是simple linear iterative cluster的简称，该算法用来生成超像素（superpixel）。

 1.SLIC在[CIE-Lab](https://github.com/RS-SWT/Machine-learing-LLY/blob/main/%E8%A1%A5%E5%85%85/%E4%BB%80%E4%B9%88%E6%98%AFCIELAB.md)五维颜色和图像空间中进行局部k-means优化（将图像从RGB颜色空间转换到[CIE-Lab](https://github.com/RS-SWT/Machine-learing-LLY/blob/main/%E8%A1%A5%E5%85%85/CIELAB.md)颜色空间，对应每个像素的（L，a，b）颜色值和（x，y）坐标组成一个5维向量V[L,a,b,x,y]。两个像素的相似性可由它们的向量距离来度量，距离越大，相似性越小。），

2.从在规则网格上选择的种子开始对像素进行聚类（类似于[K-means聚类算法](https://github.com/RS-SWT/Machine-learing-LLY/blob/main/%E8%A1%A5%E5%85%85/%23%20K-%E5%9D%87%E5%80%BC%E8%81%9A%E7%B1%BB(k-means).md)，算法首先生成K个种子点，然后在每个种子点的周围空间里搜索距离该种子点最近的若干像素，将他们归为与该种子点一类，直到所有像素点都归类完毕。然后计算这K个超像素里所有像素点的平均向量值，重新得到K个聚类中心，然后再以这K个中心去搜索其周围与其最为相似的若干像素，所有像素都归类完后重新得到K个超像素，更新聚类中心，再次迭代，如此反复直到收敛）。

在实际应用中，SLIC只有两个输入参数-所需的超像素数量s和紧度因子m，这决定了超像素的紧凑程度。 （ SLIC的作者还介绍了一个不需要紧凑性参数作为输入的版本，因为它自动将其值设置为超像素内的最大颜色距离）

> SLIC使用简单易懂。默认情况下，算法的唯一参数是k，其含义是大小大致相等的超像素的个数。对于CIELAB色彩空间中的彩色图像，聚类过程从初始化步骤开始，其中k个初始聚类中心![这里写图片描述](https://img-blog.csdn.net/20161030201720621)在间隔S个像素的规则网格上采样。为了产生大致相等大小的超像素，网格间隔为![这里写图片描述](https://img-blog.csdn.net/20161030201801916)。将中心移动到与3×3邻域中的最低梯度位置相对应的种子位置。这样做是为了避免将超像素定位在边缘上，并且减少用噪声像素接种超像素的机会。 
> 接下来，在分配步骤中，每个像素i与搜索区域与其位置重叠的最近聚类中心相关联，如图2所示。这是加速我们的算法的关键，因为限制搜索区域的大小显着地减少了距离计算的数量，并且导致相对于常规kmeans聚类的显着的速度优势，其中每个像素必须与所有聚类中心比较。这只能通过引入距离测量D来实现，该距离测量D确定每个像素的最近聚类中心，如第III-B节中所讨论的。由于超像素的预期空间范围是近似尺寸S×S的区域，因此在超像素中心周围的区域2S×2S中进行类似像素的搜索。 
> ![这里写图片描述](https://img-blog.csdn.net/20161030201633854) 
> 图.2：减少超像素搜索区域。SLIC的复杂性在图像O（N）中的像素数目中是线性的，而常规的k均值算法是O（kNI），其中I是迭代次数。这在分配步骤中提供了每个聚类中心的搜索空间。（a）在常规k均值算法中，从每个聚类中心到图像中的每个像素计算距离。（b）SLIC仅计算从每个聚类中心到2S×2S区域内的像素的距离。注意，期望的超像素大小仅为S×S，由较小的正方形表示。这种方法不仅减少了距离计算，而且使得SLIC的复杂性与超像素的数量无关。 
> 一旦每个像素已经与最近的聚类中心相关联，更新步骤将聚类中心调整为属于该聚类的所有像素的平均向量![这里写图片描述](https://img-blog.csdn.net/20161030202323238)。L2范数用于计算新聚类中心位置和先前聚类中心位置之间的残差误差E.分配和更新步骤可以迭代重复，直到错误收敛，但我们发现10次迭代足够大多数图像，并报告本文中使用此标准的所有结果。最后，后处理步骤通过将不相交像素重新分配给附近的超像素来实施连通性。算法1中总结了整个算法。 
>
> 简单线性迭代聚类算法SLIC是最突出的超像素分割算法之一。 它作为一种预处理算法的成功归功于它的简单性、计算效率和生成超像素的能力，这些超像素满足良好的1.边界粘附和有限的邻接要求。

算法：

\1. 我们先在图像平面中的规则网格上选择的像素初始化我们的质心。对N个像素的图片 *image (Image):[^5]    The input image for clustering.，设置k个以某像素为中心的SxS的聚类，*

$$
s=\sqrt{\frac{N}{K}}
$$
移动中心至 3*3 范围内梯度值最小的像素点。*size (Integer, default: 5):[^6]*
*The superpixel seed location spacing, in pixels. If 'seeds' image is provided, no grid is produced.*

\2. 对N个像素计算在 2Sx2S 范围内的聚类中心点的距离，并将该像素点归入距离最近的聚类。

（注：距离公式

![](http://m.qpic.cn/psc?/V531KnY04NKhJG1WtBH80isWXv4SJLE6/45NBuzDIW489QBoVep5mcWQAFc9ZMWmMX566XdFEv.f6mULH3lvTCZINlq2kj1*PlhqqkIUPqtGcrDZpl1c3dHnTtb4Bm9UF0rToejbqtP8!/b&bo=SwFGAAAAAAAAFz0!&rf=viewer_4)

（关于这个公式

compactness (Float, default: 1):[^7]这里用m表示
Compactness factor. Larger values cause clusters to be more compact (square). Setting this to 0 disables spatial distance weighting.

![这里写图片描述](https://img-blog.csdn.net/20161031204816179)
![这里写图片描述](https://img-blog.csdn.net/20161031204826464)

![](http://m.qpic.cn/psc?/V531KnY04NKhJG1WtBH80isWXv4SJLE6/45NBuzDIW489QBoVep5mcdQsSfv11Ib8zEhiWqgA4C4*3sfpGVTwSi1WdiwFOV8FW2MmuWMc1Zz.6Ff5nBgV*B8JXjEc36RVMGV.na8mrd4!/b&bo=5AH4AQAAAAADFy4!&rf=viewer_4)

第二个比较新颖的地方是计算距离的时候作者与传统的采用Kmeans进行分割的算法不同，并不是对整个空间的所有像素进行计算，而是限定了区域，区域大小为2S，即寻找时以初始聚类中心为寻找中心，确定一个2S*2S的矩形，如下图所示:
![这里写图片描述](https://img-blog.csdn.net/20161031204852102)
图1：减少超像素搜索区域。SLIC的复杂性在图像O（N）中的像素数目中是线性的，而常规的k均值算法是O（kNI），其中I是迭代次数。这在分配步骤中提供了每个聚类中心的搜索空间。（a）在常规k均值算法中，从每个聚类中心到图像中的每个像素计算距离。（b）SLIC仅计算从每个聚类中心到2S×2S区域内的像素的距离。注意，期望的超像素大小仅为S×S，由较小的正方形表示。这种方法不仅减少了距离计算，而且使得SLIC的复杂性与超像素的数量无关。

好处是显而易见的，限制搜索区域的大小显着地减少了距离计算的数量，这样可以极大的加快速度，可以将算法控制为线性复杂度。
接着便是对kMeans算法进行迭代，直到算法收敛或迭代次数大于某一个值，根据论文大部分图像在迭代次数为10以内，具体迭代思路如下:
![这里写图片描述](https://img-blog.csdn.net/20161031205041542)
![这里写图片描述](https://img-blog.csdn.net/20161031205052153) 

在每次k-mean迭代中，SLIC通过计算与它最近的所有像素的平均值来演化一个中心

\3. 对k个聚类中心点进行更新，更新至聚类包含的像素点均值。

\4. 迭代 2，3 步骤，直到中心点不再移动，一般迭代10次。

最后可能出现一些小的区域d被标记为归属某一块超像素但却与这块超像素没有连接，这就需要把这块小区域d重新归类为与这块小区域d连接的最大的超像素中去，以保证每块超像素的完整。

SLIC存在一些缺点。 它需要多次迭代才能使中心收敛。 它 使用与输入数量相同大小的距离图 像素，这会占用大量内存 用于图像堆栈或视频量。最后，SLIC强制执行 连接性仅作为后处理步骤。

## SNIC

与SLIC不同，SNIC算法是非迭代的，从一开始就强制连接，需要较少的内存，而且速度更快。 基于我们的算法得到的超像素边界，我们还提出了一种多边形划分算法。 我们证明了我们的超像素以及多边形划分优于各自的最先进的定量基准算法。

图像分割仍然是一个吸引特定领域和通用解决方案的挑战。为了避免在使用[传统分割算法](https://github.com/RS-SWT/Machine-learing-LLY/blob/main/%E8%A1%A5%E5%85%85/%23%23%20%E4%BC%A0%E7%BB%9F%E6%96%B9%E6%B3%95.md)[^1]时与语义分割的斗争，研究人员最近将注意力转移到一个更简单和可实现的任务上，即将图像简化为称为超像素的连通像素的小集群。超像素分割已经迅速成为一种强大的预处理工具，它可以将图像从数百万像素简化为大约两个数量级的类似像素簇。

**算法**

1.像SLIC一样，我们也用在图像平面中的规则网格上选择的像素初始化我们的质心。 用颜色和空间坐标的五维空间中的距离d来测量像素与质心的亲和力。 我们的算法使用与SLIC相同的距离度量。 这种距离结合了归一化的空间和颜色距离。 在空间位置x=[x y]^T^和CIELAB颜色c=[Lab]^T^的情况下，给出了KTH超像素质心C[k]到JT H候选像素的距离：

![](http://m.qpic.cn/psc?/V531KnY04NKhJG1WtBH80isWXv4SJLE6/45NBuzDIW489QBoVep5mcWQAFc9ZMWmMX566XdFEv.f6mULH3lvTCZINlq2kj1*PlhqqkIUPqtGcrDZpl1c3dHnTtb4Bm9UF0rToejbqtP8!/b&bo=SwFGAAAAAAAAFz0!&rf=viewer_4)

其中s和m分别是空间距离和颜色距离的归一化因子。 对于N像素的图像，每个K超像素都期望包含N/K像素。 假设一个超像素的正方形形状，方程中s的值。 设置
$$
s=\sqrt{\frac{N}{K}}
$$
和 m的值.

2.在第一部分中，SNC算法相似于SLIC，我们也用在图像平面中的规则网格上选择的像素初始化我们的质心。，并使用与SLIC相同的距离度量. 在每次k-mean迭代中，SLIC通过计算与它最近的所有像素的平均值来演化一个质心，因此，它具有与质心相同的标签。 以这种方式，SLIC需要多次迭代才能使质心收敛。



![](http://m.qpic.cn/psc?/V531KnY04NKhJG1WtBH80isWXv4SJLE6/45NBuzDIW489QBoVep5mcdQsSfv11Ib8zEhiWqgA4C45XxxZFFRpXh2Q*wOXT4JkQ30ZYu5NV82vgPsOO4YOFNEEG6xvkLRIgrIBeMFHw0g!/b&bo=hQMcAQAAAAADF6k!&rf=viewer_4)

通俗举例点来讲如图 

  1.独特标签的初始种子。 此时Q为空。我们选择4个质心

2. 对于每个种子，计算到上下左右4个未标记的邻像元的距离d，并放入Q
3. 从初始中心开始，SNIC算法使用==优先队列==.(**从质心像元开始, 向上下左右4个方向,或者包括左上等8个方向([^3])的伸展,计算到质心的d，放入Q中，并从大到小排列**4和8叫做connectivity (Integer, default: 8):[^8]Connectivity. Either 4 or 8.,)选择下一个像素添加到集群中。弹出队列中d最小的元素，并给相应的像素贴上标签。 
4. 计算这个新标记像素到最近邻域d*(neighborhoodSize (Integer, default: null):[^9]*
   *Tile neighborhood size (to avoid tile boundary artifacts). Defaults to 2 * size.)，*并放入Q。
5. 重复3.4. 继续，直到Q为空。

从初始质心开始，我们的算法使用==优先级队列==[^2] 选择下一个像素添加到集群中。 优先级队列中填充的候选像素为4或8，连接到当前增长的超像素集群。 弹出队列提供了与**质心距离d最小的像素候选**.

![](http://m.qpic.cn/psc?/V531KnY04NKhJG1WtBH80isWXv4SJLE6/45NBuzDIW489QBoVep5mcUePK5AjGgJ8WQ.QsXNdw8YLsx7ywaXpx9DrY2gxUworkPTJdwS2JB4CXhElFGYA82lie.LboWsImtLbMXL2sXM!/b&bo=PAKdAgAAAAABF5E!&rf=viewer_4)



在图像上的规则网格上，得到了初始K种子C[k]={x~k~，c~k~}的SLIC。 使用这些种子像素，创建K个元素e~i~={x~i~，c~i~，k，d~i,k~}，其中每个标签k被设置为1到K的唯一超像素标签，这些d~i,k~=0。 用这些K元素==<u>初始化优先级</u>==[^4]队列Q。 当弹出时，Q总是返回元素e~i~，其距离值d~i,k~  。到第k质心最小。 当Q不为空时，弹出最上面的元素。 如果元素指向的标签映射L上的像素位置没有标记，则给出质心的标签。 质心值是超像素中所有像素的平均值，用这个像素更新。 此外，尚未标记的邻居中的每一个，创建一个新元素，将与连接质心的距离和质心的标签分配给它。这些新元素被推到队列上。 当算法执行时，优先队列被清空，以便在一端分配标签，并在另一端填充新的候选项。 当没有剩余未标记像素添加新元素时，停止。

![](http://m.qpic.cn/psc?/V531KnY04NKhJG1WtBH80isWXv4SJLE6/45NBuzDIW489QBoVep5mcWQAFc9ZMWmMX566XdFEv.f6mULH3lvTCZINlq2kj1*PlhqqkIUPqtGcrDZpl1c3dHnTtb4Bm9UF0rToejbqtP8!/b&bo=SwFGAAAAAAAAFz0!&rf=viewer_4)

SNIC算法在不使用kmeans迭代的情况下对像素进行聚类，同时从一开始就显式地强制连接。 而且，SNIC对SLIC做了两^个重要的修改：1。 质心是利用在线平均来进化的。 2. 标签分配是使用优先级队列来实现的，该队列将具有最短距离D的元素返回到重心。**他大大减少了运算次数**

以正方形为例

![](http://m.qpic.cn/psc?/V531KnY04NKhJG1WtBH80isWXv4SJLE6/45NBuzDIW489QBoVep5mcdQsSfv11Ib8zEhiWqgA4C45XxxZFFRpXh2Q*wOXT4JkQ30ZYu5NV82vgPsOO4YOFNEEG6xvkLRIgrIBeMFHw0g!/b&bo=hQMcAQAAAAADF6k!&rf=viewer_4)



| SLIC        | 第一次 | 第二次                | 第三次 | …    | 和         |
| ----------- | ------ | --------------------- | ------ | ---- | ---------- |
| 计算d的次数 | S-K    | s~1~+s~2~+…s~k~-k=s-k | s-k    |      | （s-k）^n^ |
| **SNIC**    |        |                       |        |      |            |
| 计算d的次数 | 4k     | 3                     | 2或3   |      | 略大于s-k  |

$$

$$

在线更新质心是相当有效的，因为自然图像中的冗余通常导致相邻像素非常相似。 因此，质心迅速收敛，如图所示

<img src="http://m.qpic.cn/psc?/V531KnY04NKhJG1WtBH80isWXv4SJLE6/45NBuzDIW489QBoVep5mcSEjvPVzsrYbGAu9FvK*H7a3jZRWHN35I9W7L1Mb3c2uAiwhdFOJSfF4kG0JdF7aEK6VpmHouAVESushVOs4kSA!/b&bo=IAMZAQAAAAAAJzo!&rf=viewer_4" style="zoom:150%;" />

`图2。 在线更新的有效性。 左图显示100个空间质心，从绿色方块显示的位置开始，在收敛到红色方块显示的位置时漂移。 所占据的中间位置显示为白色。 右图显示了x、y和l质心在100个超像素上w.r.t其先前值的平均变化，即残差的图。 正如所看到的，在向超像素添加前50个像素时，误差足够下降，即质心收敛。`

> SLIC通过限制以K质心为中心的面积2s⇥2s的正方形区域内的距离计算来实现计算效率。 选择正方形区域的大小是保守的，以确保图像平面上相邻质心的正方形之间有一些重叠。 因此，即使在k均值迭代期间，质心从它们在图像平面上的原始位置移位后，每个像素也可以由最近的质心到达。
>
> 由于像素连通性在这种基于k均值的聚类中没有显式强制执行，因此可能不属于最终超像素，但位于2s⇥2s区域的像素仍然被访问，并且为它们计算距离d。 虽然重叠平方限制大大减少了要计算的距离，但冗余计算是不可避免的。
>
> 我们只计算到连接到当前增长的集群的4或8的像素的距离，以便创建填充队列的元素。 因此，即使与SLIC的单次迭代相比，我们的算法计算的距离也较少。 增强连通性的一个自然结果也是，我们不需要像SLIC那样对距离计算施加任何空间限制。 队列包含的元素比N少得多，因此它使用的内存比SLIC少，SLIC需要N大小的内存来存储距离。
>
> 通过使用优先级队列和在线平均来更新质心，我们得到了SNIC，与SLIC相比具有以下优点：
>
> 1. 从一开始就明确地执行连接。
> 2.  不需要多次k均值迭代。 
> 3. 较少的像素访问和距离计算。
> 4.  内存需求降低。

## \4. SNIC-based polygonal partitioning(4.基于SNIC的多边形分区)

其次，我们提出了一种称为SNICPOLY的多边形分割算法，它以SNIC超像素分割为基础。 图像的多角度分割已经被证明[12]特别适合于处理包含几何或人造结构的图像的应用。 图中给出了SNIC分割和SNICPOLY多边形分割的一个例子。![](http://m.qpic.cn/psc?/V531KnY04NKhJG1WtBH80isWXv4SJLE6/45NBuzDIW489QBoVep5mcc5FjXK2U572YVvLGjXQ386skEhmYeELwWUs8bmVxv*xSIyffHbk..KVTwEdomxf0tQMn3RQ0LYaPA6AacZu2Rw!/b&bo=HQInAgAAAAAAR1s!&rf=viewer_4) 

一种方法(CONPOLY)，它将图像分割成均匀大小的凸多边形，而不是创建任意形状的超像素。 这种分区在[表面重建](https://github.com/RS-SWT/Machine-learing-LLY/blob/main/%E8%A1%A5%E5%85%85/%E8%A1%A8%E9%9D%A2%E9%87%8D%E5%BB%BA.md)和对象定位等应用中得到了应用。 CONPOLY的作者使用线段检测器检测初步线段。 它们构建一个符合这些线段的[*Voronoi镶嵌*](https://github.com/RS-SWT/Machine-learing-LLY/blob/main/%E8%A1%A5%E5%85%85/VORONOI.md)，然后用附加分区对生成的多边形进行均匀化。 得到的算法计算效率高，具有良好的边界粘着特性

![](http://m.qpic.cn/psc?/V531KnY04NKhJG1WtBH80isWXv4SJLE6/45NBuzDIW489QBoVep5mcc4Gi.RxD*sElF9oFHKfReNbcU7XYUeRRZgbRS.i1XU1fYe.FNAmzNVRzcFwrMh9.*0IRGRVXF83TSJKKRMkd6o!/b&bo=IAPnAAAAAAAAJ8U!&rf=viewer_4)

我们执行的多边形划分依赖于SNIC超像素分割产生的边界。 每个超像素导致一个多边形。 多边形的创建要注意相邻的超像素共享相同的多边形边缘。 从最初开始,我们采取以下步骤：

1. 轮廓跟踪：使用标准轮廓跟踪算法跟踪沿每个超像素边界的封闭路径。 这将产生沿超像素边界的有序像素位置序列。
2. 初始顶点：由于相邻的超像素共享边界，因此选择了一些公共顶点。 沿接触至少三个超像素或至少两个超像素和图像边界的边界路径的所有像素位置作为初始共享顶点。 此外，图像的角被视为顶点。
3. 附加顶点：现在我们简化了两个顶点之间的路径段。 对于两个顶点之间的每个路径段，我们使用Douglas-Peucker算法添加新的顶点。 这将路径段从几个像素位置简化为几个多边形顶点。
4. 顶点合并：根据超像素大小，根据阈值（期望超像素半径的十分之一）被认为彼此非常接近的顶点被分配一个公共顶点。 这个公共顶点被选择为具有最高图像梯度大小的顶点。
5. 多边形生成：最后，通过将迄今获得的顶点与直线段连接起来，得到多边形

在创建多边形之后，我们根据多边形边界重新命名像素。 创建多边形和分配新标签的整个过程只需要比最初的SNIC分割多20%的时间。 这使得我们的多边形分割过程SNIC POLY比CONPOLY得更快。 作为说明，虽然我们依赖于SNIC超像素，但本节提出的多边形分割算法也可以为使用不同算法获得的超像素生成多边形。



与CONPOLY[12]不同的是，我们的一些多边形可以是非凸的，特别是对于自然图像。 如果凸多边形或三角形是应用[9]所必需的，则可以在非凸多边形内部添加边，使它们凸起来。

Arguments:










neighborhoodSize (Integer, default: null):[^9]
Tile neighborhood size (to avoid tile boundary artifacts). Defaults to 2 * size.

[^1]:阈值分割法，边缘检测法，区域分割法，直方图法





[^2]:
[^3]:  8![8](http://m.qpic.cn/psc?/V531KnY04NKhJG1WtBH80isWXv4SJLE6/45NBuzDIW489QBoVep5mcf151KaZLH0s6fBCGNn3xqEo*elV2FMOOi1n33V8BatDAj4KMToJlIp04BhYDhjacgmz5GQ3cKqNAEqjMQsMoKM!/b&bo=XgFSAQAAAAAAFz0!&rf=viewer_4)![](http://m.qpic.cn/psc?/V531KnY04NKhJG1WtBH80isWXv4SJLE6/45NBuzDIW489QBoVep5mcYktlgHoJhb2sTjGcta2*L2v7*k*Z2wT5GnKfuaK2ZWx8nEG67vtSv8*zxHCU5sa5KlP0aZ4Yvx2RhoXclTFzq4!/b&bo=YgFYAQAAAAAAFws!&rf=viewer_4)
[^5]:图片（图片）： 用于群集的输入图像。
[^6]:大小（整数，默认值：5）： 超像素种子位置间隔，以像素为单位。如果提供“种子”图像，则不会产生网格。 
[^7]:紧凑度（Float，默认值：1）： 紧凑度系数。较大的值会导致群集更紧凑（正方形）。将此设置为0将禁用空间距离加权。
[^8]:连接（整数，默认值：8）： 连通性。 4或8。
[^9]:neighborhoodSize（整数，默认值：null）： 平铺邻域大小（避免平铺边界伪影）。默认为2 *大小。
[^10]:种子（图片，默认：null）： 如果提供，则将任何非零值像素用作种子位置。触摸的像素（由“连接性”指定）被视为属于同一群集。

