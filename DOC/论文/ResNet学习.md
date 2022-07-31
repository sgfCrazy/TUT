@[TOC](ResNet学习)
# ResNet
**论文链接**[link](https://arxiv.org/pdf/1512.03385.pdf)
神经网络通过堆叠更多的层，可以让模型变得更加庞大复杂，但庞大复杂的模型不一定能比小模型取得更好的效果，且训练起来也很难收敛。ResNet的主要思想是，在堆叠了更多的层之后，要能够保证网络起码不会变的更差。即堆叠的新层的网络的能力应该要大于等于堆叠之前的。主要的贡献是提出了短链接的方式，对后来的神经网络模型影响深远。ResNet是由微软研究院的何恺明、张祥雨、任少卿、孙剑等人提出，并取得了2015 年的ILSVRC（ImageNet Large Scale Visual Recognition Challenge）比赛的冠军。
## 1. 网络结构
先放一个论文中的图
![在这里插入图片描述](https://img-blog.csdnimg.cn/50a5c7f7f48a403386b48088de72030b.png#pic_center)
通常来说，resnet有5种不同的层结构，如上图所示。每种层又基于两种基本的block，块结构，分别为BasicBlock和Bottleneck。再放一张图文中的图。
![在这里插入图片描述](https://img-blog.csdnimg.cn/68637dd31e344576a7e77e0bd8e13409.png)
不同的层即为BasicBlock或Bottleneck堆叠而成。所谓短链接即块结构中弧形线的部分。
