# Rumour的定义

这篇文章中，遵循一个普遍接受的Rumour的定义：

是一个未经核实的声明，在人与人之间流传，与公众关注的对象、事件或问题有关

它的流传在当前没有权威给予真实性，但它可能被证明是真实的，或部分或完全虚假；或者，它也可能仍未得到解决。

# Model（ERD）

1. Rumour Detection Module ：classifies whether an event constitutes a rumour
2. Checkpoint  Module : determines when to trigger the rumour detection module

之前的一些模型，要么没有包含timeline信息，要么就是所有的谣言都使用一个checkpoint，无法针对每个谣言进行特异性检测，ERD model的第二个模块整合了强化学习，可以针对每个谣言进行动态的决定checkpoint，这是ERD的创新点

## Model Architecture

$E-事件\quad from\quad x_0...x_T\quad x_i-relevant\quad posts$

目标：在保证准确率的前提下，尽早的识别出$E$是谣言

## RDM

包含3层：

1. Word Embedding Layer
2. Max-pooling Layer
3. GRU

首先：Word Embedding Layer把post  $x_i$  转化成 vector，$e_i^j$ 代表第$i$个post的第$j$个单词

之后，max-pooling提取重要特征:
$$
m_i= maxpool([W_me_i^0,W_me_i^1...,W_me_i^K])
$$


之后，GRU获取时序信息
$$
hi = GRU(mi, h_{i-1})
$$
利用final state$h_N$ 去进行softmax分类
$$
p = softmax(W_ph_N + b_p)
$$


## Checkpoint Module (CM)

CM learns the number of posts needed to trigger RDM

利用深度强化学习，基于RDM的准确率，每次奖励CM，并且每次稍微的惩罚CM，通过这种方式，CM可以达到在检测精度和时间上的权衡。

使用的是deep Q-learning model
$$
Q^*(s,a)=Es'\varepsilon[r+\gamma \max\limits_{a'}Q_i(s',a')|s,a]
$$
$r$ :reward value.       $\gamma$: the discount rate

![image-20220428121020093](/Users/chenguanjin/Library/Application Support/typora-user-images/image-20220428121020093.png)

### CM model

 Input :RDM中GRU的hidden status

通过两层前馈神经网络，得到action-value
$$
a_i = W_a(ReLu(W_hh_i + b_h)) + b_a 
$$

## Joint Training

一起训练RDM与CM，过程类似生成对抗网络，不同点是两个module是合作而不是对抗

RDM使用交叉熵损失，在训练CM时，保持RDM的参数固定，二者交替训练，一个训练若干轮次后换下一个

## Bucketing Strategy

post不是一个个进入，而是batch进入

有三种batch方法：

1. FN(a fixed number of posts) 每三个一组 最终选FN，因为实验之后发现FN识别准确率最高

2. FT(a fixed time interval) 每两个小时一组

3. DI(dynamic interval) 确保在一个区间内收集的帖子数量接近完整数据集中一小时内收集的帖子数量。

   

   最终选FN，因为实验之后发现FN识别准确率最高

   ![image-20220428124041854](/Users/chenguanjin/Library/Application Support/typora-user-images/image-20220428124041854.png)

# Experiment

## Data Set

采用weibo、Twitter数据集，10%作为验证集来进行超参数调整，剩下的3:1来分训练、测试集

## Model Comparison

baseline ：SVM with tf-idf features

以及其他的一些SOTA model：

- CSI on WEIBO
- CRF and HMM on TWITTER
- GRU-2 on both data sets
- GRU-2 的变体：simple RNN 、single-layer LSTM 、GRU-1

## Preprocessing and Hyper-parameters

### Tokenisation

TWITTER：Words are tokenised using white spaces、 stopword list is based on NLTK

WEIBO ：Jieba  、stopword list is a customised list

### Embedding

WEIBO：word2vec

TWITTER：Glove

Unknown words are initialised as zero vectors.

在模型训练时，Embedding fixed

### 超参数

$\theta$:   0.01

$\gamma$:   0.95

Optim: Adam   lr=0.001

