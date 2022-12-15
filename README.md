README
===========================

Olfactory

## Contents

* [Data Curation](#DataCuration)
    * Expert-Labeled Dataset
    * 单行文本
* [Featurization of Molecules](#Featurization)
    * 来源于网络的图片
    * GitHub仓库中的图片
* [Models](#Models)
    * MPNN Model for Graph Features


## Data Curation
### Expert-Labeled Dataset
We assembled an expert-labeled set of 5596 molecules from three separate sources in [Pyrfume](https://github.com/pyrfume) Project: the [GoodScents](https://github.com/pyrfume/pyrfume-data/tree/main/goodscents) database (n = 6157), the [Leffingwell PMP 2001](https://github.com/pyrfume/pyrfume-data/tree/main/leffingwell) database (n = 3523), and the [Ifra 2019](https://github.com/pyrfume/pyrfume-data/tree/main/ifra_2019) database (n = 1135).


The datasets share 2317 overlapping molecules. 
Molecules are labeled with one or more odor descriptors by olfactory experts (usually a practicing perfumer), creating a multi-label prediction problem. 
GoodScents describes a list of 1–15 odor descriptors for each molecule (Figure 3A), whereas Leffingwell uses free-form text. Odor descriptors were canonicalized using the GoodScents ontology, and overlapping molecules inherited the union of both datasets’ odor descriptors. After
filtering for odor descriptors with at least 30 representative molecules, 138 odor descriptors remained
(Figure 3B), including an odorless descriptor. 
Some odor descriptors were extremely common, like fruity or green, while others were rare, like radish or bready. This dataset is composed of materials
for perfumery, and so is biased away from malodorous compounds. 

|Type of Problem|Example|Strategy|
|----|-----|-----|
|Synonymous|‘cedar’ / ‘cedarwood’|unify|
|Mixed|‘toasted meat’|discard|
|Incomprehensible|‘white’|discard|
|Too broad|‘nice’|refer to Google's work|
|Chemical|‘pyrazine’|refer to Google's work|


There is also skew in label counts resulting from different levels of specificity, e.g. fruity will always be more common than pineapple.

![Image text](StatFigures/DataCurationStrategy.png)



|Dataset|Number of molecules|Number of original labels|Number of calibrated labels|
|----|-----|-----|----|
|Goodscents|6157 to 6128|679|217||
|Leffingwell|3523 to 3430|114|109|
|Ifra|1135|190|162|


### 单行文本
    Hello,大家好，我是果冻虾仁。
在一行开头加入1个Tab或者4个空格。

## Featurization of Molecules
基本格式：
```
![alt](URL title)
```
alt和title即对应HTML中的alt和title属性（都可省略）：
- alt表示图片显示失败时的替换文本
- title表示鼠标悬停在图片时的显示文本（注意这里要加引号）

URL即图片的url地址，如果引用本仓库中的图片，直接使用**相对路径**就可了，如果引用其他github仓库中的图片要注意格式，即：`仓库地址/raw/分支名/图片路径`，如：
```
https://github.com/guodongxiaren/ImageCache/raw/master/Logo/foryou.gif
```

## Models
### MPNN Model for Graph Features

<div align="center">

| 表头1  | 表头2|
| ---------- | -----------|
| 表格单元   | 表格单元   |
| 表格单元   | 表格单元   |

</div>

其他任意需要居中展示的语法，都可以放在其中。


