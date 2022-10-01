# Lab 1 - 框架及工具入门示例

## 实验目的

1. 了解深度学习框架及工作流程（Deep Learning Workload）
2. 了解在不同硬件和批大小（batch_size）条件下，张量运算产生的开销


## 实验环境

* PyTorch==1.5.0

* TensorFlow>=1.15.0

* 【可选环境】 单机Nvidia GPU with CUDA 10.0


## 实验原理

通过在深度学习框架上调试和运行样例程序，观察不同配置下的运行结果，了解深度学习系统的工作流程。

## 实验内容

### 实验流程图

![](/imgs/Lab1-flow.png "Lab1 flow chat")

### 具体步骤

1.	安装依赖包。PyTorch==1.5, TensorFlow>=1.15.0

2.	下载并运行PyTorch仓库中提供的MNIST样例程序。

3.	修改样例代码，保存网络信息，并使用TensorBoard画出神经网络数据流图。

4.	继续修改样例代码，记录并保存训练时正确率和损失值，使用TensorBoard画出损失和正确率趋势图。

5.	添加神经网络分析功能（profiler），并截取使用率前十名的操作。

6.	更改批次大小为1，16，64，再执行分析程序，并比较结果。

7.	【可选实验】改变硬件配置（e.g.: 使用/ 不使用GPU），重新执行分析程序，并比较结果。


## 实验报告

### 实验环境

||||
|--------|--------------|-------------------------------------------------------|
|硬件环境|CPU（vCPU数目）| Number of Processors:	1 Total Number of Cores:	2 |
||GPU(型号，数目)|NA||
|软件环境|OS版本|Darwin lijiahaodeMacBook-Pro-2.local 21.6.0 Darwin Kernel Version 21.6.0: Mon Aug 22 20:17:10 PDT 2022; root:xnu-8020.140.49~2/RELEASE_X86_64 x86_64||
||深度学习框架<br>python包名称及版本|$ python3 -c "import torch; print(torch.__version__)" 1.9.1  
|||$ python3 -c "import tensorflow as tf; print(tf.__version__)"2.4.0||
||CUDA版本|NA||
||||

### 实验结果

1. 模型可视化结果截图
   
|||
|---------------|---------------------------|
|<br/>&nbsp;<br/>神经网络数据流图<br/>&nbsp;<br/>&nbsp;|![](/Labs/BasicLabs/Lab1//img/nn_dataflow.png "Lab1 flow chart")|
|<br/>&nbsp;<br/>损失和正确率趋势图<br/>&nbsp;<br/>&nbsp;|![](/Labs/BasicLabs/Lab1//img/loss_acc.png "Lab1 loss and accuracy")|
|<br/>&nbsp;<br/>网络分析，使用率前十名的操作<br/>&nbsp;<br/>&nbsp;|![](/Labs/BasicLabs/Lab1//img/nn_profiling.png "Lab1 nn profiling")|
||||


1. 网络分析，不同批大小结果比较

|||
|------|--------------|
|批大小 &nbsp;| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 结果比较 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
|<br/>&nbsp;<br/>1<br/>&nbsp;<br/>&nbsp;|NA||
|<br/>&nbsp;<br/>16<br/>&nbsp;<br/>&nbsp;|NA||
|<br/>&nbsp;<br/>64<br/>&nbsp;<br/>&nbsp;|NA||
|||

## 参考代码

1.	MNIST样例程序：

    代码位置：Lab1/mnist_basic.py

    运行命令：`python mnist_basic.py`

2.	可视化模型结构、正确率、损失值

    代码位置：Lab1/mnist_tensorboard.py

    运行命令：`python mnist_tensorboard.py`

3.	网络性能分析

    代码位置：Lab1/mnist_profiler.py

## 参考资料

* 样例代码：[PyTorch-MNIST Code](https://github.com/pytorch/examples/blob/master/mnist/main.py)
* 模型可视化：
  * [PyTorch Tensorboard Tutorial](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html) 
  * [PyTorch TensorBoard Doc](https://pytorch.org/docs/stable/tensorboard.html)
  * [pytorch-tensorboard-tutorial-for-a-beginner](https://medium.com/@rktkek456/pytorch-tensorboard-tutorial-for-a-beginner-b037ee66574a)
* Profiler：[how-to-profiling-layer-by-layer-in-pytroch](https://stackoverflow.com/questions/53736966/how-to-profiling-layer-by-layer-in-pytroch)


