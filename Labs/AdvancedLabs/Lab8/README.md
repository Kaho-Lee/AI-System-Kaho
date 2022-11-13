# Lab 8 - 自动机器学习系统练习

## 实验目的

通过试用 NNI 了解自动机器学习，熟悉自动机器学习中的基本概念

## 实验环境

* Ubuntu
* Python==3.7.6
* NNI==1.8
* PyTorch==1.5.0

## 实验原理

在本实验中，我们将处理 CIFAR-10 图片分类数据集。基于一个表现较差的基准模型和训练方法，我们将使用自动机器学习的方法进行模型选择和优化、超参数调优，从而得到一个准确率较高的模型。

## 实验内容

### 实验流程图

![](/imgs/Lab8-flow.png "Lab8 flow chat")

### 具体步骤

1. 熟悉 PyTorch 和 CIFAR-10 图像分类数据集。可以先阅读教程：https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html 
   
2. 熟悉 NNI 的基本使用。阅读教程：https://nni.readthedocs.io/en/latest/Tutorial/QuickStart.html 
   
3. 运行CIFAR-10代码并观察训练结果。在实验目录下，找到 `hpo/main.py`，运行程序，记录模型预测的准确率。
   
4. 手动参数调优。通过修改命令行参数来手动调整超参，以提升模型预测准确率。记录调整后的超参名称和数值，记录最终准确率。
   
   **注：**
   main.py 暴露大量的命令行选项，可以进行调整，命令行选项可以直接从代码中查找，或通过 `python main.py -h` 查看。例如，`--model`（默认是 resnet18），`--initial_lr`（默认是 0.1），`--epochs`（默认是 300）等等。一种简单的方法是通过手工的方法调整参数（例如 `python main.py --model resnet50 --initial_lr 0.01`）然后根据结果再做调整。
5. 使用 NNI 加速参数调优过程。
   
    1. 参考NNI的基本使用教程，安装NNI（建议在Linux系统中安装NNI并运行实验）。
    2. 参照NNI教程运行 `mnist-pytorch` 样例程序（程序地址： https://github.com/microsoft/nni/tree/master/examples/trials/mnist-pytorch  ），测试安装正确性，并熟悉NNI的基本使用方法。
    3. 使用NNI自动调参功能调试hpo目录下CIFAR-10程序的超参。创建 `search_space.json` 文件并编写搜索空间（即每个参数的范围是什么），创建 `config.yml` 文件配置实验（可以视资源量决定搜索空间的大小和并行量），运行程序。在 NNI 的 WebUI 查看超参搜索结果，记录结果截图，并记录得出最好准确率的超参配置。
   
6.	（可选）上一步中进行的模型选择，是在若干个前人发现的比较好的模型中选择一个。此外，还可以用自动机器学习的方法选择模型，即网络架构搜索（NAS）。请参考nas目录下 `model.py`，采用 DARTS 的搜索空间，选择合适的 Trainer，进行搜索训练。记录搜索结果架构，并用此模型重新训练，记录最终训练准确率。

**注：** 搜索完成后得到的准确率并不是实际准确率，需要使用搜索到的模型重新进行单独的训练。具体请参考 NNI NAS 文档：https://nni.readthedocs.io/en/latest/nas.html


## 实验报告

### 实验环境

||||
|--------|--------------|--------------------------|
|硬件环境|CPU（vCPU数目）| Number of Processors:	1 Total Number of Cores:	2 |
||GPU(型号，数目)|NA||
|软件环境|OS版本|Darwin lijiahaodeMacBook-Pro-2.local 21.6.0 Darwin Kernel Version 21.6.0: Mon Aug 22 20:17:10 PDT 2022; root:xnu-8020.140.49~2/RELEASE_X86_64 x86_64||
||深度学习框架<br>python包名称及版本|python3 -c "import torch; print(torch.__version__)" 1.9.1;  nni==2.4
|||python3 -c "import tensorflow as tf; print(tf.__version__)"2.4.0||
||CUDA版本|NA||
||||

### 实验结果

1.	记录不同调参方式下，cifar10程序训练结果的准确率。

||||
|---------|-----------------|------------|
| 调参方式 | &nbsp; &nbsp; 超参名称和设置值 &nbsp; &nbsp; | &nbsp; &nbsp; 模型准确率 &nbsp; &nbsp; |
| &nbsp; <br /> &nbsp; 原始代码 &nbsp; <br /> &nbsp; |--model=resnet18 --epochs=30|76.9%|
| &nbsp; <br /> &nbsp; 手动调参 &nbsp; <br /> &nbsp; |NA|NA|
| &nbsp; <br /> &nbsp; NNI自动调参 &nbsp; <br /> &nbsp; |--model=resnet18 --epochs=30 --initial_lr 0.04608779092881806 --weight_decay 0.00011698805173194891 --optimizer "sgd" --grad_clip 1.6605854375598232|81.2%|
| &nbsp; <br /> &nbsp; 网络架构搜索 <br />&nbsp; &nbsp; （可选） <br /> &nbsp; |--epochs 30 --batch_size=64 learning rate and weight decay follow DARTS implementation|87%|
||||
2.	提交使用NNI自动调参方式，对 main.py、search_space.json、config.yml 改动的代码文件或截图。
    main.py

        --- a/Labs/AdvancedLabs/Lab8/hpo/main.py
        +++ b/Labs/AdvancedLabs/Lab8/hpo/main.py
        @@ -27,6 +27,7 @@ import os
        import pprint
        
        import numpy as np
        +import nni
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        @@ -145,26 +146,36 @@ def main(args):
            for epoch in range(1, args.epochs + 1):
                train(model, train_loader, criterion, optimizer, scheduler, args, epoch, device)
                top1, _ = test(model, test_loader, criterion, args, epoch, device)
        -    logger.info("Final accuracy is: %.6f", top1)
        +        nni.report_intermediate_result(top1)
        +        logger.debug('test accuracy %g', top1)
        +
        +    logger.debug("Final accuracy is: %.6f", top1)
        +    nni.report_final_result(top1)
        
        
        if __name__ == '__main__':
        -    available_models = ['resnet18', 'resnet50', 'vgg16', 'vgg16_bn', 'densenet121', 'squeezenet1_1',
        -                        'shufflenet_v2_x1_0', 'mobilenet_v2', 'resnext50_32x4d', 'mnasnet1_0']
        +    # available_models = ['resnet18', 'resnet50', 'vgg16', 'vgg16_bn', 'densenet121', 'squeezenet1_1',
        +    #                     'shufflenet_v2_x1_0', 'mobilenet_v2', 'resnext50_32x4d', 'mnasnet1_0']
            parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
            parser.add_argument('--initial_lr', default=0.1, type=float, help='initial learning rate')
            parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
            parser.add_argument('--ending_lr', default=0, type=float, help='ending learning rate')
            parser.add_argument('--cutout', default=0, type=int, help='cutout length in data augmentation')
            parser.add_argument('--batch_size', default=128, type=int, help='batch size')
        -    parser.add_argument('--epochs', default=300, type=int, help='epochs')
        +    parser.add_argument('--epochs', default=1, type=int, help='epochs')
            parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer type', choices=['sgd', 'rmsprop', 'adam'])
            parser.add_argument('--momentum', default=0.9, type=int, help='optimizer momentum (ignored in adam)')
            parser.add_argument('--num_workers', default=2, type=int, help='number of workers to preprocess data')
        -    parser.add_argument('--model', default='resnet18', choices=available_models, help='the model to use')
        +    parser.add_argument('--model', default='resnet18', help='the model to use')
            parser.add_argument('--grad_clip', default=0., type=float, help='gradient clip (use 0 to disable)')
            parser.add_argument('--log_frequency', default=20, type=int, help='number of mini-batches between logging')
            parser.add_argument('--seed', default=42, type=int, help='global initial seed')
        +    
            args = parser.parse_args()
        -
        -    main(args)
        +    tuner_params = nni.get_next_parameter()
        +    logger.debug(tuner_params)
        +    params = nni.utils.merge_parameter(args, tuner_params)
        +    logger.debug('tunner parameter ', params)
        +    main(params)

    config.yml

        experimentName: lab8-cifar10-hpo
        searchSpaceFile: search_space.json
        trialCommand: python main.py
        trialGpuNumber: 0
        trialCodeDirectory: .

        trialConcurrency: 2
        maxTrialNumber: 20
        tuner:
        name: TPE
        classArgs:
            optimize_mode: maximize
        trainingService:
        platform: local
        useActiveGpu: false

        assessor:
            Name: Medianstop
            classArgs:
            optimize_mode: maximize
            start_step: 4
    
    search_space.json

        {
            "initial_lr": {
                "_type": "uniform",
                "_value": [1e-4, 0.1]
            },
            "weight_decay": {
                "_type": "uniform",
                "_value": [1e-6, 1e-3]
            },
            "batch_size": {
                "_type": "choice",
                "_value": [128]
            },
            "epochs": {
                "_type": "choice",
                "_value": [8]
            },
            "optimizer": {
                "_type": "choice",
                "_value": ["adam", "rmsprop", "sgd"]
            },
            "model": {
                "_type": "choice",
                "_value": ["resnet18"]
            },
            "grad_clip": {
                "_type": "uniform",
                "_value": [0.0, 5.0]
            }
        }

<br />

<br />

3.	提交使用NNI自动调参方式，Web UI上的结果截图。

![](/Labs/AdvancedLabs/Lab8/img/NNI_HPO.png "Lab8 NNI HPO")

<br />

<br />


4.	（可选）提交 NAS 的搜索空间、搜索方法和搜索结果（得到的架构和最终准确率）。

Search Space can be found in nas/model.py, which is same as DARTS paper. One difference is the initial number of channel
for searching is truncated to 8 for time efficiency. The initial number of channel for re-traing searched model is truncated to 8

Search Method is DARTS with Second Order Optimization turned off

The final searched model can be found at nas/checkpoint.json


<br />

<br />

<br />

<br />

<br />


## 参考代码

### 自动调参

代码位置：`Lab8/hpo`

参考答案：`Lab8/hpo-answer`

### 网络架构搜索（NAS）

代码位置：`Lab8/nas`


## 参考资料

* Cifar10简介：https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html 
* NNI文档：https://nni.readthedocs.io/en/latest/ 
* NNI mnist-pytorch代码：https://github.com/microsoft/nni/tree/v1.9/examples/trials/mnist-pytorch
* NNI NAS 文档：https://nni.readthedocs.io/en/latest/nas.html 
* DARTS GitHub：https://github.com/quark0/darts 
