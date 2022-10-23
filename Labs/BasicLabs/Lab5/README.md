# Lab 5 - 配置Container进行云上训练或推理

## 实验目的

1. 理解Container机制
2. 使用Container进行自定义深度学习训练或推理

## 实验环境

* PyTorch==1.5.0
* Docker Engine

## 实验原理

计算集群调度管理，与云上训练和推理的基本知识

## 实验内容

### 实验流程图

![](/imgs/Lab5-flow.png "Lab5 flow chat")

### 具体步骤

1.	安装最新版Docker Engine，完成实验环境设置

2.	运行一个alpine容器

    1. Pull alpine docker image
    2. 运行docker container，并列出当前目录内容
    3. 使用交互式方式启动docker container，并查看当前目录内容
    4. 退出容器

3.	Docker部署PyTorch训练程序，并完成模型训练

    1. 编写Dockerfile：使用含有cuda10.1的基础镜像，编写能够运行MNIST样例的Dockerfile
    2. Build镜像
    3. 使用该镜像启动容器，并完成训练过程
    4. 获取训练结果

4.	Docker部署PyTorch推理程序，并完成一个推理服务

    1. 克隆TorchServe源码
    2. 编写基于GPU的TorchServe镜像
    3. 使用TorchServe镜像启动一个容器
    4. 使用TorchServe进行模型推理
    5. 返回推理结果，验证正确性


## 实验报告

### 实验环境

||||
|--------|--------------|--------------------------|
|硬件环境|CPU（vCPU数目）|&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
||GPU(型号，数目)||
|软件环境|OS版本||
||深度学习框架<br>python包名称及版本||
||CUDA版本||
||||

### 实验结果

1.	使用Docker部署PyTorch MNIST 训练程序，以交互的方式在容器中运行训练程序。提交以下内容：

    1. 创建模型训练镜像，并提交Dockerfile
    #docker build  -f mnist_docker_train -t docker_model_train .
    [+] Building 1.4s (14/14) FINISHED                                                                                                                     
 => [internal] load build definition from mnist_docker_train                                                                                      0.0s
 => => transferring dockerfile: 45B                                                                                                               0.0s
 => [internal] load .dockerignore                                                                                                                 0.0s
 => => transferring context: 2B                                                                                                                   0.0s
 => [internal] load metadata for docker.io/library/ubuntu:18.04                                                                                   1.3s
 => [auth] library/ubuntu:pull token for registry-1.docker.io                                                                                     0.0s
 => [1/8] FROM docker.io/library/ubuntu:18.04@sha256:40b84b75884ff39e4cac4bf62cb9678227b1fbf9dbe3f67ef2a6b073aa4bb529                             0.0s
 => [internal] load build context                                                                                                                 0.0s
 => => transferring context: 44B                                                                                                                  0.0s
 => CACHED [2/8] RUN mkdir -p /src/app                                                                                                            0.0s
 => CACHED [3/8] WORKDIR /src/app                                                                                                                 0.0s
 => CACHED [4/8] COPY pytorch_mnist_basic.py /src/app                                                                                             0.0s
 => CACHED [5/8] RUN apt-get update && apt-get install wget bzip2 -y                                                                              0.0s
 => CACHED [6/8] RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh                                   0.0s
 => CACHED [7/8] RUN bash miniconda.sh -b -p /opt/conda                                                                                           0.0s
 => CACHED [8/8] RUN conda install pytorch torchvision cpuonly -c pytorch                                                                         0.0s
 => exporting to image                                                                                                                            0.0s
 => => exporting layers                                                                                                                           0.0s
 => => writing image sha256:e5af7a163762f3d32559403193708bd12db07e18a58bbcef3f495653a6ee8181                                                      0.0s
 => => naming to docker.io/library/docker_model_train                       
    2. 提交镜像构建成功的日志
    docker run -p 80:80 --name training docker_model_train
    3. 启动训练程序，提交训练成功日志（例如：MNIST训练日志截图）
    ![](/Labs/BasicLabs/Lab5/img/docker_mnist_train_start.png "Lab5 Docker Training Start")
    Docker Training Start
    ![](/Labs/BasicLabs/Lab5/img/docker_mnist_train_end.png "Lab5 Docker Training End")
    Docker Training End
<br/>

<br/>

<br/>

<br/>

<br/>

2.	使用Docker部署MNIST模型的推理服务，并进行推理。提交以下内容：
    1. 创建模型推理镜像，并提交Dockerfile
    #docker build --file Dockerfile.infer.cpu -t torchserve:0.1-cpu .
    1. 启动容器，访问TorchServe API，提交返回结果日志
    #docker run --rm -it -p 8080:8080 -p 8081:8081 torchserve:0.1-cpu
    WARNING: sun.reflect.Reflection.getCallerClass is not supported. This will impact performance.
    2022-10-23T20:57:34,081 [INFO ] main org.pytorch.serve.servingsdk.impl.PluginsManager - Initializing plugins manager...
    2022-10-23T20:57:34,259 [INFO ] main org.pytorch.serve.ModelServer - 
    Torchserve version: 0.6.0
    TS Home: /usr/local/lib/python3.7/dist-packages
    Current directory: /home/model-server
    Temp directory: /home/model-server/tmp
    Number of GPUs: 0
    Number of CPUs: 2
    Max heap size: 984 M
    Python executable: /usr/bin/python3
    Config file: /home/model-server/config.properties
    Inference address: http://0.0.0.0:8080
    Management address: http://0.0.0.0:8081
    Metrics address: http://127.0.0.1:8082
    Model Store: /home/model-server/model-store
    Initial Models: N/A
    Log dir: /home/model-server/logs
    Metrics dir: /home/model-server/logs
    Netty threads: 32
    Netty client threads: 0
    Default workers per model: 2
    Blacklist Regex: N/A
    Maximum Response Size: 6553500
    Maximum Request Size: 6553500
    Limit Maximum Image Pixels: true
    Prefer direct buffer: false
    Allowed Urls: [file://.*|http(s)?://.*]
    Custom python dependency for model allowed: false
    Metrics report format: prometheus
    Enable metrics API: true
    Workflow Store: /home/model-server/model-store
    Model config: N/A
    1. 使用训练好的模型，启动TorchServe，在新的终端中，使用一张图片进行推理服务。提交图片和推理程序返回结果截图。
   #docker ps
    CONTAINER ID   IMAGE                COMMAND                  CREATED         STATUS         PORTS                              NAMES
    8a14460188d5   torchserve:0.1-cpu   "/usr/local/bin/dock…"   2 minutes ago   Up 2 minutes   0.0.0.0:8080-8081->8080-8081/tcp   thirsty_leakey
   #docker exec -it 8a14460188d5 /bin/bash
   #history 16
    7  cd  /home/model-server/model-store/
    8  apt-get update  
    9  apt-get install wget
   10  wget https://download.pytorch.org/models/densenet161-8d451a50.pth
   11  cd /serve/model-archiver
   12  pip install .
   13  LS
   14  ls
   15  torch-model-archiver --model-name densenet161 --version 1.0 --model-file /serve/examples/image_classifier/densenet_161/model.py --serialized-file /home/model-server/model-store/densenet161-8d451a50.pth --export-path /home/model-server/model-store --extra-files /serve/examples/image_classifier/index_to_name.json --handler image_classifier
   16  ls
   17  torchserve --stop
   18  cd /home/model-server/
   19  torchserve --start --ncs --model-store model-store --models densenet161.mar
   20  history 10
   21  history 20
   22  history 16
    ![](/Labs/BasicLabs/Lab5/img/Kitten_Inference.png "Lab5 Docker Inference")
    Docker Inference

<br/>

<br/>

<br/>

<br/>

<br/>

## 参考代码

本次实验基本教程:

* [1. 实验环境设置](./setup.md)
* [2. 运行你的第一个容器 - 内容，步骤，作业](./alpine.md)
* [3. Docker部署PyTorch训练程序 - 内容，步骤，作业](./train.md)
* [4. Docker部署PyTorch推理程序 - 内容，步骤，作业](./inference.md)
* [5. 进阶学习](./extend.md)
* [6. 常见问题](./issue.md)

## 参考资料

* [Docker Tutorials and Labs](https://github.com/docker/labs/)
* [A comprehensive tutorial on getting started with Docker!](https://github.com/prakhar1989/docker-curriculum)
* [Please-Contain-Yourself](https://github.com/dylanlrrb/Please-Contain-Yourself)
* [Create TorchServe docker image](https://github.com/pytorch/serve/tree/master/docker)


