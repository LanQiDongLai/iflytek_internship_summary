# TorchServe

## TorchServe 安装部署

#### Pip安装

```bash
# 需要提前安装Java环境
pip install torchserve
```

#### Conda安装

```bash
# 需要提前安装Java环境
conda install torchserve -c pytorch
```

#### Docker安装

```bash
docker pull torchserve
```

#### 启动 TorchServe

1. –ts-config {config-path}(可选):如果TS_CONFIG_FILE环境并没有被设置的话，定义配置文件config.properties路径
2. --model-store:定义或覆盖config.properties中的model_store路径
3. --models:定义或覆盖config.properties中的load_models路径
4. –log-config:覆盖默认的log4j2.xml
5. -foreground:TorchServe启动后，运行在前台而非后台

```bash
# 启动
torchserve --start --model-store /home/Lan/model_store
# 测试正常运行
curl http://localhost:8080/ping
```

应当返回

```json
{
  "status": "Healthy"
}
```

#### 关闭TorchServe

```bash
torchserve --stop
```

#### TorchServe启动配置

配置导入config.properties

1. 设置TS_CONFIG_FILE环境变量，其值作为config.properties的路径
2. 若没有设置，则使用启动命令选项-ts-config所指定的路径
3. 若没有设置，则使用TorchServe的同级目录下的config.properties
4. 若以上都没有配置，则TorchServe使用默认配置

```properties
# 定义推理端口
inference_address=http://0.0.0.0:8080
# 定义控制端口
management_address=http://0.0.0.0:8081
# 定义度量端口
metrics_address=http://0.0.0.0:8082
# 线程数量
number_of_netty_threads=32
# 工作队列大小
job_queue_size=1000
# 模型目录
model_store=/home/model-server/model-store
# 工作目录
workflow_store=/home/model-server/wf-store
```



## TorchServe gRPC API

> TorchServe Google Remote Procedure Call API
>
> TorchServe远程系统调用API（基于RESTful API）

### 管理部分

#### 检查运行状态

```bash
curl http://localhost:8080/ping
```

```json
{
  "status": "Healthy"
}
```

#### 检查所有已装载模型

```bash
curl http://localhost:8081/models
```

```json
{
  "models": [
    {
      "modelName": "mnist",
      "modelUrl": "/home/model-server/model-store/mnist.mar"
    }
  ]
}
```

#### 网络上传模型

1. url:模型存档下载地址. 支持以下位置
   1. 本地文件地址，文件必须在model-store文件夹下的一级目录中
   2. HTTP(s)协议地址，文件可以通过TorchServe在互联网上下载
2. model_name(可选):模型名称
3. handler(可选):推理处理程序的入口
4. runtime(可选):模型网络或推理服务代码的运行环境，可选python,python2,python3
5. batch_size(可选):推理时批处理数目，默认为1
6. max_batch_delay(可选):批量处理聚合的最大延迟，默认为100ms
7. initial_workers:进程创建数目，默认为0，TorchServe推理至少需要1个进程
8. synchronous(可选):进程的创建是否同步，默认为否，TorchServe将创建新的进程，而无需等待前一进程已联机的确认
9. response_timeout(可选):如果模型的后端程序在此超时期内没有响应推理响应，进程将会被重新启动，单位为s，默认为120s

```bash
curl -X POST "http://localhost:8081/models?initial_workers=1&&url=/home/model-server/model-store/mnist.mar"
```

```json
{
  "status": "Model \"mnist\" Version: 1.0 registered with 1 initial workers"
}
```

#### 描述模型

1. modelName:模型名称
2. modelVersion:模型版本
3. modelUrl:模型地址
4. runtime:模型网络或推理服务代码的运行环境
5. minWorkers:最小工作数目
6. maxWorkers:最大工作数目
7. batchSize:批处理数目
8. maxBatchDelay:批量处理聚合的最大延迟
9. loadedAtStartup:启动时加载
10. workers:工作进程列表
    1. id:ID号
    2. startTime:开始时间
    3. status:当前状态
    4. memoryUsage:已使用的内存（单位byte）
    5. pid:进程号
    6. gpu:是否使用GPU
    7. gpuUsage:所使用的GPU

```bash
curl http://localhost:8081/models/mnist
```

```json
[
  {
    "modelName": "mnist",
    "modelVersion": "1.0",
    "modelUrl": "/home/model-server/model-store/mnist.mar",
    "runtime": "python",
    "minWorkers": 1,
    "maxWorkers": 1,
    "batchSize": 1,
    "maxBatchDelay": 100,
    "loadedAtStartup": false,
    "workers": [
      {
        "id": "9000",
        "startTime": "2023-07-01T06:31:37.034Z",
        "status": "READY",
        "memoryUsage": 285503488,
        "pid": 117,
        "gpu": false,
        "gpuUsage": "N/A"
      }
    ]
  }
]
```

#### 取消已装载的模型

```bash
curl -X DELETE http://localhost:8081/models/mnist
```

```json
{
  "status": "Model \"mnist\" unregistered"
}
```

#### 调节工作进程数量

1. min_worker:工作进程最小数量，TorchServe尝试为指定的模型保持工作进程的最小数量，默认为1
2. max_worker:工作进程最大数量，TorchServe保持工作进程数目不超过工作进程的最大数量，默认和min_worker相同
3. timeout:工作进程完成所有挂起请求的指定等待时间，如果超过，则终止该工作进程，如果为0，则立刻终止，如果为-1，则无限期等待，默认为-1

```bash
curl -X PUT "http://localhost:8081/models/mnist?min_worker=3"
```

### 用户部分

#### 请求推理预测

```bash
curl http://localhost:8080/predictions/mnist -T ./Docker/mnist_images/0.png
```

```
0
```

#### 检查所有已装载模型

```bash
curl http://localhost:8081/models
```

```json
{
  "models": [
    {
      "modelName": "mnist",
      "modelUrl": "/home/model-server/model-store/mnist.mar"
    }
  ]
}
```

## 如何在TorchServe中部署大型模型

### 原理

TorchServe能够让多个工作进程处理一个大型模型，默认情况下，TorchServe使用循环算法将GPU资源分配给每个工作进程，分配给每个工作人员的GPU是根据model_config.yaml中指定的GPU数量自动计算的

用于大型模型推理的PyTorch原生解决方案(Pippy)，并不适合在只有一个GPU设备上对大模型推理提供了流水线并行性

他根据模型，并将其拆分为相等的大小或阶段，根据您指定的设备数量进行分区，然后使用微批量来运行您的批量输入进行推理，所以请将batch_size 设置大于CPU数来得到更好的优化

### 如何使用

要在Torchserve中使用Pippy，我们需要新建自定义处理handler文件，并设置类继承于base_Pippy_handler，并将我们的设置放入model-config.yaml中

其中，自定义处理handler文件只是一个python脚本，它定义了特定于工作流的模型加载、预处理、推理和后处理逻辑。

```python
# 导入必要的包
from ts.torch_handler.distributed.base_pippy_handler import BasePippyHandler
from ts.handler_utils.distributed.pt_pippy import initialize_rpc_workers, get_pipline_driver
class ModelHandler(BasePippyHandler, ABC):
    def __init__(self):
        super(ModelHandler, self).__init__()
        self.initialized = False

    def initialize(self, context):
        model = # 从你的model文件夹中导入模型
        self.device = self.local_rank %  torch.cuda.device_count()# 被用来将所有输入数据传输到指定设备(self.device)
        self.model = get_pipline_driver(model,self.world_size, ctx)
```

以下为model-config.yaml的配置所需要的参数，这个配置文件非常灵活，您可以添加与前端、后端和处理程序相关的设置。

```yaml
#frontend settings
minWorkers: 1
maxWorkers: 1
maxBatchDelay: 100
responseTimeout: 120
deviceType: "gpu"
parallelType: "pp" # options depending on the solution, pp(pipeline parallelism), tp(tensor parallelism), pptp ( pipeline and tensor parallelism)
                   # This will be used to route input to either rank0 or all ranks from fontend based on the solution (e.g. DeepSpeed support tp, PiPPy support pp)
torchrun:
    nproc-per-node: 4 # specifies the number of processes torchrun starts to serve your model, set to world_size or number of
                      # gpus you wish to split your model
#backend settings
pippy:
    chunks: 1 # This sets the microbatch sizes, microbatch = batch size/ chunks
    input_names: ['input_ids'] # input arg names to the model, this is required for FX tracing
    model_type: "HF" # set the model type to HF if you are using Huggingface model other wise leave it blank or any other model you use.
    rpc_timeout: 1800
    num_worker_threads: 512 #set number of threads for rpc worker init.

handler:
    max_length: 80 # max length of tokens for tokenizer in the handler
```

将yaml中的内容导入到handler中

```python
def initialize(self, ctx):
    model_type = ctx.model_yaml_config["pippy"]["model_type"]
```

打包成mar文件，确保附加了yaml文件

```bash
torch-model-archiver --model-name bloom --version 1.0 --handler pippy_handler.py --extra-files $MODEL_CHECKPOINTS_PATH -r requirements.txt --config-file model-config.yaml --archive-format tgz
```

## Handler 文件详解

### 什么是Handler文件

Handler脚本自定义了TorchServe的行为，在和其他文件一起打包成mar文件后，TorchServe会在mar文件中执行handler脚本

执行分为以下几个步骤：

- 初始化模型实例
- 在将输入数据发送到模型进行推理或Captum解释之前对其进行预处理
- 自定义如何调用模型进行推理或解释
- 在发送回响应之前对模型的输出进行后期处理

### 开始

#### 创建一个类

你可以通过具有任何名称的类来创建自定义处理程序，但它必须具有initialize和handle方法

如果一个Handler脚本文件中有很多类，请保证handler类是列表中的第一个

下面是handler类的模板：

```python
class ModelHandler(object):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None

    def initialize(self, context):
        """
        Invoke by torchserve for loading a model
        :param context: context contains model server system properties
        :return:
        """

        #  load the model
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        self.model = torch.jit.load(model_pt_path)

        self.initialized = True


    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        pred_out = self.model.forward(data)
        return pred_out
```

在这里我们可以看到ModelHandler类中基本的结构

#### ModelHandler类的执行顺序

- \__init__(self):类的构造函数
- initialize(self, context):初始化
- handle(self, data, context):预处理，推理和后处理过程，三个步骤经常写在别的三个函数并在此函数中调用
  - preprocess(self, data):预处理过程
  - inference(self, model_input):推理过程
  - postprocess(self, data):后处理过程

除此之外，你可以选择继承BaseHandler，BaseHandler实现了您需要的大部分功能，大多数时候，您只需要覆盖预处理或后处理

#### TorchServe预置类

同时，TorchServe也提供了其他类，帮助完成了相关的操作，以下是他们的继承关系及其应用说明

Basehandler:基础处理过程

1. VisionHandler:图像基础处理过程
   1. ObjectDetector:物品追踪
   2. ImageSegmenter:图像分割
   3. ImageClassifier:图像分类
2. TextHandler:文本基础处理过程
3. DenseNetHandler:密集网络处理过程

下面是MNIST手写数字处理数据集的handler文件样例

```python
from torchvision import transforms
from ts.torch_handler.image_classifier import ImageClassifier
from torch.profiler import ProfilerActivity


class MNISTDigitClassifier(ImageClassifier):
    """
    MNISTDigitClassifier handler class. This handler extends class ImageClassifier from image_classifier.py, a
    default handler. This handler takes an image and returns the number in that image.

    Here method postprocess() has been overridden while others are reused from parent class.
    """

    image_processing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    def __init__(self):
        super(MNISTDigitClassifier, self).__init__()
        self.profiler_args = {
            "activities" : [ProfilerActivity.CPU],
            "record_shapes": True,
        }

	# 将data列表中的数据取最大的（预测结果中匹配度最高的并以列表进行输出）
    def postprocess(self, data):
        """The post process of MNIST converts the predicted output response to a label.

        Args:
            data (list): The predicted output from the Inference with probabilities is passed
            to the post-process function
        Returns:
            list : A list of dictionaries with predictions and explanations is returned
        """
        return data.argmax(1).tolist()
```



#### 多GPU处理

TorchServe在vCPU或GPU上扩展后端工作程序。在多个gpu的情况下，TorchServe以循环方式选择gpu设备，并将此设备id传递给上下文对象中的模型处理程序。用户应使用此GPU ID创建pytorch设备对象，以确保不是在同一GPU中创建所有工作进程。以下代码片段可以在模型处理程序中用于创建PyTorch设备对象

```python
import torch

class ModelHandler(object):
    """
    A base Model handler implementation.
    """

    def __init__(self):
        self.device = None

    def initialize(self, context):
        properties = context.system_properties
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
```

## 动态升级模型

TorchServe支持不停止服务更新模型，只需要将原先的模型从服务中取消装载，在重写装载一个新版本的mnist模型即可

```bash
curl -X DELETE http://localhost:8081/models/mnist
curl -X POST "http://localhost:8081/models?initial_workers=1&&url=/home/model-server/model-store/mnist.mar"
```

或者在添加新版本的model时添加一个新的版本号，这样在访问的时候可以通过版本号来对相同的模型进行选择

```bash
# model_archiver 生产打包模型阶段
torch-model-archiver --model-name mnist --version 2.0 --model-file./mnist2.py --serialized-file ./mnist2_cnn.pt --handler ./mnist2_handler.py 
# 部署模型
curl -X POST "http://localhost:8081/models?initial_workers=1&&url=/home/model-server/model-store/mnist2.mar"
# 访问模型
curl http://localhost:8080/predictions/mnist/2.0 -T ./Docker/mnist_images/0.png
```

