# model_archiver



```bash
torch-model-archiver --model-name mnist --version 1.0 --model-file examples/image_classifier/mnist/mnist.py --serialized-file examples/image_classifier/mnist/mnist_cnn.pt --handler examples/image_classifier/mnist/mnist_handler.py 
```

##### --model-name MODEL_NAME

> 导出MAR文件的名字，名字将命名为MODEL_NAME.mar，如果没有使用--export-path规定导出文件的位置，那么文件将保存在当前目录下

##### --serialized-file SERIALIZED_FILE

> 一个包含状态字典的pt或pth文件（待考证：此序列化文件若包含了模型的网络结构，则不需要--model-file来定义模型体系结构）

##### --model-file MODEL_FILE

> 定义模型体系结构的python文件路径，此文件只能包含一个从torch.nn.Mod ule扩展的类定义

```python
import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

```

--handler HANDLER

> HANDLER为默认处理程序名称或默认处理Python脚本文件名称，用于处理自定义的推理逻辑

规范

1. 继承基类Basehandler或其派生类
2. ```def initialize(self, context):```模型初始化
3. ```def _load_pickled_model(self, mode_dir, model_file, model_pt_pth):```模型加载
4. ```def preprocess(self, data):```前处理过程
5. ```inference(self, data, *args, **kwargs):```推理过程
6. ```def postprocess(self, data):```后处理过程

同时，TorchServe也提供了其他类，帮助完成了相关的操作，以下是他们的继承关系及其应用说明

Basehandler:基础处理过程

1. VisionHandler:图像基础处理过程
   1. ObjectDetector:物品追踪
   2. ImageSegmenter:图像分割
   3. ImageClassifier:图像分类
2. TextHandler:文本基础处理过程
3. DenseNetHandler:密集网络处理过程

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

