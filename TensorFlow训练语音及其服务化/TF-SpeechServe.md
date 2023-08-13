# TF-SpeechServe

## TensorFlow语音训练过程

准备工作

导入必要的模块和依赖项

> 注：TensorFlow运行需要avx扩展指令集，对于一些老牌的CPU可能不支持该指令集（执行时会报「段错误：非法指令」）

```python
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
```



#### 准备数据集

> 在这里我使用了speech_commands数据集，其中包含了down, yes, no, go, right, left, stop语音，每个种类包含1000个数据

```python
DATASET_PATH = 'data/mini_speech_commands'

data_dir = pathlib.Path(DATASET_PATH)
if not data_dir.exists():
    # 下载speech_commands数据集
    tf.keras.utils.get_file(
        'mini_speech_commands.zip',
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        extract=True,
        cache_dir='.', cache_subdir='data')
```

数据集的音频片段存储在与每个语音命令相对应的八个文件夹中

```python
# 获取所有文件夹名称 array(['right', 'yes', 'README.md', 'up', 'left', 'no', 'go', 'stop','down'], dtype='<U9')
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[(commands != 'README.md') & (commands != '.DS_Store')]
print('Commands:', commands)
```

将数据集分为训练集和测试集，同时设置output_sequence_length=16000表示将一个音频的长度设置为1s（不足填充，超过截断），设置类别所对应的标签

```python
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64,
    validation_split=0.2,
    seed=0,
    output_sequence_length=16000,
    subset='both')

label_names = np.array(train_ds.class_names)
```

数据集只包含单声道，使用tf.squeeze删除多余的轴

```python
# 删除所有只有一单位大小的维度
def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels
# 处理前(64, 16000, 1) 处理后(64, 16000)
train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)
# 将测试集分成两个分片
test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)
```

~~绘制一些音频的波形图谱~~

```python	
rows = 3
cols = 3
n = rows * cols
fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

for i in range(n):
    if i>=n:
        break
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    ax.plot(example_audio[i].numpy())
    ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
    label = label_names[example_labels[i]]
    ax.set_title(label)
    ax.set_ylim([-1.1, 1.1])
# 显示plt
plt.show()
```

![image-20230729161906733](/home/LanQiDongLai/.config/Typora/typora-user-images/image-20230729161906733.png)

#### 将波形图转化为频谱图

接下来通过计算短时傅里叶变化（STFT）来将时域信号转换成时频信号，以将波形转换为频谱图

```python
def get_spectrogram(waveform):
    # 通过STFT将波形转换为频谱图
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    # 获取STFT的大小
    spectrogram = tf.abs(spectrogram)
    # 添加“通道”维度，使得谱图可以用作具有卷积层的类似图像的输入数据（其期望形状（“batch_size”、“height”、“width”、“channels”）
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram
```

~~查看一些数据的属性，其中包含数据的波形数据和时频数据~~

```python
for i in range(3):
    label = label_names[example_labels[i]]
    waveform = example_audio[i]
    spectrogram = get_spectrogram(waveform)
    print('Label:', label)
    print('Waveform shape:', waveform.shape)
    print('Spectrogram shape:', spectrogram.shape)
    print('Audio playback')
    display.display(display.Audio(waveform, rate=16000))
```

~~绘制音频的波形图和时频图~~

```python
def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)
fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')
axes[0].set_xlim([0, 16000])

plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.suptitle(label.title())
plt.show()

```

![image-20230729163326627](/home/LanQiDongLai/.config/Typora/typora-user-images/image-20230729163326627.png)

将原来的时域数据集转变为时频数据集

```python
def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)
train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)
```

~~查看数据集不同示例的频谱图~~

```python
for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
  break
rows = 3
cols = 3
n = rows*cols
fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

for i in range(n):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    plot_spectrogram(example_spectrograms[i].numpy(), ax)
    ax.set_title(label_names[example_spect_labels[i].numpy()])

plt.show()
```

![image-20230729180824918](/home/LanQiDongLai/.config/Typora/typora-user-images/image-20230729180824918.png)

#### 创建并训练模型

在训练模型时，添加Dataset.cache和Dataset.prefetch操作以减少读取延迟

```python
train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
```

模型网络结构构建

```python
input_shape = example_spectrograms.shape[1:]
print('Input shape:', input_shape)
num_labels = len(label_names)

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

model = models.Sequential([
    layers.Input(shape=input_shape),
    # Downsample the input.
    layers.Resizing(32, 32),
    # Normalize.
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])
# 查看网络模型结构
model.summary()
```

使用Adam优化器和交叉熵损失配置Keras模型

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)
```

设置训练集和验证集，以及迭代次数(EPOCHS)

```python
EPOCHS = 10
history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)
```

~~通过绘制训练集和验证集的损失曲线，来展示模型参数在训练过程中的改进情况~~

```python
metrics = history.history
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Loss [CrossEntropy]')

plt.subplot(1,2,2)
plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
plt.legend(['accuracy', 'val_accuracy'])
plt.ylim([0, 100])
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')
```

![image-20230803122046671](/home/LanQiDongLai/.config/Typora/typora-user-images/image-20230803122046671.png)

~~使用混淆矩阵来查看每种数据的分类效果~~

```python
y_pred = model.predict(test_spectrogram_ds)
y_pred = tf.argmax(y_pred, axis=1)
y_true = tf.concat(list(test_spectrogram_ds.map(lambda s,lab: lab)), axis=0)
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
            xticklabels=label_names,
            yticklabels=label_names,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()
```

![image-20230803122025757](/home/LanQiDongLai/.config/Typora/typora-user-images/image-20230803122025757.png)

导出带有预训练的模型

```
tf.saved_model.save(model, "saved/1")
```

> 注：为何要进行傅里叶变换
>
> 1. 特征更明显：傅里叶变换可以将语音信号从时域转换到频域，在语音识别方面，我们更应该关心的是语音在时间上的频率变换而不是在时间上的波形震动
> 2. 训练更容易：转换成频域的语言数据更加的平滑，相比数据在时域上的高频震动，训练难度要下滑很多
> 3. 降噪和滤波：在转换的过程中可以应用各种滤波器来对数据进行增强处理，以改善语音的质量

## Docker部署TensorFlowServing

#### 准备镜像和容器

下载TensorFlowServing的镜像

```bash
docker pull tensorflow/serving
```

运行镜像

1. 启动格式 ```docker run -p 8501:8501 --mount type=bind,source=${SOURCE_DIR},target=${WORK_DIR} -e MODEL_NAME=￥{MODEL_NAME} -t tensorflow/serving```

2. -p 选项代表端口映射，8501为TensorFlowServing RESTful接口的端口号

3. -t tensorflow/serving，启动镜像tensorflow/serving

4. --model_config_file=/models/multiModel/models.config，多模型的配置文件

```bash
docker run -p 8501:8501 --mount type=bind,source=/models/multiModel/,target=/models/multiModel -t tensorflow/serving --model_config_file=/models/multiModel/models.config
```

下图为本地机器上的模型存储的目录及其包含的文件，对于要推理的pb模型，必须命名成saved_model.pb

<img src="/home/LanQiDongLai/.config/Typora/typora-user-images/image-20230803105023863.png" alt="image-20230803105023863" style="zoom:80%;" />

models.config文件的内容

```json
model_config_list:{
    config:{
      name:"MNIST",
      base_path:"/models/multiModel/MNIST",
      model_platform:"tensorflow"
    },
    config:{
      name:"SPEECHCOMMAND",
      base_path:"/models/multiModel/SPEECHCOMMAND",
      model_platform:"tensorflow"
    },
}
```

#### RESTful API

RESTful API是一种基于HTTP/1.1协议格式的API，通过请求和响应来进行服务之间的数据传输

##### 请求出错

对于TensorFlowServing，当请求出现错误时，所有API都会返回一个JSON对象，且格式如下

```json
{
    "error": <error message string>
}
```

下图为请求一个不存在模型时出现的错误

<img src="/home/LanQiDongLai/.config/Typora/typora-user-images/image-20230801154358113.png" alt="image-20230801154358113" style="zoom:80%;" />

##### 请求模型状态

1. host:port TensorFlowServing服务器所在域名或IP和TensorFlow开放RESTful API的端口号（默认为8501）
2. MODEL_NAME 模型的名称
3. VERSION 模型版本
4. LABEL 模型标签

```bash
curl http://host:port/v1/models/${MODEL_NAME}[/version/${VERSION}|/label/${LABEL}]
```

下图为请求模型状态的样例

<img src="/home/LanQiDongLai/.config/Typora/typora-user-images/image-20230801153736824.png" alt="image-20230801153736824" style="zoom:80%;" />

##### 推理预测

```bash
curl -X POST http://host:port/v1/models/${MODEL_NAME}[/version/${VERSION}|/labels/${LABEL}]:perdict
```

推理预测的请求载荷

```json
{
    "instance":<value>|<list-of-objects>
}
```

> 注：<list-of-objects>是什么
>
> list-of-object是一个数组，里面包含了转换成数组对象的数据，数据的格式对应着模型网络结构的输入层
>
> 格式例如：
>
> ```"instance": [ [1, 2], [2, 3] ]```
