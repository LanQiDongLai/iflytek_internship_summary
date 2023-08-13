# Model文件加载过程

#### Model文件的种类

Model文件按照**两个属性**进行分类，分别是["是否为压缩文件格式"，"是否只包含权重信息"]，其中如果是压缩文件，需要判断是否是TorchScript所保存的模型文件，这个文件应当使用```torch.jit.load(file: str)```方法来加载，使用```torch.load(file: str)```会给出警告，但是不影响使用

下图为不同的文件类型所对应的不同分支

![image-20230724113449060](/home/LanQiDongLai/.config/Typora/typora-user-images/image-20230724113449060.png)

#### Model文件的加载过程

##### 介绍load, \_load和\_legacy\_load方法

load方法使用用户加载模型的标准方法，它包含以下几个参数

```python
def load(
    f: FILE_LIKE,
    map_location: MAP_LOCATION = None,
    pickle_module: Any = None,
    *,
    weights_only: bool = False,
    **pickle_load_args: Any
) -> Any
# 第一个参数，文件路径
# 第二个参数，指定如何重新映射存储位置的函数，参数类型可以为字符串，字典，torch.device类型
# 第三个参数，自定义的文件解析器，此解析器用于读取模型中的元数据和对象模块，如果此参数为None，那么解析器会使用PyTorch内部解析器（_weights_only_unpickler或者pickle）
# weights_only参数，指明是否解析器限制为仅加载张量、基元类型和字典，如果设置为True，那么第三个参数必须为None，解析器会被设置为_weights_only_unpickler解析器，如果模型文件包含元数据则需设置为False，此参数的目的是防止模型文件中包含了不安全的操作，在确定来源的基础上，可以忽略该参数
# pickle_load_args参数，额外配置，如模型文件的字符编码
```

使用实例：

```python
# 将模型Tensor映射到cuda1,cuda0
torch.load('model.pt', map_location={'cuda:1': 'cuda:0'})
# 将所用Tensor加载到CPU，并指明文件编码为ascii
torch.load('model.pt', map_location=torch.device('cpu'), , encoding='ascii')
```



_load方法用于解析加载zip压缩文件格式的文件，此方法为内部方法，被load方法所调用

```python
def _load(zip_file, map_location, pickle_module, pickle_file='data.pkl', **pickle_load_args)
# 第一个参数，一个PyTorchFileReader，用来读取zip文件
# 第二个参数，指定如何重新映射存储位置的函数，参数类型可以为字符串，字典，torch.device类型
# 第三个参数，自定义的文件解析器，此解析器用于读取模型中的元数据和对象模块
# 第四个参数，一个解析器文件
# 第五个参数，额外配置，如模型文件的字符编码
```

\_legacy\_load方法用于解析加载非压缩格式的模型文件，此方法为内部方法，被load方法所调用

```python
def _legacy_load(f, map_location, pickle_module, **pickle_load_args)
# 第一个参数，一个已打开的文件
# 第二个参数，指定如何重新映射存储位置的函数，参数类型可以为字符串，字典，torch.device类型
# 第三个参数，自定义的文件解析器，此解析器用于读取模型中的元数据和对象模块
# 第四个参数，额外配置，如模型文件的字符编码
```



##### pickle解析器

###### 选择过程

pickle_module解析器如果在load函数中没有被设置，或者值为None的话，那么会根据weight_only的值来决定默认的解析器

下图表示pickle_module解析器的选择过程

![image-20230724144002452](/home/LanQiDongLai/.config/Typora/typora-user-images/image-20230724144002452.png)

![image-20230724144038423](/home/LanQiDongLai/.config/Typora/typora-user-images/image-20230724144038423.png)

###### 调用过程

下面着重讲解pickle解析器解析非压缩文件，并且包含元数据的模型文件（mnist_cnn.pt 手写数字识别的模型文件）

第一次调用用于确定文件的魔数，来判断文件是否是非压缩格式的模型文件

第二次调用用于获取协议版本

第三次调用用于获取该文件解析所需要确定的基本参数，如大小端，short类型的长度，int类型长度，long类型长度

第四次调用用于获取模型的基本结构，但是所有模型参数的值都设置为0，但是读取好的模型参数以及存储在deserialized_objects变量之中

第五次调用用于设置模型的权重参数

![image-20230724153343111](/home/LanQiDongLai/.config/Typora/typora-user-images/image-20230724153343111.png)

###### 具体步骤

下图为mnist_cnn.pt文件的HEX编码

![image-20230724153220130](/home/LanQiDongLai/.config/Typora/typora-user-images/image-20230724153220130.png)

模型文件中的数据分为两种，一种称作操作码，一种称作存储值，pickle的工作方式类似于图灵机，通过读取操作码，并根据操作码来执行特定的操作，将数据进行压栈，出栈，备忘，或对解析器内部状态进行改变，或终止解析

操作码对应具体的操作可以查看pickle.py文件中的定义

下面给出第一次和第二次调用中文件的HEX码所表示的过程

> 80 02 8A 0A 6C FC 9C 46 F9 20 6A A8 50 19 2E
>
> 协议：[80]读取协议 协议号02
>
> 魔数确认：[8A]读取无符号长整型 0A长度为10 后十字节确认号MAGIC_NUMBER 0x1950a86a20f9469cfc6c 
>
> 终止：[2E]结束读取

> 80 02 4D E9 03 2E
>
> 协议：80读取协议 协议号02
>
> 协议版本：4D读取无符号短整型 E903对应协议号整数1001并压栈
>
> 终止：2E结束读取

##### persistent_load方法

对于不同类型的文件，pickle会采用不同的persistent_load方法，该方法用于读取模型参数

该方法会被[51]PERSID和[51]BINPERSID操作码调用


