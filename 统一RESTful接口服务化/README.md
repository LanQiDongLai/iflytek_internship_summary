# 统一RESTful接口的中转服务

> 此服务中转程序是为了将客户端传来的请求经过转换传输到各个服务器中
>
> 具体作用于对TorchServe和TF-Serving的RESTful接口的合并

## 部署环境

1. Linux

2. [C++ Crow库](www.github,com/ipkn/crow) 一个用于创建HTTP或Websocket web服务的C++框架，此处用于接受客户端请求和响应客户端

3. [C cURL库](www.github.com/curl/curl) 一个用于发送请求报文和接受响应的网络微框架，此处用于将消息中转到目标服务器，并接受服务端的请求

4. [C lua库](www.github.com/lua/lua) 一个用于运行lua脚本的库

### 如何发送请求

对于TorchServe的用户请求，使用curl <ip>:<port>/pytorch/inference/<path>

对于TorchServe的管理请求，使用curl <ip>:<port>/pytorch/management/<path>

对于TF-Serving，使用curl <ip>:<port>/tensorflow/<path>

> 此处的<ip>:<port>为中转服务器IP和port，<path>对应于原本直接发送给服务端的请求url

原本的请求URL格式请详见

1. [TorchServe RESTful API](../TorchServe服务的使用/PyTorch服务化工具.md##TorchServe gRPC API)
2. [TF-SpeechServe RESTful API](../TensorFlow训练语音及其服务化/TF-SpeechServe.md#RESTful API)

例如：

原来要对PyTorch的服务请求

```bash
curl -X POST "http://localhost:8081/models?initial_workers=1&&url=/home/model-server/model-store/mnist.mar"
```

现在经过中转服务器统一之后，客户端向中转服务的请求变为

```bash
curl -X POST "http://localhost:18080/pytorch/management/models?initial_workers=1&&url=/home/model-server/model-store/mnist.mar"
```

