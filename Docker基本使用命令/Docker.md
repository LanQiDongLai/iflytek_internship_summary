# Docker

### 基本命令

#### 镜像相关

##### 查看所有镜像

```bash
docker images
```

##### 删除指定镜像

```bash
docker rmi {image-id}
```

##### 查找Docker Hub上的镜像

```bash
docker search {image-name}
```

##### 拉取镜像

```bash
docker pull {image-name}
```



#### 容器相关

##### 查看所有容器

1. 默认：仅查看运行中的容器
2. -a：查看所有容器
3. -q：只显示容器的ID号

```bash
docker ps
```

##### 删除指定容器

1. -f 强制删除

```bash
docker rm {container-id}
```

##### 运行容器

1. -i：交互式，保持容器的标准输入打开，使得用户可以交互
2. -t：终端，为执行命令分配一个伪终端
3. -u {user}：用户，以何种用户进入容器
4. -p {ip:host-port:container-port}：端口映射，对内：主机如何通过网络连接到容器，对外：主机牺牲一个端口号将该端口号上接受或发送的数据映射到容器内
5. --name {container-name}：容器名称

```bash
docker run -it -p {ip:host-port:container-port} -u {user} --name {container-name} {image-id} /bin/bash
```

##### 停止运行中的容器

```bash
docker stop {container-id}
```

##### 强制关停容器

```bash
docker kill {container-id}
```

##### 启动关闭的容器

```bash
docker start {container-id}
```

##### 进入运行中的容器

1. -i：交互式，保持容器的标准输入打开，使得用户可以交互
2. -t：终端，为执行命令分配一个伪终端
3. -u {user}：用户，以何种用户进入容器

```bash
docker exec -it {container-id} /bin/bash
```

