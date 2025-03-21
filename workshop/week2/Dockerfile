# 使用 Jupyter 官方镜像
FROM jupyter/base-notebook:latest

# 切换到 root 用户
USER root

# 安装系统级依赖（如果需要）
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    liblzma-dev \
    && apt-get clean

# 复制 requirements.txt 并安装依赖
COPY . . /home/jovyan/work/
RUN pip install --upgrade pip && pip install -r /home/jovyan/work/requirements.txt

# 设置工作目录
WORKDIR /home/jovyan/work

# 启动 Jupyter Notebook
CMD ["start-notebook.sh", "--NotebookApp.ip='0.0.0.0'", "--NotebookApp.port=8888", "--NotebookApp.open_browser=False"]
