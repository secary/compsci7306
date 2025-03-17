# 1. 使用 Jupyter 官方基础镜像
FROM jupyter/base-notebook:latest

# 2. 设置工作目录
WORKDIR /home/jovyan/work

# 3. 复制项目文件（可选）
COPY . .

# 4. 创建虚拟环境
RUN python -m venv /home/jovyan/venv

# 5. 激活虚拟环境并安装依赖
RUN /bin/bash -c "source /home/jovyan/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt"

# 6. 配置 Jupyter 让它识别虚拟环境
RUN /bin/bash -c "source /home/jovyan/venv/bin/activate && \
    python -m ipykernel install --user --name=venv --display-name 'Python (venv)'"

# 7. 暴露端口 8888 用于 Jupyter Notebook
EXPOSE 8888

# 8. 启动 Jupyter Notebook，设置为监听所有接口并不自动打开浏览器
CMD ["start-notebook.sh", "--NotebookApp.ip='0.0.0.0'", "--NotebookApp.port=8888", "--NotebookApp.open_browser=False"]
