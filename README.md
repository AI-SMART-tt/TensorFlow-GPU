## TensorFlow GPU 环境配置步骤

### 1. 环境准备
打开 Anaconda Prompt，执行以下命令：

```bash
# 创建新的conda环境
conda create -n tf_gpu python=3.8 -y 

# 激活环境
conda activate tf_gpu
```

### 2. 安装CUDA和cuDNN
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1 -y
```

### 3. 安装TensorFlow
```bash
# 安装tensorflow
pip install tensorflow==2.10.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装tensorflow-gpu
pip install tensorflow-gpu==2.10.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 4. 配置Jupyter Notebook
```bash
# 安装必要的包
conda install jupyter notebook ipykernel

# 注册内核
python -m ipykernel install --user --name=tf_gpu --display-name="Python (tf_gpu)"
```

### 5. 验证安装
在 Jupyter Notebook 或 Python 中运行以下代码验证 GPU 配置：

```python
import tensorflow as tf

# 检查 GPU 设备列表
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("\n===== GPU 可用 =====")
    for gpu in gpus:
        print("设备名称:", gpu.name)
        print("设备类型:", gpu.device_type)
    print("====================\n")
else:
    print("\n!!! 未检测到 GPU，TensorFlow 将使用 CPU !!!\n")

# 检查 TensorFlow 是否可用 GPU
print("TensorFlow 版本:", tf.__version__)
print("GPU 设备名称:", tf.test.gpu_device_name() or "无")
print("是否支持 GPU:", "是" if tf.test.is_built_with_cuda() else "否")
print("GPU 是否可用:", "是" if tf.config.list_physical_devices('GPU') else "否")

# 运行一个简单计算验证 GPU 加速
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print("\n矩阵乘法结果（应位于 GPU 上）:\n", c.numpy())
```

### 版本信息汇总
- **Python**: 3.8
- **CUDA Toolkit**: 11.2
- **cuDNN**: 8.1
- **TensorFlow**: 2.10.0
- **TensorFlow-GPU**: 2.10.0

### 参考资料
https://zhuanlan.zhihu.com/p/1897427950598608126
