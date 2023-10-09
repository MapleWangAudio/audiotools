# 安装

先自行根据自己设备情况安装相对应的pytorch（本项目基于python3.10, pytorch2.0进行的构建, 其他版本请自行检查兼容性）

然后

```
pip install -r requirements.txt
```

或者

```
conda install -c pytorch -c defaults -c conda-forge --file requirements.txt
```

requirements_extra.txt中有一些常用的额外的库, 可以选择性安装。当然, 不安装extra也能正常使用本库

```
conda install --file requirements_extra.txt
```

# 使用方法

放在工程目录下, 与main.py同级, 然后

```
import audiotools.analysis as analysis
import audiotools.process as process
```

不同级的话, 需要在上列代码之前额外添加如下代码

```
import sys
sys.path.append("/your/audiotools/path")
# 如audiools在/home/user/audiotools
# 则为/home/user即可
# 当然, 也可以使用相对路径
```

**注意**：

本库中所有的音频处理方法都是一个点一个点的进行的, 如果对于超大数组的一次性处理 (如48/24规格超过10s的音频), 可以考虑使用torchaudio中的方法, 如: amp和dB的相互转换, 滤波器等.

分析类的方法输入输出格式都是 [通道, 数据]

# 版本号定义

稳定性.新py文件.新方法的加入.日常维护和bug修正

稳定性解释：

    v：正式版本, 不出意外的话, 稳定可用

    t：过程版本, 有大量未解决的问题, 不可正常使用

# 各个py文件说明

process：各种处理方法. 

analysis：分析绘图&特征提取.

# 常见问题

## 关于matplotlib不显示

wsl可能没有gui, 如果想要gui显示, 需要下载gui支持, 可以尝试：

```
sudo apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
```

这里使用savefig解决


## 使用line_profiler进行性能分析

这个可以对函数进行每行的性能分析, 如果需要函数整个运行的性能分析, 可以使用visual studio或者pycharm的性能分析器

```
# 把下行代码加入到需要测试的函数上
@profile
def profile():
    pass
# 进入终端，进行测试 
kernprof -l profile.py
# 读取测试结果
python -m line_profiler profile.py.lprof
```

## 使用torch时将数据放在gpu上运行的方法：

``````
# 检查是否有可用的 GPU 设备
if torch.cuda.is_available():
    # 设置默认的 GPU 设备为 GPU 0
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
# 将 tensor 移动到设备 device 上
test = test.to(device)
``````

## 将tensor以float64计算的方法:

对音频处理, float32不太够用, 尽量使用float64. 大部分时候我们使用numpy进行的计算, 默认是64位, 无需修改. 如需使用pytorch的64位, 则在主函数的开头加入:

```
torch.set_default_dtype(torch.float64)
```

这将使得所有的tensor都以64位进行计算, 但是会降低计算速度, 请谨慎使用.