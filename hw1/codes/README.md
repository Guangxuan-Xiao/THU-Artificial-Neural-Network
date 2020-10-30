# 运行方法

将`data/`文件夹放置在`codes/`所在目录，进入`codes`目录运行`python run_mlp.py`即可开始跑网络。通过调整`run_mlp.py`和`mlp1.py`，`mlp2.py`中的结构可以调整网络结构。

# 新文件

`mlp1.py`：用于构造单隐层MLP。

`mlp2.py`：用于构造双隐层MLP。

# 除了填入代码外的修改

`run_mlp.py`: 添加了`plot`函数用于画图，添加了最后的测试、画图和结果输出，图会输出到`../plots/`下，结果会输出到`../results/`下。

`solve_net.py`：在`train_net`和`test_net`函数加入了返回平均`loss`和`acc`的逻辑。

`layers.py`, `network.py `：加入了`__str__`方法，用于输出模型的名称方便统计。

