# 0 安装

## 0.1 mmdetection安装

安装miniconda 或 anaconda

创建环境

```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

安装pytorch（有问题，安装的是cpu版）

```shell
conda install pytorch torchvision -c pytorch
```

使用pytorch官方方法安装cu118版本

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

安装MMEngine和MMCV

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

安装mmdetection，在本目录下

```shell
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

测试是否安装完成(可不要)

下载测试配置文件

```shell
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
```

测试

```shell
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
```

