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

# 1. 目录说明

- model_zoo/ : 模型存储文件夹，

- my_configs/: 自定义配置文件
- data/ : 训练数据集，软链接至此

- adv/： package 包含推理，攻击，生成补丁

# 2.推理

推理API：adv.inference

```python
def inference(model_name: str, img_path: str, is_save: bool = False, save_path: str = None):
    """
    模型推理
    :param model_name: 模型名称
    :param img_path: 图片路径
    :param is_save: 是否保存结果
    :param save_path: 保存路径
    :return: pred_instances 预测结果
    """
```

model_name：模型名称，为字符串有以下选项:

- faster_rcnn: 在coco数据集上训练的faster rcnn模型
- faster_rcnn_voc:  在pascal_voc数据集上训练的faster rcnn模型
- ssd: 在coco数据集上训练的ssd模型
- ssd_voc： 在pascal_voc数据集上训练的ssd模型
- retinanet：在coco数据集上训练的retinanet模型
- retinanet_voc：在pascal_voc数据集上训练的retinanet模型
- centernet：在coco数据集上训练的centernet模型
- centernet_update：在coco数据集上训练的centernet_update模型
- yolov3：在coco数据集上训练的yolov3模型

img_path：图片路径，为字符串

is_save：是否保存推理结果可视化图片

save_path：保存路径

返回结果pred_instances：

- pred_instances.bboxes：为bboxes的tensor
- pred_instances.scores：为scores的tensor
- pred_instances.labels: 为labels的tensor

使用样例:

```python
from adv import inference


if __name__ == '__main__':
    model_name = "ssd"
    img_path = "demo/demo.jpg"
    pred_instances = inference(model_name, img_path, is_save=True, save_path="outputs")
    print("bbox: ", pred_instances.bboxes)
    print("scores: ", pred_instances.scores)
    print("labels: ", pred_instances.labels)
```


# 3.train

设置训练gpu

```shell
export CUDA_VISIBLE_DEVICES=0
```

训练方法: 
    
```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

如:
    
```shell
python tools/train.py my_configs/yolov3_d53_8xb8-ms-608-273e_tt100k.py
```

训练结果位于 `work_dirs/`下

修改训练的epoch数目或val频率, 在配置文件中修改

```python
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=64, val_interval=1)
```