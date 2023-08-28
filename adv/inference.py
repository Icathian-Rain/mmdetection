import cv2
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS

# visulizer配置
vis_backends = [dict(type='LocalVisBackend')]
visualizer_cfg = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer', save_dir="outputs")

 # 初始化可视化工具
visualizer = VISUALIZERS.build(visualizer_cfg)

# 模型配置文件和模型路径
#----------------------------------------------------------
configs = {
    "faster_rcnn": {
        "config": "configs/faster_rcnn/faster-rcnn_r101_fpn_2x_coco.py",
        "checkpoint": "model_zoo/faster_rcnn/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth"
    },
    "faster_rcnn_voc": {
        "config": "configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712.py",
        "checkpoint": "model_zoo/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712_20220320_192712-54bef0f3.pth"
    },
    "ssd": {
        "config": "configs/ssd/ssd512_coco.py",
        "checkpoint": "model_zoo/ssd/ssd512_coco_20210803_022849-0a47a1ca.pth"
    },
    "ssd_voc": {
        "config": "configs/pascal_voc/ssd512_voc0712.py",
        "checkpoint": "model_zoo/ssd/ssd512_voc0712_20220320_194717-03cefefe.pth"
    },
    "retinanet": {
        "config": "configs/retinanet/retinanet_r101_fpn_2x_coco.py",
        "checkpoint": "model_zoo/retinanet/retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth"
    },
    "retinanet_voc": {
        "config": "configs/pascal_voc/retinanet_r50_fpn_1x_voc0712.py",
        "checkpoint": "model_zoo/retinanet/retinanet_r50_fpn_1x_voc0712_20200617-47cbdd0e.pth"
    },
    "centernet": {
        "config": "configs/centernet/centernet_r18-dcnv2_8xb16-crop512-140e_coco.py",
        "checkpoint": "model_zoo/centernet/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth"
    },
    "centernet_update": {
        "config": "configs/centernet/centernet-update_r50-caffe_fpn_ms-1x_coco.py",
        "checkpoint": "model_zoo/centernet/centernet-update_r50-caffe_fpn_ms-1x_coco_20230512_203845-8306baf2.pth"
    },
    "yolov3": {
        "config": "configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py",
        "checkpoint": "model_zoo/yolov3/yolov3_d53_fp16_mstrain-608_273e_coco_20210517_213542-4bc34944.pth"
    }
}



def inference(model_name: str, img_path: str, is_save: bool = False, save_path: str = None):
    """
    模型推理
    :param model_name: 模型名称
    :param img_path: 图片路径
    :param is_save: 是否保存结果
    :param save_path: 保存路径
    :return: pred_instances 预测结果
    """
    config_file = configs[model_name]["config"]
    checkpoint_file = configs[model_name]["checkpoint"]
    # 根据配置文件和 checkpoint 文件构建模型
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    # 推理图片
    result = inference_detector(model, img_path)
    # 可视化
    if is_save:
        if save_path is None:
            save_path = "outputs"
        # 读取图片
        img = mmcv.imread(img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        # 加载模型配置
        visualizer.dataset_meta = model.dataset_meta
        # 保存结果
        visualizer.add_datasample(
            save_path,
            img,
            data_sample=result,
            draw_gt=False,
            show=False)
    return result.pred_instances



if __name__ == '__main__':
    model_name = "faster_rcnn"
    img_path = "demo/demo.jpg"
    pred_instances = inference(model_name, img_path)
    print("bbox: ", pred_instances.bboxes)
    print("scores: ", pred_instances.scores)
    print("labels: ", pred_instances.labels)