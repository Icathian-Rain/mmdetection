from adv import inference


if __name__ == '__main__':
    model_name = "ssd"
    img_path = "demo/demo.jpg"
    pred_instances = inference(model_name, img_path, is_save=True, save_path="outputs")
    print("bbox: ", pred_instances.bboxes)
    print("scores: ", pred_instances.scores)
    print("labels: ", pred_instances.labels)