#coding=utf-8

import torch
import torchvision
from torchvision import transforms
from network_files.faster_rcnn_framework import FasterRCNN, FastRCNNPredictor
from backbone.resnet50_fpn_model import resnet50_fpn_backbone
from network_files.rpn_function import AnchorsGenerator
from backbone.mobilenetv2_model import MobileNetV2
from draw_box_utils import draw_box
from PIL import Image
import json
import matplotlib.pyplot as plt
from datasets.BottleCapDataset.dataset import BottleCapDataSet
import numpy as np
import time


def create_model(num_classes):
    backbone = resnet50_fpn_backbone()
    model = FasterRCNN(backbone=backbone, num_classes=num_classes)

    return model


if __name__ == "__main__":
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # create model
    model = create_model(num_classes=8)

    # load train weights
    train_weights = "./save_weights/resNetFpn-model-30.pth"
    model.load_state_dict(torch.load(train_weights)["model"])
    model.to(device)

    # read class_indict
    # category_index = {}
    # try:
    #     json_file = open('./pascal_voc_classes.json', 'r')
    #     class_dict = json.load(json_file)
    #     category_index = {v: k for k, v in class_dict.items()}
    # except Exception as e:
    #     print(e)
    #     exit(-1)
    # category_index = BottleCapDataSet().get_id_dict()
    category_index = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10}

    # load image
    train_data_set = BottleCapDataSet("/media/host/新加卷1/dy/天池缺陷检测比赛/dataset/", None, True)
    val_data_set = BottleCapDataSet("/media/host/新加卷1/dy/天池缺陷检测比赛/dataset/", None, False)
    dataset = val_data_set

    model.eval()
    with torch.no_grad():
        # for train, val
        for original_img, target in dataset:

            data_transform = transforms.Compose([transforms.ToTensor()])
            img = data_transform(original_img)
            img = torch.unsqueeze(img, dim=0)
            time_start = time.time()
            predictions = model(img.to(device))[0]
            time_end = time.time()
            print("{}s/it.".format(time_end-time_start))
            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            # label2id
            predict_classes = [dataset.label2id(cls) for cls in predict_classes]
            predict_scores = predictions["scores"].to("cpu").numpy()

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")

            original_img_predict = original_img.copy()
            draw_box(original_img_predict,
                     predict_boxes,
                     predict_classes,
                     predict_scores,
                     category_index,
                     thresh=0.5,
                     line_thickness=5)

            original_img_label = original_img.copy()
            draw_box(original_img_label,
                     target["boxes"].to("cpu").numpy(),
                     # predict_classes,
                     target["labels"].to("cpu").numpy(),
                     # ["GT:{}".format(label) for label in target["labels"]],
                     np.ones(len(target["labels"])),
                     category_index,
                     thresh=0.5,
                     line_thickness=5
                     )

            plt.figure(figsize=(20, 20))
            plt.subplot(1, 2, 1)
            plt.imshow(original_img_predict)
            plt.subplot(1, 2, 2)
            plt.imshow(original_img_label)
            plt.show()



