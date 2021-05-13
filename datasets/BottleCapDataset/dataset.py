import os
from torch.utils.data import Dataset
import json
import torch
from PIL import Image
import time



def annotations_trans(annoatations):
    image_list = annoatations["images"]
    for image in image_list:
        image["bbox"] = list()
        image["category_id"] = list()
        image["iscrowd"] = list()
        image["area"] = list()

    for annoatation in annoatations["annotations"]:
        for image in image_list:
            if image["id"] == annoatation["image_id"]:
                image["bbox"].append(annoatation["bbox"])
                image["category_id"].append(annoatation["category_id"])
                image["iscrowd"].append(annoatation["iscrowd"])
                image["area"].append(annoatation["area"])
    # 瓶帽筛选
    def body_filter(image):
        return image["id"]<4106
    image_list = filter(body_filter, image_list)
    # [print(image) for image in image_list]
    # a = [0]*15
    # for image in image_list:
    #     for id in image["category_id"]:
    #         a[id] += 1
    # print(a)
    # exit()
    return list(image_list)


"""
{
    "info"
    "license"
    "images":
        [
            {"file_name":"cat.jpg", "id":1, "height":1000, "width":1000},
            {"file_name":"dog.jpg", "id":2, "height":1000, "width":1000},
            ...
        ]
    "annotations":
        [
            {"image_id":1, "bbox":[100.00, 200.00, 10.00, 10.00], "category_id": 1}     # xmin,ymin,w,h
            {"image_id":2, "bbox":[150.00, 250.00, 20.00, 20.00], "category_id": 2}
            ...
        ]
    "categories":
        [
            {"id":0, "name":"bg"}
            {"id":1, "name":"cat"}
            {"id":2, "name":"dog"}
            ...
        ]
}
-->>
"images":
    [
        {"file_name":"cat.jpg", "height":1000, "width":1000, "bbox":[[xmin1, ymin1, w1, h1], [xmin2, ymin2, w2, h2]], "category_id":[0, 1]}
    ]
"""

class BottleCapDataSet(Dataset):
    """
    天池酒瓶缺陷检测(瓶帽)
    0: '背景',    1160
    1: '瓶盖破损', 1619
    2: '瓶盖变形', 705
    3: '瓶盖坏边', 656
    4: '瓶盖打旋', 480
    5: '瓶盖断点', 614
    9: '喷码正常', 489
    10: '喷码异常' 199

    """

    def __init__(self, root=None, transforms=None, train_set=True):

        # self.root = root
        self.root = root
        self.img_root = os.path.join(self.root, "images")
        self.annotations = json.load(open(os.path.join(self.root, "annotations.json")))
        self.image_list = annotations_trans(self.annotations)

        if train_set:
            self.image_list = self.image_list[:int(len(self.image_list) * 0.8)]
            # self.image_list = self.image_list[:100]
        else:
            self.image_list = self.image_list[int(len(self.image_list) * 0.8):]
            # self.image_list = self.image_list[-50:]

        self.transforms = transforms

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_root, self.image_list[idx]["file_name"]))
        boxes = [(box[0], box[1], box[0] + box[2], box[1] + box[3]) for box in self.image_list[idx]["bbox"]]
        labels = [self.id2label(id) for id in self.image_list[idx]["category_id"]]
        iscrowd = self.image_list[idx]["iscrowd"]
        area = self.image_list[idx]["area"]

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = torch.as_tensor(area, dtype=torch.float32)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def coco_index(self, idx):
        image = Image.open(os.path.join(self.img_root, self.image_list[idx]["file_name"]))
        boxes = [(box[0], box[1], box[0] + box[2], box[1] + box[3]) for box in self.image_list[idx]["bbox"]]
        labels = [self.id2label(id) for id in self.image_list[idx]["category_id"]]
        iscrowd = self.image_list[idx]["iscrowd"]
        area = self.image_list[idx]["area"]

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = torch.as_tensor(area, dtype=torch.float32)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return (self.image_list[idx]["height"], self.image_list[idx]["width"]), target

    def get_height_and_width(self, idx):
        return self.image_list[idx]["height"], self.image_list[idx]["width"]

    @staticmethod
    def id2label(id):
        return id if id <= 5 else id - 3

    @staticmethod
    def label2id(label):
        return label if label <= 5 else label + 3

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

    @staticmethod
    def get_num_classes():
        return 8

    @staticmethod
    def get_id_dict():
        return {
            0: '背景',
            1: '瓶盖破损',
            2: '瓶盖变形',
            3: '瓶盖坏边',
            4: '瓶盖打旋',
            5: '瓶盖断点',
            6: '标贴歪斜',
            7: '标贴起皱',
            8: '标贴气泡',
            9: '喷码正常',
            10: '喷码异常'
        }




if __name__ == "__main__":
    dataset = BottleCapDataSet()
