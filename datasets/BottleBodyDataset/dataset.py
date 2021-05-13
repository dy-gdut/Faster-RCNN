import os
from torch.utils.data import Dataset
import json
import torch
from PIL import Image


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
    # 瓶身筛选
    def body_filter(image):
        return image["id"]>=4106
    image_list = filter(body_filter, image_list)
    # [print(image) for image in image_list]
    # a = [0]*15
    # for image in image_list:
    #     for id in image["category_id"]:
    #         a[id] += 1
    # print(a)

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

class BottleBodyDataSet(Dataset):
    """
    天池酒瓶缺陷检测(瓶身)
    0: '背景':    10
    6: '标贴歪斜': 186
    7: '标贴起皱': 384
    8: '标贴气泡': 443

    """

    def __init__(self, root=None, transforms=None, train_set=True):

        # self.root = root
        self.root = "/media/host/新加卷3/dy/天池缺陷检测比赛/dataset/"
        self.img_root = os.path.join(self.root, "images")
        self.annotations = json.load(open(os.path.join(self.root, "annotations.json")))
        self.image_list = annotations_trans(self.annotations)
        self.label_dict = self.get_label_dict()
        # trans label
        for image in self.image_list:
            image["category_id"] = [self.label_dict[id] for id in image["category_id"]]
            # print(image["category_id"])


        if train_set:
            self.image_list = self.image_list[:int(len(self.image_list) * 0.9)]
        else:
            self.image_list = self.image_list[int(len(self.image_list) * 0.9):]

        self.transforms = transforms

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_root, self.image_list[idx]["file_name"]))
        boxes = [(box[0], box[1], box[0] + box[2], box[1] + box[3]) for box in self.image_list[idx]["bbox"]]
        labels = self.image_list[idx]["category_id"]
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


    def get_height_and_width(self, idx):
        return self.image_list[idx]["height"], self.image_list[idx]["width"]

    def coco_index(self, idx):
        image = Image.open(os.path.join(self.img_root, self.image_list[idx]["file_name"]))
        boxes = [(box[0], box[1], box[0] + box[2], box[1] + box[3]) for box in self.image_list[idx]["bbox"]]
        labels = self.image_list[idx]["category_id"]
        iscrowd = self.image_list[idx]["iscrowd"]
        area = self.image_list[idx]["area"]

        data_height = int(self.image_list[idx]["height"])
        data_width = int(self.image_list[idx]["width"])

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

        return (data_height, data_width), target
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

    @staticmethod
    def get_num_classes():
        return len(BottleBodyDataSet.get_label_dict().keys())

    @staticmethod
    def get_label_dict():
        return {
            0: 0,
            6: 1,
            7: 2,
            8: 3
        }


if __name__ == "__main__":
    dataset = BottleBodyDataSet()
    # print(BottleBodyDataSet.get_num_classes())
