import time

import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, set_logging
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect_custom(dataset, model, device, half, img_size=640, conf_thres=0.3,
                  iou_thres=0.45, augment=True, agnostic_nms=True, classes=None):
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = img_size
    old_img_b = 1

    t0 = time.time()

    print(test_images)
    predictions = {}
    for path, img, im0s, vid_cap in dataset:
        img_name = 'img' + path.split('/img')[1]

        if img_name not in test_images:
            continue
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                results = {}
                confidence = [(names[int(cls)], round(conf.item(), 2)) for *xyxy, conf, cls in det]
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    if names[int(c)] in results.keys():
                        results[names[int(c)]] = results[names[int(c)]] + 1
                    else:
                        results[names[int(c)]] = 1

                predictions[img_name] = {'results': results, 'confidence': confidence}
            else:
                predictions[img_name] = {'results': {}, 'confidence': {}}

            #if len(predictions.keys()) == 5:
            #    return predictions

        print('Done with ' + img_name)
    print(f'Done. ({time.time() - t0:.3f}s)')
    return predictions


def detect_customs(sources, weights='yolov7-e6e.pt', img_size=640):
    # Initialize
    set_logging()
    device = select_device()
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    img_size = check_img_size(img_size, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Set Dataloader
    dataset = LoadImages(sources, img_size=img_size, stride=stride)

    return detect_custom(dataset, model, device, half)


# ----------------------------------------------------------------
import csv
import numpy

test_images = []
with open(f'../images/test.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        test_images.append(row)
test_images = list(numpy.concatenate(test_images[1:]).flat)

# -----------------------------------------------------------------
import os
import json
import time

directory = f'../images/images/'

image_paths = []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        image_paths.append(f)

start = time.time()
results = detect_customs(directory)
end = time.time()
print('Time to run:')
print(end - start)
#print(json.dumps(results, indent=4))

# -----------------------------------------------------------------

coco_tartu_map = {
    'person': ['l0'],
    'bicycle': ['l11'],
    'car': ['l9', 'l50'],
    'motorcycle': ['l7'],
    'airplane': ['l63'],
    'bus': ['l52', 'l67'],
    'train': ['l65'],
    'truck': ['l67'],
    'boat': ['l61', 'l6', 'l10'],
    'traffic light': ['l16', 'l14', 'l7', 'l12'],
    'fire hydrant': ['l7'],
    'stop sign': ['l16', 'l14', 'l72'],
    'parking meter': ['l23', 'l7'],
    'bench': ['l30', 'l55'],
    'bird': ['l10'],
    'cat': ['l64'],
    'dog': ['l35'],
    'sheep': ['l10'],
    'cow': ['l10', 'l2'],
    'horse': ['l10', 'l2'],
    'bear': ['l10'],
    'backpack': ['l81'],
    'umbrella': [],
    'handbag': ['l81'],
    'tie': ['l68', 'l78'],
    'suitcase': ['l81'],
    'frisbee': ['l10'],
    'skis': ['l4', 'l17'],
    'snowboard': ['l4', 'l17'],
    'sports ball': [],
    'kite': ['l90', 'l39', 'l26'],
    'baseball bat': [],
    'baseball glove': [],
    'skateboard': [],
    'surfboard': ['l6'],
    'tennis racket': [],
    'bottle': ['l0', 'l8'],
    'wine glass': ['l44'],
    'cup': ['l44'],
    'fork': ['l83'],
    'knife': ['l83'],
    'spoon': ['l83'],
    'bowl': ['l83'],
    'banana': [],
    'apple': [],
    'sandwich': [],
    'orange': [],
    'broccoli': [],
    'carrot': [],
    'hot dog': [],
    'pizza': [],
    'donut': [],
    'cake': [],
    'chair': ['l66'],
    'couch': ['l66'],
    'potted plant': ['l13'],
    'bed': [],
    'dining table': ['l83'],
    'toilet': [],
    'tv': [],
    'laptop': [],
    'mouse': [],
    'remote': [],
    'keyboard': [],
    'cell phone': [],
    'microwave': [],
    'oven': [],
    'toaster': [],
    'sink': [],
    'refrigerator': [],
    'book': ['l88', 'l89'],
    'clock': [],
    'vase': [],
    'scissors': [],
    'teddy bear': [],
    'hair drier': [],
    'toothbrush': []
}


# -----------------------------------------------------------------

submition_data = {}
for img in results:
    submition_data[img] = []
    r = results[img]['results']
    for obj in r:
        try:
            submition_data[img].append(coco_tartu_map[obj])
        except KeyError:
            continue

for img in submition_data:
    print(submition_data[img])
    if len(submition_data[img]) == 0:
        submition_data[img].append('l1')
        submition_data[img] = set(submition_data[img])
    else:
        submition_data[img] = set(list(numpy.concatenate(submition_data[img]).flat))
print(results)
print(submition_data)


# ----------------------------------------------------------------

with open('./submission_siim.csv', 'w', newline='') as csvfile:
    fieldnames = ['image_id', 'labels']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for img in submition_data:
        labels = ''
        for l in submition_data[img]:
            labels += l + ' '
        labels = labels.rstrip()

        writer.writerow({'image_id': img, 'labels': labels})


# -------------------------------------------------

