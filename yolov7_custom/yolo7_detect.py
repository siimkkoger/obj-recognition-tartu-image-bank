import torch
import time

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages
from yolov7.utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, set_logging
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized


def detect_custom(dataset, model, device, half, img_size=640, conf_thres=0.3, iou_thres=0.45, augment=True,
                  agnostic_nms=True, classes=None):
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = img_size
    old_img_b = 1

    t0 = time.time()

    predictions = {}
    for path, img, im0s, vid_cap in dataset:
        img_name = 'img' + path.split('/img')[1]

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

