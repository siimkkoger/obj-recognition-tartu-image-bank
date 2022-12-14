import csv
import numpy
import json

from yolo7_detect import detect_customs

directory = f'../../competition_data/test_images'
results = detect_customs(directory)
coco_tartu_map = json.load(open('../../../../yolov7/coco_tartu_mapping.json'))

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
