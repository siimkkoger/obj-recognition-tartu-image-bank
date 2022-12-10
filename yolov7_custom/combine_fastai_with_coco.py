fastai_to_coco_map = {
    'snow': 'l4',
    'trees': 'l1',
    'buildings': 'l3',
    'leaves': 'l36, l70',
    'bridge': 'l20',
    'beach': 'l54, l22',
    'eesti_lipp': 'l26',
    'stroller': 'l85',
    'stairs': 'l18, l71',
    'crosswalk': 'l76',
    'flowers': 'l13',
    'suudlevad_tudengid': 'l28',
    'tractor': 'l42',
    'town_hall': 'l24',
}

import pandas as pd
import csv
import numpy

labels_df_yolo = pd.read_csv('yolov7/submission_siim.csv')
labels_df_yolo = labels_df_yolo.to_numpy()

labels_df_fastiai = pd.read_csv('submission_siim_fastai.csv')
labels_df_fastiai = labels_df_fastiai.to_numpy()

to_yolo = []
for fai in labels_df_fastiai:
    for label in fastai_to_coco_map:
        fai[1] = fai[1].replace(label, fastai_to_coco_map[label])



submition_data = {}
for img_yolo, img_fastai in zip(labels_df_yolo, labels_df_fastiai):
    img_name = img_yolo[0]
    img_name2 = img_fastai[0]

    if img_name != img_name2:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    submition_data[img_name] = set(img_yolo[1].split(' ') + img_fastai[1].split(' '))

print(submition_data)



with open('./combined_submission_siim.csv', 'w', newline='') as csvfile:
    fieldnames = ['image_id', 'labels']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for img in submition_data:
        labels = ''
        for l in submition_data[img]:
            labels += l + ' '
        labels = labels.rstrip()
        labels = labels.replace(',', '')
        print(labels)

        writer.writerow({'image_id': img, 'labels': labels})