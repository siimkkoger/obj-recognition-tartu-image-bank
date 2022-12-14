import glob
import os
import csv
import shutil

import numpy

test_images = []
with open(f'competition_data/test.csv', newline='') as csvfile:
    test_images_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in test_images_reader:
        test_images.append(row)
test_images = list(numpy.concatenate(test_images[1:]).flat)
print(test_images)

src_dir = "competition_data/images"
dst_dir = "competition_data/test_images"
for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
    #shutil.copy(jpgfile, dst_dir)
    name = jpgfile.replace('competition_data/images/', '')
    if name in test_images:
        shutil.copy(jpgfile, dst_dir)