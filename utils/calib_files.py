import numpy as np
import tensorflow as tf
import csv
import random
import os

# Out of all labeled images remove the ones used for validation
train_list = os.listdir("./train_images")

with open('valid_files.csv', newline='') as csvvalid:
  reader = csv.reader(csvvalid, delimiter=',', quotechar='|')
  i = 0
  for im_i in range(2512):
    row = reader.__next__()
    img_name = row[0]
    train_list.remove(img_name)

# Shuffle to avoid calibrating with a single class
random.shuffle(train_list)

with open('calib_files.csv', 'w', newline='') as csvcalib:
    writer = csv.writer(csvcalib, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for image in train_list:
        writer.writerow([image,'label'])
