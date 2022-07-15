import numpy as np
import tensorflow as tf
import csv
from PIL import Image
import matplotlib.pyplot as plt

# Define Dice coefficient metric
def dice_coefficient(y_true, y_pred, smooth=1.):
    # compatible with evaluation in prediction loop
    smooth = tf.constant(smooth)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    denom = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + smooth
    dice = (tf.constant(2.) * intersection + smooth)/denom
    return dice


folders = ['./valid_masks/','./masks_result_fp16/']
thresh_upper = [0.7,0.7,0.7,0.7]
thresh_lower = [0.4,0.5,0.4,0.5]
min_area     = [180, 260, 200, 500]
dices = [], [], [], []; 


with open('valid_files.csv', newline='') as csvvalid:
  reader = csv.reader(csvvalid, delimiter=',', quotechar='|')
  i = 0
  for im_i in range(2512):
    row = reader.__next__()
    mask_name = row[2]

    with Image.open(folders[0] + mask_name) as gt_mask_:
        gt_mask = np.array(gt_mask_)/255
    with Image.open(folders[1] + mask_name) as pred_mask_:
        pred_mask = np.array(pred_mask_)/255

    for ch in range(4):
        # Flatten output to compare
        ch_truth = gt_mask[ch*128:(ch+1)*128,:].reshape(-1)
        ch_probs = pred_mask[ch*128:(ch+1)*128,:].reshape(-1)
    
        ch_pred = (ch_probs > thresh_upper[ch]).astype(int)
        if ch_pred.sum() < min_area[ch]: ch_pred = np.zeros([128*800])
        else:
            ch_pred = (ch_probs > thresh_lower[ch]).astype(int)

        dc = dice_coefficient(ch_truth, ch_pred)
        dices[ch].append(dc.numpy())


print('Average Dice coefficient:', np.mean(dices))
ch_dices = ['Class {}: {:.4f}'.format(i+1, dc) for i, dc in zip(range(4), [np.mean(d) for d in dices])]
print('\nAverage Dice coefficient per type of defect: ', (4*'\n{}').format(*ch_dices))
