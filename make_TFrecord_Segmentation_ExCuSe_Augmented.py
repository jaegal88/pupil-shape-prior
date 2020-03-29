import numpy as np
import os
import cv2
import tensorflow as tf
import sys
import parameters
import random

flag_train = 'train'
Size_X = parameters.Size_X
Size_Y = parameters.Size_Y

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

grayImgFolderPath = "./ExCuSe_Origin/"
GTImgFolderPath = "./Seg/"
folder_list = os.listdir(GTImgFolderPath)

folder_list_double = folder_list + folder_list

#Extract max number of the eye data sequence
num_lines_max = 0
for train_set_num in range(len(folder_list)):
    sub_folder_name = folder_list_double[train_set_num]
    filenamelist = os.listdir(GTImgFolderPath + sub_folder_name)
    num_lines = len(filenamelist)
    if num_lines > num_lines_max:
        num_lines_max = num_lines

if parameters.ratio == '25':
    num_lines_max = int(num_lines_max/4)

for train_set_num in range(len(folder_list)):
    numb_image = 0
    #For Cross Validate
    sub_folder_name = folder_list_double[train_set_num]
    filenamelist = os.listdir(GTImgFolderPath + sub_folder_name)
    num_lines = len(filenamelist)
    random.shuffle(filenamelist)

    filenamelist = filenamelist[:num_lines]

    while(len(filenamelist) < num_lines_max):
        filenamelist = filenamelist + filenamelist


    for num_split_dataset in range(10):
        test_filename = './tfRecords/' + str(num_split_dataset).zfill(2) + flag_train + str(train_set_num).zfill(3) + '.tfrecords'
        print(test_filename)
        writer = tf.python_io.TFRecordWriter(test_filename)


        for j in range(400):  # image_file_num_insequence_
        # for j in range(num_lines_max):  # image_file_num_insequence_
            grayimg_name = grayImgFolderPath + sub_folder_name + '/' + filenamelist[j+num_split_dataset*10]
            GTimg_name = GTImgFolderPath + sub_folder_name + '/' + filenamelist[j+num_split_dataset*10]
            gray_img = cv2.imread(grayimg_name, cv2.IMREAD_GRAYSCALE)
            binaryGT_img = cv2.imread(GTimg_name, cv2.IMREAD_GRAYSCALE)
            height, width = binaryGT_img.shape
            random_degree = random.uniform(-5.0, 5.0)
            random_scale = random.uniform(0.95, 1.05)
            matrix = cv2.getRotationMatrix2D((width/2, height/2), random_degree, random_scale)

            ########################################## make Affine
            gray_img = cv2.warpAffine(gray_img, matrix, (width, height))
            binaryGT_img = cv2.warpAffine(binaryGT_img, matrix, (width, height))
            binaryGT_img = cv2.threshold(binaryGT_img, 127, 255, cv2.THRESH_BINARY)[1]
            ##################################################

            if gray_img.shape[1] != Size_X:
                gray_img = cv2.resize(gray_img, (Size_X, Size_Y), interpolation=cv2.INTER_CUBIC)
                binaryGT_img = cv2.resize(binaryGT_img, (Size_X, Size_Y), interpolation=cv2.INTER_CUBIC)
                binaryGT_img = cv2.threshold(binaryGT_img, 127, 255, cv2.THRESH_BINARY)[1]

            # Create a feature
            train_set_num_np = np.zeros([24, 1], np.uint8())
            train_set_num_np[train_set_num][0] = 1
            feature = {'train/image': _bytes_feature(tf.compat.as_bytes(gray_img.tostring())),
                       'train/label': _bytes_feature(tf.compat.as_bytes(binaryGT_img.tostring())),
                       'train/seq_num': _bytes_feature(tf.compat.as_bytes(train_set_num_np.tostring()))
                       # 'train/label': _bytes_feature(tf.compat.as_bytes(labelFloat.tostring()))
                       }
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            numb_image = numb_image + 1
            if (numb_image + 1) % 100 == 0:
                print((numb_image + 1), 'images completed.')

    writer.close()
    sys.stdout.flush()




