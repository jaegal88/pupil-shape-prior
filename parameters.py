import tensorflow as tf
import os
import re
import numpy as np
import OS_Server_Set

Size_X = 192
Size_Y = 144
os_server = OS_Server_Set.os_server_num
curr = OS_Server_Set.current_operating_on
batchsize = 2
shuffle_buffer = 16000
total_epoch = 10
initial_learning_rate = 1e-4
final_learning_rate = 1e-6
rate_decay = 0.95

ratio = '400'

if Size_X == 192:
    size_name = '192-144'

tfFolderPath_splitted = './tfRecords/'

tf_size = [6554, 504, 9799, 2655, 2831, 2135, 4400, 4890, 630, 840, 655, 524, 491, 469, 13454, 363, 392, 268, 10630, 10258, 9094, 10304, 636, 961]

tf_size = tf_size+tf_size

def tf_path_splitted(value):
    folder_list = ['train000.tfrecords', 'train001.tfrecords', 'train002.tfrecords', 'train003.tfrecords', 'train004.tfrecords', 'train005.tfrecords',
                   'train006.tfrecords', 'train007.tfrecords', 'train008.tfrecords', 'train009.tfrecords', 'train010.tfrecords', 'train011.tfrecords',
                   'train012.tfrecords', 'train013.tfrecords', 'train014.tfrecords', 'train015.tfrecords', 'train016.tfrecords', 'train017.tfrecords',
                   'train018.tfrecords', 'train019.tfrecords', 'train020.tfrecords', 'train021.tfrecords', 'train022.tfrecords', 'train023.tfrecords']
    folder_list_double = folder_list + folder_list
    list_selected = folder_list_double[value + 1:value + len(folder_list)]
    size_selected = tf_size[value+1: value + len(folder_list)]

    if ratio == '25':
        size_ = int(sum(size_selected) / 4)
    elif ratio == '100':
        size_ = sum(size_selected)
    elif ratio == '400':
        size_ = 23*400*10
    list_selected_ten = []
    for ii in range(10):
        list_selected_ten += list_selected

    for jj in range(10):
        for ii in range(len(list_selected)):
            list_selected_ten[jj*len(list_selected)+ii] = tfFolderPath_splitted + str(jj).zfill(2) +list_selected_ten[jj*len(list_selected)+ii]

    return [list_selected_ten, size_]

def find_latest_model_name(name_load_model, value):
    name_load_model = name_load_model + str(value).zfill(3) + '/'

    model_list = os.listdir(name_load_model)
    max_num = 0
    try:
        for s in model_list:
            numbers = int(re.findall('\d+', s)[0])
            if numbers > max_num:
                max_num = numbers
                result_value = s
        result_value = name_load_model + result_value
        print(C_GREEN + result_value + C_END)
        return result_value
    except:
        return 0

def get_digits(text):
    return filter(str.isdigit, text)

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

feature = {'train/image': tf.FixedLenFeature([], tf.string),
           'train/label': tf.FixedLenFeature([], tf.string)}

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.parse_single_example(example_proto, feature)

def calc_lr(initial_lr, steps, global_step):
    """Sets the learning rate to the initial LR"""
    lr = initial_lr * (rate_decay ** (steps//global_step))
    if lr < final_learning_rate:
        lr = final_learning_rate
    return lr

def adjust_learning_rate(optimizer, num_iter, global_step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = calc_lr(initial_learning_rate, num_iter, global_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

C_END = "\033[0m"
C_BOLD = "\033[1m"
C_INVERSE = "\033[7m"

C_BLACK = "\033[30m"
C_RED = "\033[31m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_BLUE = "\033[34m"
C_PURPLE = "\033[35m"
C_CYAN = "\033[36m"
C_WHITE = "\033[37m"

C_BGBLACK = "\033[40m"
C_BGRED = "\033[41m"
C_BGGREEN = "\033[42m"
C_BGYELLOW = "\033[43m"
C_BGBLUE = "\033[44m"
C_BGPURPLE = "\033[45m"
C_BGCYAN = "\033[46m"
C_BGWHITE = "\033[47m"
