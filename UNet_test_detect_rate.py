from __future__ import print_function

import argparse
import torch
import time
import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

import parameters
import Model_UNet_Segmentation
import cv2
import common

opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
conf = tf.ConfigProto(gpu_options=opts)
tfe.enable_eager_execution(config=conf)
nf_nch = '4f32ch'
Size_X = parameters.Size_X
Size_Y = parameters.Size_Y

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='UNet test')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', ## Total Batch
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', ## Epoch
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', ## Learning Rate
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--cross_val_num', type=int, default=14,
                        help='For Cross Validation')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    name_load_model = './trained_model/UNet/'
    # name_load_model = './trained_model_bk/Cross_validation/base/4f32ch/'
    cross_val_num = parser.parse_args().cross_val_num



    start_time = time.time()
    end_time = time.time()
    image_num = 94113
    avg_cost= 0
    model = Model_UNet_Segmentation.UNet4f32ch_sigmoid()
    if os.path.exists(name_load_model):
        load_saved_model_name = parameters.find_latest_model_name(name_load_model, cross_val_num)
        model.load_state_dict(torch.load(load_saved_model_name))
        print(parameters.C_GREEN + 'Check point Successfully Loaded' + parameters.C_END)
    else:
        print(parameters.C_RED + 'Check point Not Found' + parameters.C_END)

    model.eval()
    model.to(device)
    print('Test Started!')


    image_path = './ExCuSe_Origin/' #1~10까지만의 데이터셋
    # image_path = 'C:/Users/HanSY/Python/Dataset/ExCuSe/gray/' #1~10까지만의 데이터셋

    GTtxt_list = './GT_label_New/'
    result_path = './result_files/'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    txt_list = os.listdir(GTtxt_list)
    cnt = 0
    rangefiles = range(len(txt_list))  # num_sequence
    numfile = 0
    for i in rangefiles:
        numfile += len(os.listdir(image_path + txt_list[i].replace(".txt", "")))
    print(str(numfile) + ' files')
    prev_x = 0
    prev_y = 0
    for i in rangefiles:

        if i != cross_val_num:
            continue

        sub_folder_name = txt_list[i].replace(".txt", "")

        txtFile = open(GTtxt_list + '/' + txt_list[i], 'r')
        line = txtFile.readline()
        resultTxt = open(result_path + txt_list[i], 'w')
        resultTxt.write(line)
        for j in range(len(os.listdir(image_path + sub_folder_name))):
            cnt = cnt + 1
            ## print expected time required
            if cnt == 2:
                start_time = time.time()
            if cnt == 12:
                end_time = time.time()
                cost = end_time - start_time
                common.print_expected_time_test(numfile,start_time,cost)
            ###############################
            line = txtFile.readline()
            file_name = sub_folder_name + '/' + line.split(" ")[1].zfill(10) + '.png'
            inputImg = cv2.imread(image_path + file_name)
            inputImg = inputImg[:, :, 0]
            inputImg = cv2.resize(inputImg, (Size_X, Size_Y), interpolation=cv2.INTER_CUBIC)
            inputImg_BK = inputImg.copy()
            inputImg = inputImg[np.newaxis, np.newaxis, :]
            inputImg = inputImg.astype(np.float32)/255

            image = torch.from_numpy(inputImg)
            image = image.to(device)
            output = model(image)
            output_bk = output[:, 0].clone().detach().cpu().numpy()
            result_temp = output_bk.copy()
            ttt = output_bk
            ttt[ttt < 0.5] = 0
            ttt[ttt >= 0.5] = 1
            for_print_out = np.zeros((Size_Y, Size_X), np.float32)
            if np.count_nonzero(ttt) == 0:
                output_bk[output_bk < 0.25] = 0
                output_bk[output_bk >= 0.25] = 1
            else:
                output_bk[output_bk < 0.5] = 0
                output_bk[output_bk >= 0.5] = 1

            ## Connected Component Analysis
            if np.count_nonzero(output_bk) != 0:
                _, labels, stats, center = cv2.connectedComponentsWithStats(output_bk[0, :, :].astype(np.uint8))

                stats = stats[1:, :]
                pupil_candidate = np.argmax(stats[:, 4]) + 1
                txt = line.split(" ")[0] + ' ' + line.split(" ")[1] + ' ' + str(round(center[pupil_candidate][0]*2, 3)) + ' ' + str(round(center[pupil_candidate][1]*2, 3)) + '\n'
                output_bk[0, :, :][labels != pupil_candidate] = 0
                prev_x = round(center[pupil_candidate][0] * 2, 3)
                prev_y = round(center[pupil_candidate][1]*2, 3)

            else:
                if cnt == 0:
                    result_temp = cv2.blur(result_temp, (21, 21))
                    max_indices = np.unravel_index(result_temp.argmax(), result_temp.shape)
                    txt = line.split(" ")[0] + ' ' + line.split(" ")[1] + ' ' + str(max_indices[1]*2) + ' ' + str(max_indices[2]*2) + '\n'
                    prev_x = max_indices[1]*2
                    prev_y = max_indices[2]*2
                else:
                    txt = line.split(" ")[0] + ' ' + line.split(" ")[1] + ' ' + str(prev_x) + ' ' + str(prev_y) + '\n'
            resultTxt.write(txt)

            savename = result_path + file_name
            cv2.imwrite(savename, (output_bk[0]*255+inputImg_BK)/2)

            if cnt % 500 == 0:
                print(cnt)
        resultTxt.close()
        sys.stdout.flush()
    now = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    print(s)

if __name__ == '__main__':
    main()
