from __future__ import print_function

import argparse
import torch
import time
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

import parameters
import Model_UNet_Segmentation
import cv2
import warnings
import common
import math

# device = parameters.device
nf_nch = '4f32ch'

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--cross_val_num', type=int, default=18,
                        help='For Cross Validation')
    parser.add_argument('--gpu_num', type=str, default=parameters.os_server)

    os.environ['CUDA_VISIBLE_DEVICES'] = parser.parse_args().gpu_num
    opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    conf = tf.ConfigProto(gpu_options=opts)
    tfe.enable_eager_execution(config=conf)

    use_cuda = torch.cuda.is_available()
    cross_val_num = parser.parse_args().cross_val_num
    torch.manual_seed(1)
    warnings.filterwarnings("ignore")

    device = torch.device("cuda" if use_cuda else "cpu")
    [tf_path, image_num] = parameters.tf_path_splitted(cross_val_num)

    batchsize = parameters.batchsize
    shuffle_buffer = image_num

    global_step = int(image_num / batchsize)
    Size_X = parameters.Size_X
    Size_Y = parameters.Size_Y

    raw_image_dataset = tf.data.TFRecordDataset(tf_path)
    parsed_image_dataset = raw_image_dataset.map(parameters._parse_image_function).shuffle(buffer_size=int(shuffle_buffer/10)).batch(
        batch_size=batchsize).repeat()

    result_path = './result_image/'
    trained_model_path = './trained_model/'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(trained_model_path):
        os.mkdir(trained_model_path)
    trained_model_path = trained_model_path + 'UNet/'
    if not os.path.exists(trained_model_path):
        os.mkdir(trained_model_path)
    trained_model_path = trained_model_path + str(cross_val_num).zfill(3) +'/'
    if not os.path.exists(trained_model_path):
        os.mkdir(trained_model_path)

    model = Model_UNet_Segmentation.UNet4f32ch_sigmoid()

    print(parameters.C_GREEN + nf_nch + ', ' + str(cross_val_num).zfill(3) + ' : ' + str(image_num) + ' images' + parameters.C_END)

    name_load_model = './trained_model/UNet/'
    try:
        load_saved_model_name = parameters.find_latest_model_name(name_load_model, cross_val_num)
        model.load_state_dict(torch.load(load_saved_model_name))
        print('Model Loaded')
    except:
        print('Falied To Load Trained Model')

    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=parameters.initial_learning_rate)


    start_time = time.time()
    end_time = time.time()
    current_epoch = 0
    current_batch = 0
    iter = 0
    total_batch = int(image_num / batchsize)

    avg_cost = 0
    folder_num = 0
    print('Learning Started!\tBatch Size : {}\tTotal Epoch : {}'.format(batchsize, parameters.total_epoch))

    for image_features in parsed_image_dataset:
        image = tf.decode_raw(image_features['train/image'], tf.uint8)
        label = tf.decode_raw(image_features['train/label'], tf.uint8)
        image = tf.reshape(image, [-1, Size_Y, Size_X])
        label = tf.reshape(label, [-1, Size_Y, Size_X])
        label_inv = 255 - label
        image = tf.expand_dims(image, 1)
        label = tf.expand_dims(label, 1)
        label_inv = tf.expand_dims(label_inv, 1)
        label_2ch = tf.concat([label, label_inv], axis=1)
        image = image.numpy().astype(np.float32)/255
        label_2ch = label_2ch.numpy().astype(np.float32)/255
        label = label.numpy().astype(np.float32) / 255
        label = torch.from_numpy(label)
        label = label.to(device)

        image = torch.from_numpy(image)
        label_2ch = torch.from_numpy(label_2ch)

        image, label_2ch = image.to(device), label_2ch.to(device)
        output = model(image)
        output = nn.functional.sigmoid(output)

        # Shape Prior Loss
        #########################################
        output_bk = output.clone().detach()
        bat_len = len(output_bk[:, 0, 0, 0])
        output_bk = output_bk.cpu().numpy()
        output_bk[output_bk < 0.5] = 0
        output_bk[output_bk >= 0.5] = 1
        loss_t2 = 0
        num2 = 0
        flag = False
        ###########################################
        for bat in range(bat_len):
            if np.count_nonzero(output_bk[bat, 0, :, :]) != 0:
                _, labels, stats, center = cv2.connectedComponentsWithStats(output_bk[bat, 0, :, :].astype(np.uint8))
                stats = stats[1:, :]
                pupil_candidate = np.argmax(stats[:, 4]) + 1
                center_pt = (int(center[pupil_candidate][0]), int(center[pupil_candidate][1]))
                output_bk[bat, 0, :, :][labels != pupil_candidate] = 0
                edges = cv2.Canny(output_bk[bat, 0, :, :].astype(np.uint8), 0.25, 0.5)

                temp_mat = np.zeros((Size_Y, Size_X), np.uint8)
                cv2.line(temp_mat, (center_pt[0] + 100, center_pt[1] + 100), (center_pt[0] - 100, center_pt[1] - 100),
                         255)
                cv2.line(temp_mat, (center_pt[0] + 100, center_pt[1] - 100), (center_pt[0] - 100, center_pt[1] + 100),
                         255)
                cv2.line(temp_mat, (center_pt[0] + 100, center_pt[1]), (center_pt[0] - 100, center_pt[1]), 255)
                cv2.line(temp_mat, (center_pt[0], center_pt[1] + 100), (center_pt[0], center_pt[1] - 100), 255)

                cv2.line(temp_mat, (center_pt[0] + 100, center_pt[1] + 241), (center_pt[0] - 100, center_pt[1] - 241), 255)
                cv2.line(temp_mat, (center_pt[0] + 100, center_pt[1] - 241), (center_pt[0] - 100, center_pt[1] + 241), 255)
                cv2.line(temp_mat, (center_pt[0] + 241, center_pt[1] + 100), (center_pt[0] - 241, center_pt[1] - 100), 255)
                cv2.line(temp_mat, (center_pt[0] + 241, center_pt[1] - 100), (center_pt[0] - 241, center_pt[1] + 100), 255)
                edges = cv2.bitwise_and(temp_mat, edges)
                if np.sum(edges) != 0:
                    edge_loca = cv2.findNonZero(edges)
                    edge_loca = np.squeeze(edge_loca, axis=1)
                    np.random.shuffle(edge_loca)
                    edge_loca = edge_loca[:min(8, len(edge_loca[:, 0])), :] #extracting random 8 point on boundary points

                    angle_loca_temp = np.zeros((len(edge_loca[:, 0]), 1), np.float32)
                    edge_loca_temp = np.zeros(edge_loca.shape, np.int)
                    delta = edge_loca - center_pt

                    for ind in range(len(edge_loca[:, 0])):
                        angle_loca_temp[ind] = math.degrees(math.atan2(delta[ind, 1], delta[ind, 0]))
                    sort_index = np.argsort(angle_loca_temp.squeeze(axis=1))
                    for ind in range(len(sort_index)):
                        edge_loca_temp[ind, :] = edge_loca[sort_index[ind], :]


                    edge_loca = edge_loca_temp

                    for edge_ind in range(len(edge_loca[:, 0])):
                        edge_x_0 = edge_loca[edge_ind, 0]
                        edge_y_0 = edge_loca[edge_ind, 1]
                        edge_x_1 = edge_loca[(edge_ind+2) % len(edge_loca[:, 0]), 0]
                        edge_y_1 = edge_loca[(edge_ind+2) % len(edge_loca[:, 0]), 1]


                        line_mat = np.zeros((Size_Y, Size_X), np.float32)
                        cv2.line(line_mat, (edge_x_0, edge_y_0), (edge_x_1, edge_y_1), 255)
                        line_mat[edge_y_0, edge_x_0] = 0
                        line_mat[edge_y_1, edge_x_1] = 0

                        if np.sum(line_mat) != 0:
                            line_loca = cv2.findNonZero(line_mat)
                            line_loca = np.squeeze(line_loca, axis=1)
                            np.random.shuffle(line_loca)
                            line_loca = line_loca[:min(8, len(line_loca[:, 0])), :]

                            for [line_x, line_y] in line_loca:
                                if label_2ch[bat, 0, line_y, line_x] == label_2ch[bat, 0, edge_y_0, edge_x_0] == label_2ch[bat, 0, edge_y_1, edge_x_1]:
                                    loss_t2 += torch.abs(label_2ch[bat, 0, edge_y_0, edge_x_0] - output[bat, 0, edge_y_0, edge_x_0])\
                                            * torch.abs(label_2ch[bat, 0, edge_y_1, edge_x_1] - output[bat, 0, edge_y_1, edge_x_1])\
                                            * torch.abs(output[bat, 0, edge_y_0, edge_x_0] + output[bat, 0, edge_y_1, edge_x_1] - 2*output[bat, 0, line_y, line_x])
                                    num2 += 1
                        if flag == True:
                            break
        #########################################
        loss = F.binary_cross_entropy(output, label)+ loss_t2 / (num2+1e-10) * 1e-3 # BCE loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_cost += float(loss) / total_batch
        current_batch = current_batch + 1

        del image, label, label_2ch, label_inv, output, loss, loss_t2, output_bk

        iter = iter + 1
        if iter == 2:
            start_time = time.time()
        if iter == 12:
            end_time = time.time()
            cost = end_time - start_time
            common.print_expected_time_train(total_batch=total_batch, total_epoch=parameters.total_epoch, start_time=start_time, cost=cost)

        if current_batch == total_batch:
            current_epoch = current_epoch + 1
            current_batch = 0
            current_time = time.localtime(time.time())

            print(str(cross_val_num).zfill(3) + ' Time : {:02}:{:02}:{:02}\tEpoch: {:03}\tIters: {:06}\tTraining Loss: {:.6f}\t lr: {:.6f}'.format(
                current_time.tm_hour, current_time.tm_min, current_time.tm_sec,
                current_epoch, iter, avg_cost, parameters.calc_lr(parameters.initial_learning_rate, iter, global_step)))
            avg_cost = 0

            if current_epoch % (parameters.total_epoch / 10) == 0:
                savename = trained_model_path+'my_test_model_' + '%08d' % (iter) + 'iters.pt'
                torch.save(model.state_dict(), savename)
            if current_epoch == parameters.total_epoch:
                break

if __name__ == '__main__':
    main()