#keep compatability among different python version
from __future__ import division, print_function, absolute_import

import os
import random
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import multiprocessing

from src.nyu_preprocessing import translationHand, scaleHand, rotateHand, joint3DToImg, jointImgTo3D, cropImage
from src.nyu_preprocessing import multiProcess, serialProcess
from src.util import world2pixel, get_nyu_dataset
from src.util import elapsed, save_results

import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES']='2,3'

#Define our input and output data
#load all augmented depth images and labels
dir = get_nyu_dataset()  # nyu dataset

X_in_image = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 1], name='X_in_image')
X_in_label = tf.placeholder(dtype=tf.float32, shape=[None, 42], name='X_in_label')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

steps = []
dis_loss_list = []

dataset_path_train = dir + 'train/'
label_path_train = dir + 'train/joint_data.mat'
epoches = 50
batch_size = 256
learning_rate = 0.0001

labels = sio.loadmat(label_path_train)
joint_uvd = labels['joint_uvd'][0]  # shape: (72757,36,3)
joint_xyz = labels['joint_xyz'][0]  # shape: (72757,36,3)
joint_id =  np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32])

"""
========================================
Test the model
========================================
"""
def test_model(dir, epoch):
    """
    load the test sets
    """
    data_names =   ['test']
    cube_sizes =   [320]
    id_starts =    [0]
    id_ends =      [8252]
    num_packages = [3]
    joint_id = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32])

    depth_center = np.zeros(((8252, 128, 128)))
    joint_xyz_center = np.zeros(((8252, 14, 3)))

    test_image_path = dir + 'nyu_image_test.npy'
    test_label_path = dir + 'nyu_label_test.npy'
    if 'nyu_image_test.npy' in os.listdir(dir) and 'nyu_label_test.npy' in os.listdir(dir):
        image_test = np.load(test_image_path).reshape([-1, 128, 128, 1])
        label_test = np.load(test_label_path).reshape([-1, 42])
    else:
        for D in range(0, len(data_names)):

            data_name = data_names[D]   # train
            cube_size = cube_sizes[D]   # 340
            id_start = id_starts[D]     # 0
            id_end = id_ends[D]         # 72756
            chunck_size = (id_end - id_start) / num_packages[D] #(72756-0)/3 = 24252

            data_type = 'train' if data_name == 'train' else 'test'  # train
            data_path = '{}/{}'.format(dir, data_type)      # NYU/train/
            label_path = '{}/joint_data.mat'.format(data_path)       # NYU/train/joint_data.mat

            print(label_path)

            labels = sio.loadmat(label_path)
            joint_uvd = labels['joint_uvd'][0]  # shape: (72757,36,3)
            joint_xyz = labels['joint_xyz'][0]  # shape: (72757,36,3)

            for id in range(id_start, id_end):

                img_path = '{}/depth_1_{:07d}.png'.format(data_path, id + 1)  # NYU/train/depth_1_{:07d}.png
                print(img_path)

                if not os.path.exists(img_path):
                    print('{} Not Exists!'.format(img_path))
                    continue

                img = cv2.imread(img_path) # shape:(480,640,3)
                ori_depth = np.asarray(img[:, :, 0] + img[:, :, 1]*256) #shape: (480,640)

                depth = cropImage(ori_depth, joint_uvd[id, 34], cube_size=cube_sizes[0]) #shape: (128,128)
                com3D = joint_xyz[id, 34] # shape: (3, )

                depth_crop = (depth - com3D[2]) / (cube_sizes[0] / 2)
                joint_center = (joint_xyz[id][joint_id] - com3D) / (cube_sizes[0] / 2) # shape: (31,3)->(14,3)

                depth_center[id] = depth_crop
                joint_xyz_center[id] = joint_center

        image_test = depth_center.reshape([-1, 128, 128, 1])
        label_test = joint_xyz_center
        np.save(test_image_path, image_test)
        np.save(test_label_path, label_test)

    test_dir = os.path.join(dir, 'test/')
    labels_test = sio.loadmat(test_dir + "joint_data.mat")
    joint_uvd_test = labels_test['joint_uvd'][0]  # shape: (8252,36,3)
    joint_xyz_test = labels_test['joint_xyz'][0]  # shape: (8252,36,3)
    joint_id = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32])


    """
    # calculate the average error and maxiunum error
    """
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(init)
        saver.restore(sess, "../model/nyu_model_{}.ckpt".format(epoch))

        # obtain the ground-truth joints
        labels_norm = np.zeros(((8252, 14, 3)))
        for i, labels_gt in enumerate(label_test):
            labels_norm[i] = labels_gt.reshape([14,3]) * 150 + joint_xyz_test[i,34]

        # obtain the joints that network outputs
        outputs = np.zeros((8252,42))
        for i in range(0, label_test.shape[0]//50):
            image = image_test[i*50:(i+1)*50]
            outputs[i*50:(i+1)*50] = sess.run(pred, feed_dict={X_in_image: image, keep_prob: 1})

        outputs_labels = np.zeros(((8252, 14, 3)))
        for i in range(0, 8252):
            outputs_labels[i] = outputs[i].reshape([14,3]) * 150. + joint_xyz_test[i,34]

        assert labels_norm.shape == outputs_labels.shape

        # calculate the average error, max error between the ground-truth and the prediction joints
        average_error = np.nanmean(np.nanmean(np.sqrt(np.square(labels_norm - outputs_labels).sum(axis=2)), axis=1))
        max_error = np.nanmax(np.sqrt(np.square(labels_norm - outputs_labels).sum(axis=2)))

        print("average_error: %f mm"%(average_error))
        print("max_error: %f mm\n"%(max_error))

        # save the uvd prediction joints
        result_dir = '../result/'
        out_file = os.path.join(result_dir, "epoch_%d_%.2fmm.txt" % (epoch, float(average_error)))
        outputs_labels_uvd = world2pixel(outputs_labels, 588.036865, 587.075073, 320, 240)

        save_results(outputs_labels_uvd, out_file)

    return average_error


"""
========================================
Train the model
========================================
"""
class network():

    def __init__(self):
        pass

    def conv_op(self, x, name, n_out, training, useBN, kh=3, kw=3, dh=1, dw=1, padding="SAME", activation=tf.nn.relu):

        n_in = x.get_shape()[-1].value

        with tf.name_scope(name) as scope:
            w = tf.get_variable(scope + "w", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b = tf.get_variable(scope + "b", shape=[n_out], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.01))
            conv = tf.nn.conv2d(x, w, [1, dh, dw, 1], padding=padding)
            z = tf.nn.bias_add(conv, b)
            if useBN:
                z = tf.layers.batch_normalization(z, trainable=training)
            if activation:
                z = activation(z)
            return z

    def max_pool_op(self, x, name, kh=2, kw=2, dh=2, dw=2, padding="SAME"):

        return tf.nn.max_pool2d(x,
                              ksize=[1, kh, kw, 1],
                              strides=[1, dh, dw, 1],
                              padding=padding,
                              name=name)

    def fc_op(self, x, name, n_out, activation=tf.nn.relu):

        n_in = x.get_shape()[-1].value

        with tf.name_scope(name) as scope:
            w = tf.get_variable(scope + "w", shape=[n_in, n_out],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(scope + "b", shape=[n_out], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.01))

            fc = tf.matmul(x, w, name=name) + b

            out = activation(fc)

        return fc, out

    def res_block_layers(self, x, name, n_out_list, change_dimension=False, block_stride=1):

        if change_dimension:
            short_cut_conv = self.conv_op(x, name + "_ShortcutConv", n_out_list[1], training=True, useBN=True, kh=1,
                                          kw=1,
                                          dh=block_stride, dw=block_stride,
                                          padding="SAME", activation=None)
        else:
            short_cut_conv = x

        block_conv_1 = self.conv_op(x, name + "_lovalConv1", n_out_list[0], training=True, useBN=True, kh=1, kw=1,
                                    dh=block_stride, dw=block_stride,
                                    padding="SAME", activation=tf.nn.relu)

        block_conv_2 = self.conv_op(block_conv_1, name + "_lovalConv2", n_out_list[0], training=True, useBN=True, kh=3,
                                    kw=3,
                                    dh=1, dw=1,
                                    padding="SAME", activation=tf.nn.relu)

        block_conv_3 = self.conv_op(block_conv_2, name + "_lovalConv3", n_out_list[1], training=True, useBN=True, kh=1,
                                    kw=1,
                                    dh=1, dw=1,
                                    padding="SAME", activation=None)

        block_res = tf.add(short_cut_conv, block_conv_3)
        res = tf.nn.relu(block_res)
        return res

    def bulid_resnet(self, x, training=True, usBN=True):

        conv1 = self.conv_op(x, "conv1", 32, training, usBN, 3, 3, 1, 1)
        # conv2 = self.conv_op(conv1, "conv2", 32, training, usBN, 3, 3, 1, 1)
        pool1 = self.max_pool_op(conv1, "pool1", kh=2, kw=2)
        # print("pool1.shape", pool1.shape)

        block1_1 = self.res_block_layers(pool1, "block2_1", [64, 256], True, 2)
        block1_2 = self.res_block_layers(block1_1, "block2_2", [64, 256], False, 1)
        block1_3 = self.res_block_layers(block1_2, "block2_3", [64, 256], False, 1)

        block2_1 = self.res_block_layers(block1_3, "block2_1", [128, 512], True, 2)
        block2_2 = self.res_block_layers(block2_1, "block2_2", [128, 512], False, 1)
        block2_3 = self.res_block_layers(block2_2, "block2_3", [128, 512], False, 1)
        block2_4 = self.res_block_layers(block2_3, "block2_4", [128, 512], False, 1)
        # print("block2_4", block2_4.shape)

        block3_1 = self.res_block_layers(block2_4, "block3_1", [256, 1024], True, 2)
        block3_2 = self.res_block_layers(block3_1, "block3_2", [256, 1024], False, 1)
        block3_3 = self.res_block_layers(block3_2, "block3_3", [256, 1024], False, 1)
        block3_4 = self.res_block_layers(block3_3, "block3_4", [256, 1024], False, 1)
        block3_5 = self.res_block_layers(block3_4, "block3_5", [256, 1024], False, 1)
        block3_6 = self.res_block_layers(block3_5, "block3_6", [256, 1024], False, 1)
        # print("block3_6.shape", block3_6)

        block4_1 = self.res_block_layers(block3_6, "block4_1", [256, 1024], True, 2)
        block4_2 = self.res_block_layers(block4_1, "block4_2", [256, 1024], False, 1)
        block4_3 = self.res_block_layers(block4_2, "block4_3", [256, 1024], False, 1)
        # print("block4_3.shape", block4_3)

        #fc = tf.reshape(block4_3, (-1,4*4*1024))
        fc = tf.layers.flatten(block4_3)
        # print("fc.shape", fc.shape)

        fc1, _ = self.fc_op(fc, "fc1", 1024)
        fc1 = tf.layers.dropout(fc1, keep_prob)

        fc2, _ = self.fc_op(fc1, "fc2", 1024)
        fc2 = tf.layers.dropout(fc2, keep_prob)

        fc3, _ = self.fc_op(fc2, "fc3", 42)

        output = tf.multiply(fc3, 1, "output")

        return output


pred = network().bulid_resnet(X_in_image)
loss_joints =  tf.reduce_mean(tf.reduce_sum(tf.squared_difference(pred, X_in_label), 1))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_joints)


'''
Start session and initialize all the variables
'''
def main():

    saver = tf.train.Saver(max_to_keep=1)  # save the latest model
    train_start = time.time()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state('../model/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")
        else:
            print("No Model !!!")

        # epoch iteration
        for epoch in range(epoches):

            idx = list(random.sample(range(0, 72757), 72757))

            start_epoch = time.time()
            # batch at every epoch
            for num_batch in range(0, len(idx) // batch_size):

                ## multi core
                data_processing_start_time = time.time()
                ids = [id for id in idx[batch_size * num_batch: batch_size * (num_batch + 1)]]
                # create a pool object
                pool = multiprocessing.Pool(processes=8)  # cpu_cout = 48
                # map list to target function
                pool_result = pool.map(multiProcess, ids)
                pool.close()
                pool.join()

                batch_image_train = []
                batch_label_train = []
                for image, joint in pool_result:
                    batch_image_train.append(image)
                    batch_label_train.append(joint)
                del pool_result

                ## single core
                # data_processing_start_time = time.time()
                # ids = [id for id in idx[batch_size * num_batch: batch_size * (num_batch + 1)]]
                # batch_image_train, batch_label_train = serialProcess(ids, batch_size)

                idx_batch = list(random.sample(range(0, batch_size), batch_size))
                batch_image = np.array(batch_image_train)[idx_batch].reshape(batch_size,128,128,1)
                batch_joint = np.array(batch_label_train).reshape(batch_size,42)[idx_batch]
                data_processing_time = time.time() - data_processing_start_time

                start_gpu_time = time.time()
                loss = sess.run(loss_joints, feed_dict={X_in_image: batch_image, X_in_label: batch_joint,
                                                        keep_prob: 0.3})  # , Noise: batch_noise

                sess.run(optimizer, feed_dict={X_in_image: batch_image, X_in_label: batch_joint, keep_prob: 0.3})
                gpu_time = time.time() - start_gpu_time

                if num_batch % 10 == 0:
                    print("Epoch: %d, Step: %d,  Dis_loss: %f, data_processing(CPU): %f sec, gpu_processing(GPU): %f sec"
                          % (epoch, num_batch, loss, data_processing_time, gpu_time))

                # save the loss of generator and discriminator
                steps.append(epoch * len(idx) // batch_size + num_batch)
                dis_loss_list.append(loss)

            duration = time.time() - start_epoch
            print("Each epoch costs: ", elapsed(duration))

            saver.save(sess, "../model/nyu_model_{}.ckpt".format(epoch))
            test_model(dir, epoch)

        total_time = time.time() - train_start
        print("The total training time: ", elapsed(total_time))


    '''
    #show the loss of generaor and discriminator at every batch
    '''
    fig = plt.figure(figsize=(8,6))
    plt.plot(steps, dis_loss_list, label='dis_loss')

    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('The loss of train')
    plt.legend()
    plt.legend(loc = 'upper right')
    plt.savefig(os.path.join('../result/', 'loss_curve.png'))

if __name__ == '__main__':

    main()





