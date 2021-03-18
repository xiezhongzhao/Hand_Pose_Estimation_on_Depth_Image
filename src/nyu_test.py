import tensorflow as tf
import os
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from src.nyu_preprocessing import cropImage
from src.visualization import figure_joint_skeleton, crop_joint_img
from src.util import world2pixel, get_nyu_dataset

def checkNodeName(model_path):
    '''
    check the node name
    :param model_path: eg: ../model/nyu_model_0.ckpt
    :return:
    '''
    ckpt = model_path
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(ckpt + '.meta', clear_devices=True)
        graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
        node_list = [n.name for n in graph_def.node]

        for node in node_list:
            print("node_name", node)

def ckpt2pb(input_path, output_path):
    '''
    convert ckpt model to pb model
    :param input_path: eg. ../model/nyu_model_0.ckpt
    :param output_path: eg. ../model/frozen_model.pb
    :return:
    '''
    saver = tf.train.import_meta_graph(input_path + ".meta", clear_devices=True)
    output_nodes = ["output"] # 填写输出节点的名称（node name）

    with tf.Session(graph=tf.get_default_graph()) as sess:
        input_graph_def = sess.graph.as_graph_def()
        saver.restore(sess, input_path)
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                        input_graph_def,
                                                                        output_nodes)
        with open(output_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())


def testGraph(pb_path):
    '''
    test the pb model
    :param image_path:
    :param pb_path:
    :return:
    '''
    with tf.gfile.FastGFile(pb_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        input_image_tensor = sess.graph.get_tensor_by_name("X_in_image:0")
        output_tensor_name = sess.graph.get_tensor_by_name("output:0")

        ### read the depth image
        id = 8000
        dir = get_nyu_dataset()  # nyu dataset
        data_type = 'test'
        data_path = '{}/{}'.format(dir, data_type)
        img_path = '{}/depth_1_{:07d}.png'.format(data_path, id+1)
        img = cv2.imread(img_path)  # shape:(480,640,3)
        ori_depth = np.asarray(img[:, :, 0] + img[:, :, 1] * 256)  # shape: (480,640)

        ### read the joints
        label_path = '{}/joint_data.mat'.format(data_path)
        labels = sio.loadmat(label_path)
        joint_uvd = labels['joint_uvd'][0]  # shape: (72757,36,3)
        joint_xyz = labels['joint_xyz'][0]  # shape: (72757,36,3)
        cube_sizes = [300]

        ### normalize the depth image
        depth = cropImage(ori_depth, joint_uvd[id, 34], cube_size=cube_sizes[0])  # shape: (128,128)
        com3D = joint_xyz[id, 34]
        img = np.array((depth - com3D[2]) / (cube_sizes[0] / 2)).reshape(-1, 128, 128, 1)

        ### run the model to predict the 3d joints
        result = sess.run(output_tensor_name, feed_dict={input_image_tensor: img})

        result_world = result.reshape([14,3]) * 150. + joint_xyz[id,34]
        result_camera = world2pixel(result_world.reshape([1, 14, 3]), 588.036865, 587.075073, 320, 240)

        figure_joint_skeleton(ori_depth, result_camera.reshape(14, 3), 0)
        crop_joint_img(ori_depth, result_camera.reshape(14,3), joint_uvd[id, 34])
        plt.show()

if __name__ == '__main__':

    ckpt_path = "../model/nyu_model_1.ckpt"
    pb_path = "../model/frozen_model.pb"

    # checkNodeName(ckpt_path) # check the node name
    if "frozen_model.pb" not in os.listdir("../model/"):
        ckpt2pb(ckpt_path, pb_path)

    testGraph(pb_path)
























