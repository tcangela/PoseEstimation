# Copyright 2018 Zihua Zeng (edvard_hua@live.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===================================================================================
# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
from networks import get_network
import os

from resnet18_seg import resnet_18_seg
from mobilenet_seg import mobilenetv1_seg
from pprint import pprint

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser(description='Tensorflow Pose Estimation Graph Extractor')
parser.add_argument('--size', type=int, default=112)
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint path')
#parser.add_argument('--output_node_names', type=str, default='output/BiasAdd')
parser.add_argument('--output_node_names', type=str, default='sigmoid_output')
parser.add_argument('--output_graph', type=str, default='./model.pb', help='output_freeze_path')
parser.add_argument('--quantize', type=bool, default=False)
args = parser.parse_args()

input_node = tf.placeholder(tf.float32, shape=[1, args.size, args.size, 3], name="image")

with tf.Session() as sess:
    #net = get_network(args.model, input_node, trainable=False)
    #prediction, net = resnet_18_seg(input_node, is_training=False)
    prediction, net = mobilenetv1_seg(input_node, depth_multiplier=0.25, is_training=False)
    
    if args.quantize == True:
         tf.contrib.quantize.create_eval_graph()

    tf.train.write_graph(sess.graph, "./", 'graph_eval.pbtxt', as_text=True)

    saver = tf.train.Saver()
    print(args.checkpoint)
    saver.restore(sess, args.checkpoint)

    input_graph_def = tf.get_default_graph().as_graph_def()



    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,  # The session
        input_graph_def,  # input_graph_def is useful for retrieving the nodes
        args.output_node_names.split(",")
    )


with tf.gfile.GFile(args.output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
