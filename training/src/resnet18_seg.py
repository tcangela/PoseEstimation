# Copyright 2018 Chu Tang (chutang1022@gmail.com)
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
import tensorflow.contrib.slim as slim
import resnet_v2_18_200 as resnet_v2_18


def resnet_18_seg(image, is_training=True):
	if image.get_shape().ndims != 4:
		raise ValueError('Input must be of size [batch, height, width, 3]')

	with slim.arg_scope(resnet_v2_18.resnet_arg_scope()):
		net, endpoints = resnet_v2_18.resnet_v2_18(inputs=image,
                                                 num_classes=None,
                                                 is_training=is_training,
                                                 global_pool=False,
                                                 output_stride=16)
		endpoints['added_deconv1'] = tf.layers.conv2d_transpose(net, 512, [3,3], strides=(2, 2), padding='same')
		endpoints['added_deconv2'] = tf.layers.conv2d_transpose(endpoints['added_deconv1'], 256, [3,3], strides=(2,2), padding='same')
		endpoints['added_deconv3'] = tf.layers.conv2d_transpose(endpoints['added_deconv2'], 128, [3,3], strides=(2,2), padding='same')
		endpoints['output'] = tf.layers.conv2d(endpoints['added_deconv3'], 1, [1,1], padding='same')
	
	return endpoints['output']
		
		
