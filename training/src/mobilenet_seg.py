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
from nets import mobilenet_v1




def mobilenetv1_seg(image, depth_multiplier=1.0, is_training=True):
	if image.get_shape().ndims != 4:
		raise ValueError('Input must be of size [batch, height, width, 3]')

	with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=is_training)):
		#net, endpoints = mobilenet_v1.mobilenet_v1(image, num_classes=False, depth_multiplier=depth_multiplier, is_training = is_training, global_pool=False, spatial_squeeze=False)
		net, endpoints = mobilenet_v1.mobilenet_v1_base(image, depth_multiplier=depth_multiplier, output_stride=8, scope='MobilenetV1')
		#endpoints['added_deconv1'] = tf.layers.conv2d_transpose(endpoints['Conv2d_13_pointwise'], int(512 * depth_multiplier), [3,3], strides=(2, 2), padding='same')
		#endpoints['added_deconv2'] = tf.layers.conv2d_transpose(endpoints['added_deconv1'], int(256 * depth_multiplier), [3,3], strides=(2,2), padding='same')
		#endpoints['added_deconv3'] = tf.layers.conv2d_transpose(endpoints['added_deconv2'], int(128 * depth_multiplier), [3,3], strides=(2,2), padding='same')
		#endpoints['added_deconv4'] = tf.layers.conv2d_transpose(endpoints['added_deconv3'], int(64 * depth_multiplier), [3,3], strides=(2,2), padding='same')
		#endpoints['added_conv'] = tf.layers.conv2d(endpoints['added_deconv3'], 1, [1,1], padding='same', name='output')
		
		#image_shape = tf.shape(image)
		#endpoints['added_resize'] = tf.image.resize_bilinear(endpoints['Conv2d_13_pointwise'], [image_shape[1], image_shape[2]], name='resize' )
		#endpoints['added_conv'] = tf.layers.conv2d(endpoints['added_resize'], 1, [1,1], padding='same', name='output')
		#endpoints['final_output'] = tf.sigmoid(endpoints['added_conv'], name='sigmoid_output')	

		image_shape = tf.shape(image)
		endpoints['added_resize'] = tf.image.resize_bilinear(endpoints['Conv2d_13_pointwise'], [image_shape[1], image_shape[2]], name='resize' )
		endpoints['added_spconv'] = tf.layers.separable_conv2d(endpoints['added_resize'], int(512 * depth_multiplier), [3,3], padding='same')
		endpoints['added_conv'] = tf.layers.conv2d(endpoints['added_spconv'], 1, [1,1], padding='same', name='output')
		endpoints['final_output'] = tf.sigmoid(endpoints['added_conv'], name='sigmoid_output')	


	return endpoints['final_output'], 'MobilenetV1'
	
	#return endpoints['added_conv'], 'MobilenetV1'
