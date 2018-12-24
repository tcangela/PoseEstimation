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
import os
import time
import numpy as np
import configparser
#import dataset

from datetime import datetime

#from dataset import get_train_dataset_pipeline, get_valid_dataset_pipeline
from resnet18_seg import resnet_18_seg
from mobilenet_seg import mobilenetv1_seg
from dataset_prepare import CocoPose
from dataset_augment import set_network_input_wh, set_network_scale
import math
import cv2
import random

r_mean = 118.06
g_mean = 113.75
b_mean = 106.56
norm_scale = 0.176

is_rgb = False

def get_loss_and_output(model_name, batchsize, input_image, input_label, reuse_variables=None):
    losses = []

    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
       if model_name == 'resnet18':
          prediction, net_scope = resnet_18_seg(input_image, is_training=True)
       if model_name == 'mobilenetv1':
          prediction, net_scope = mobilenetv1_seg(input_image, is_training=True)
       else:
          raise ValueError('Unsupported model!', model_name)
 
    loss = tf.nn.l2_loss(prediction - input_label, name='loss')
    total_loss = tf.reduce_sum(loss) / batchsize
    return total_loss, prediction, net_scope


def average_gradients(tower_grads):
    """
    Get gradients of all variables.
    :param tower_grads:
    :return:
    """
    average_grads = []

    # get variable and gradients in differents gpus
    for grad_and_vars in zip(*tower_grads):
        # calculate the average gradient of each gpu
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

keys_to_features = {
	'image/height': 
		tf.FixedLenFeature((), tf.int64, 1),
	'image/width':
		tf.FixedLenFeature((), tf.int64, 1),
 	'image/encoded':
		tf.FixedLenFeature((), tf.string, default_value=''),
 	'mask/encoded':
		tf.FixedLenFeature((), tf.string, default_value='')
  }

def find_min_max(kp_x, kp_y):
	valid_kp_x = []
	valid_kp_y = []
	#filter the invisible keypoints
	for idx in range(0, len(kp_x)):
		if kp_x[idx] >= 0 and kp_y[idx] >= 0:
			valid_kp_x.append(kp_x[idx])
			valid_kp_y.append(kp_y[idx])


	x_min = np.amin(valid_kp_x)
	x_max = np.amax(valid_kp_x)
	y_min = np.amin(valid_kp_y)
	y_max = np.amax(valid_kp_y)
	return x_min, x_max, y_min, y_max

MIN_PATCH_SIZE = 0.8 

def random_crop(image, mask):
	with tf.name_scope('crop_image'):
		random_ratio_left = random.uniform(0, 1)
		random_ratio_top =  random.uniform(0, 1)
		random_ratio_right = random.uniform(0, 1)
		random_ratio_bottom = random.uniform(0, 1)

		random_patch_size_width = MIN_PATCH_SIZE + (1.0 - MIN_PATCH_SIZE) * random_ratio_right
		random_patch_size_height = MIN_PATCH_SIZE + (1.0 - MIN_PATCH_SIZE) * random_ratio_bottom

		random_patch_x = (1 - random_patch_size_width) * random_ratio_left
		random_patch_y = (1 - random_patch_size_height) * random_ratio_top

		origin_image_shape = tf.shape(image)
		origin_mask_shape = tf.shape(mask)

		origin_image_shape = tf.cast(origin_image_shape, tf.float64)
		origin_mask_shape = tf.cast(origin_mask_shape, tf.float64)

		image_width = origin_image_shape[1] * random_patch_size_width
		image_height = origin_image_shape[0] * random_patch_size_height
		image_x = origin_image_shape[1] * random_patch_x
		image_y = origin_image_shape[0] * random_patch_y

		mask_width = origin_mask_shape[1] * random_patch_size_width
		mask_height = origin_mask_shape[0] * random_patch_size_height
		mask_x = origin_mask_shape[1] * random_patch_x
		mask_y = origin_mask_shape[0] * random_patch_y

		#now whether to flip
		flip = random.uniform(0, 1) > 0.5

		#do the fucking crop!
		image_begin = [image_y, image_x, 0]
		image_size = [image_height, image_width, 3]

		mask_begin = [mask_y, mask_x, 0]
		mask_size = [mask_height, mask_width, 1]

		image_begin = tf.cast(image_begin, tf.int32)
		image_size = tf.cast(image_size, tf.int32)
		mask_begin = tf.cast(mask_begin, tf.int32)
		mask_size = tf.cast(mask_size, tf.int32)

		cropped_image = tf.slice(image, image_begin, image_size)
		cropped_mask = tf.slice(mask, mask_begin, mask_size)

		

		if flip:
			cropped_image = tf.reverse(cropped_image, [1])
			cropped_mask = tf.reverse(cropped_mask, [1])

	return cropped_image, cropped_mask

	

input_height = 192
input_width = 192
scale = 2


def decode_record(filename_queue):
	with tf.name_scope('decode_record'):
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)
		features = tf.parse_single_example(
				serialized_example,
				keys_to_features)
		image = tf.decode_raw(features['image/encoded'], tf.uint8)
		mask = tf.decode_raw(features['mask/encoded'], tf.uint8)

		height = tf.cast(features['image/height'], tf.int32)
		width = tf.cast(features['image/width'], tf.int32)

		image = tf.reshape(image, [height, width, 3])
		mask = tf.reshape(mask, [height, width, 1])

		image, mask = random_crop(image, mask)

		
		#print(image.get_shape())
		#print(mask.get_shape())

		image = tf.cast(image, tf.float32)
		image = tf.image.resize_images(image, [input_height, input_width])
		
		mask = tf.cast(mask, tf.float32)
		mask = tf.image.resize_images(mask, [int(input_height / scale), int(input_width / scale)])	
	
		image = tf.reshape(image, [input_height, input_width, 3])
		mask = tf.reshape(mask, [int(input_height / scale), int(input_width / scale), 1])

		if not is_rgb:
			image = image[..., ::-1]
		
		print(image.get_shape())
		print(mask.get_shape())

		boolean_mask = tf.greater(mask, 0.5)
		mask = tf.multiply(mask, tf.cast(boolean_mask, mask.dtype))		

		image = (image - [[[b_mean, g_mean, r_mean]]]) * norm_scale
		return image, mask


def fetch_data_batch(filename, batch_size, num_epochs=None):
    '''num_features := width * height for 2D image'''
    print(filename)
    filename_queue = tf.train.string_input_producer(
            filename, shuffle=True)
    #filename, num_epochs = num_epochs, shuffle=True)

    example, label = decode_record(filename_queue)
   
     
    #print('read_decode done.')
    min_after_dequeue = 64
    capacity = min_after_dequeue + 10 * batch_size
   
    example_batch, label_batch = tf.train.shuffle_batch(
            [example, label], batch_size=batch_size, num_threads = 40, capacity=capacity,
            min_after_dequeue=min_after_dequeue)

    print("batch size = %d" % batch_size)
    #example_batch, label_batch = tf.train.batch(
    #        [example, label], batch_size = batch_size, num_threads = 20,
    #        capacity = capacity)

    return example_batch, label_batch





def main(argv=None):
    # load config file and setup
    params = {}
    config = configparser.ConfigParser()
    config_file = "experiments/resnet18_seg.cfg"
    if len(argv) != 1:
        config_file = argv[1]
    config.read(config_file)
    for _ in config.options("Train"):
        params[_] = eval(config.get("Train", _))

    os.environ['CUDA_VISIBLE_DEVICES'] = params['visible_devices']
    gpus_index = params['visible_devices'].split(",")
    params['gpus'] = len(gpus_index)

    if not os.path.exists(params['modelpath']):
        os.makedirs(params['modelpath'])
    if not os.path.exists(params['logpath']):
        os.makedirs(params['logpath'])

    #dataset.set_config(params)
    set_network_input_wh(params['input_width'], params['input_height'])
    set_network_scale(params['scale'])

    training_name = '{}_batch-{}_lr-{}_gpus-{}_{}x{}_{}'.format(
        params['model'],
        params['batchsize'],
        params['lr'],
        params['gpus'],
        params['input_width'], params['input_height'],
        config_file.replace("/", "-").replace(".cfg", "")
    )

    input_height = params['input_height']
    input_width = params['input_width']
    target_size = params['input_height'] // params['scale']
    r_mean = params['r_mean']
    g_mean = params['g_mean']
    b_mean = params['b_mean']
    norm_scale = params['norm_scale']
    is_rgb = params['is_rgb']
    scale = params['scale']	
    init_checkpoint = params['initial_checkpoint']   

    #with tf.Graph().as_default(), tf.device("/cpu:0"):
    with tf.Graph().as_default():
        input_image_batch, input_heat_batch = fetch_data_batch([params['training_data']], params['batchsize'])
        valid_input_image_batch, valid_input_heat_batch = fetch_data_batch([params['validation_data']], params['batchsize'])
       
        input_image = tf.placeholder(tf.float32, shape=[None, params['input_height'], params['input_width'], 3])
        input_heat = tf.placeholder(tf.float32, shape=[None, target_size, target_size, 1])
        


        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(float(params['lr']), global_step,
                                                   decay_steps=10000, decay_rate=float(params['decay_rate']), staircase=True)
        
        opt = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
        
        #opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        tower_grads = []
        reuse_variable = False

        loss, pred_heat, net_scope = get_loss_and_output(params['model'], params['batchsize'], input_image, input_heat, reuse_variable)
        reuse_variable = True
        grads = opt.compute_gradients(loss)
        tower_grads.append(grads)
        valid_loss, valid_pred_heat, _ = get_loss_and_output(params['model'], params['batchsize'], input_image, input_heat, reuse_variable)
 
        grads = average_gradients(tower_grads)
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram("gradients_on_average/%s" % var.op.name, grad)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        MOVING_AVERAGE_DECAY = 0.99
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_to_average = (tf.trainable_variables() + tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variable_to_average)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        init_variables_backbone = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, net_scope)
        with tf.control_dependencies(update_ops):
            train_op = tf.group(apply_gradient_op, variables_averages_op)
            #train_op = tf.group(apply_gradient_op)

        #saver = tf.train.Saver(init_variables_backbone)
        saver = tf.train.Saver()

        tf.summary.scalar("learning_rate", learning_rate)
        tf.summary.scalar("loss", loss)
        summary_merge_op = tf.summary.merge_all()

        pred_summary = tf.summary.image(name='prediction', tensor=valid_pred_heat, max_outputs=8, collections=None, family='segmentation')
        image_summary = tf.summary.image(name='image', tensor=input_image, max_outputs=8, collections=None, family='segmentation')


        #pred_result_image = tf.placeholder(tf.float32, shape=[params['val_batchsize'], 480, 640, 3])
        #pred_result__summary = tf.summary.image("pred_result_image", pred_result_image, params['val_batchsize'])


        #for resume a training
        checkpoint_saver = tf.train.Saver(max_to_keep=0)
        resume = params['resume']

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        # occupy gpu gracefully
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            if resume:
                last_checkpoint = tf.train.latest_checkpoint(params['checkpoint'])
                print ('Restoring from checkpoint: {}'.format(last_checkpoint))
                checkpoint_saver.restore(sess, last_checkpoint)
            else:
                init.run()
                #if init_checkpoint is not None:
                #     saver.restore(sess, init_checkpoint)	

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
            start_step = sess.run(global_step)

            summary_writer = tf.summary.FileWriter(os.path.join(params['logpath'], training_name), sess.graph)
            total_step_num = params['num_train_samples'] * params['max_epoch'] // (params['batchsize'] * params['gpus'])
            print("Start training...")
            for idx in range(start_step, total_step_num):
                start_time = time.time()
                
                #fetch training data
                images, labels = sess.run([input_image_batch, input_heat_batch])
                fetch_time = time.time() - start_time
                #print(step)
                #print(images)
                #print(labels)
                feed_dict = {
                    input_image: images,
                    input_heat: labels
                }

                #_, loss_value, lh_loss, in_image, in_heat, p_heat = sess.run(
                #    [train_op, loss, last_heat_loss, input_image, input_heat, pred_heat]
                #)
                _, merge_op, loss_value, p_heat, step = sess.run(
                    [train_op, summary_merge_op, loss, pred_heat, global_step], feed_dict = feed_dict)
               
                duration = time.time() - start_time



                
                if step != 0 and step % params['per_update_tensorboard_step'] == 0:
                    # False will speed up the training time.
                    if params['pred_image_on_tensorboard'] is True:

                        #valid_loss_value, valid_lh_loss, valid_in_image, valid_in_heat, valid_p_heat = sess.run(
                        #    [valid_loss, valid_last_heat_loss, valid_input_image, valid_input_heat, valid_pred_heat]
                        #)

                           
                        val_images, val_labels = sess.run([valid_input_image_batch, valid_input_heat_batch])
                        val_feed_dict = {
                           input_image: val_images,
                           input_heat: val_labels
                        }
                        valid_loss_value, valid_p_heat, prediction, inputs = sess.run(
                           [valid_loss, valid_pred_heat, pred_summary, image_summary], feed_dict = val_feed_dict)

                        summary_writer.add_summary(prediction, step)
                        summary_writer.add_summary(inputs, step)
                        

                # print train info
                num_examples_per_step = params['batchsize'] * params['gpus']
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / params['gpus']
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch; %.3f sec/fetch_batch)')
                print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch, fetch_time))

                # tensorboard visualization
                #merge_op = sess.run(summary_merge_op)
                summary_writer.add_summary(merge_op, step)

                # save model
                if step % params['per_saved_model_step'] == 0:
                    checkpoint_path = os.path.join(params['modelpath'], training_name, 'model')
                    saver.save(sess, checkpoint_path, global_step=step)
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
