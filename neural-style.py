#!/usr/bin/env python
from operator import add

from scipy.misc import imread, imresize, imsave
from tensorflow.contrib.opt import ScipyOptimizerInterface
import h5py
import numpy as np
import tensorflow as tf


class NeuralStyle:

    # https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
    VGG19_MEAN_BGR = (103.939, 116.779, 123.68)
    VGG19_NO_FC = (
        ('conv', 'conv1_1', (3, 3, 3, 64)),
        ('conv', 'conv1_2', (3, 3, 64, 64)),
        ('pool', 'pool1',   (1, 2, 2, 1)),
        ('conv', 'conv2_1', (3, 3, 64, 128)),
        ('conv', 'conv2_2', (3, 3, 128, 128)),
        ('pool', 'pool2',   (1, 2, 2, 1)),
        ('conv', 'conv3_1', (3, 3, 128, 256)),
        ('conv', 'conv3_2', (3, 3, 256, 256)),
        ('conv', 'conv3_3', (3, 3, 256, 256)),
        ('conv', 'conv3_4', (3, 3, 256, 256)),
        ('pool', 'pool3',   (1, 2, 2, 1)),
        ('conv', 'conv4_1', (3, 3, 256, 512)),
        ('conv', 'conv4_2', (3, 3, 512, 512)),
        ('conv', 'conv4_3', (3, 3, 512, 512)),
        ('conv', 'conv4_4', (3, 3, 512, 512)),
        ('pool', 'pool4',   (1, 2, 2, 1)),
        ('conv', 'conv5_1', (3, 3, 512, 512)),
        ('conv', 'conv5_2', (3, 3, 512, 512)),
        ('conv', 'conv5_3', (3, 3, 512, 512)),
        ('conv', 'conv5_4', (3, 3, 512, 512)),
        ('pool', 'pool5',   (1, 2, 2, 1))
    )


    def __init__(self, content, style, output,
            style_layers=('conv1_1','conv2_1','conv3_1','conv4_1','conv5_1'),
            content_layers=('conv4_2',),
            style_weight=1e2,
            content_weight=5e0,
            vgg_model_file='model/vgg19_weights.h5',
            maxsize=500,
            maxiter=500):
        self._style_layers = style_layers
        self._content_layers = content_layers
        self._style_weight = style_weight
        self._content_weight = content_weight
        self._vgg_model_file = vgg_model_file
        self._content, self._style, self._output = content, style, output
        self._maxsize = maxsize
        self._maxiter = maxiter
        self._nodes = {}


    def run(self):
        content = self._load_image(self._content)
        style = self._load_image(self._style)
        image = tf.Variable(style, dtype=tf.float32, validate_shape=False, name='image')
        self._build_vgg19(image)
        self._add_gramians()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            with tf.name_scope('losses'):
                style_losses = self._setup_style_losses(sess, image, style)
                content_losses = self._setup_content_losses(sess, image, content)
                loss = tf.foldl(add, content_losses+style_losses, name='loss')
            image.set_shape(content.shape)
            sess.run(tf.assign(image, content, validate_shape=False))
            opt = ScipyOptimizerInterface(loss,
                options={'maxiter': self._maxiter, 'disp': 1},
                method='L-BFGS-B')
            opt.minimize(sess)
            self._save_image(self._output, sess.run(image))


    def _build_vgg19(self, prev):
        weights = KerasVGG19(self._vgg_model_file)
        vgg_trunc = NeuralStyle.VGG19_NO_FC[0:self._vgg_last_useful_layer()+1]
        with tf.name_scope('vgg19_truncated'):
            for layer_type, name, shape in vgg_trunc:
                if layer_type == 'conv':
                    prev = self._conv(prev, name, shape, weights)
                elif layer_type == 'pool':
                    prev = self._pool(prev, name, shape)
                else:
                    raise ValueError('Unknown layer: %s' % layer_type)


    def _add_gramians(self):
        with tf.name_scope('gramians'):
            for style_layer_name in self._style_layers:
                name = 'gramian_' + style_layer_name
                node = self._nodes[style_layer_name]
                with tf.name_scope('gramian'):
                    shape = tf.shape(node) # NxHxWxC
                    h,w = shape[1]*shape[2], shape[3]
                    flat = tf.reshape(node, tf.stack([h,w]))
                    gramian = tf.matmul(flat, flat, transpose_a=True)
                    norm_gramian = gramian / tf.cast(h*w, dtype=tf.float32)
                self._nodes[name] = norm_gramian


    def _vgg_last_useful_layer(self):
        useful_layers = self._style_layers + self._content_layers
        vgg_layers = [name for _, name, _ in NeuralStyle.VGG19_NO_FC]
        return max(vgg_layers.index(layer) for layer in useful_layers)


    def _setup_style_losses(self, sess, image, image_data):
        losses = []
        with tf.name_scope('style_losses'):
            gramians = [self._nodes['gramian_'+l] for l in self._style_layers]
            activations = sess.run(gramians, {image: image_data})
            losses = []
            for i, l in enumerate(self._style_layers):
                prediction = gramians[i]
                target = tf.constant(activations[i], name='const_gramian_'+l)
                loss = tf.reduce_mean(tf.squared_difference(prediction, target))
                loss = tf.multiply(loss, self._style_weight)
                losses.append(loss)
        return losses


    def _setup_content_losses(self, sess, image, image_data):
        losses = []
        with tf.name_scope('content_losses'):
            layers = [self._nodes[l] for l in self._content_layers]
            activations = sess.run(layers, {image: image_data})
            for i, l in enumerate(self._content_layers):
                prediction = self._nodes[l]
                target = tf.constant(activations[i], name='const_'+l)
                loss = tf.reduce_mean(tf.squared_difference(prediction, target))
                loss = tf.multiply(loss, self._content_weight)
                losses.append(loss)
        return losses


    def _conv(self, prev, name, shape, weights):
        with tf.name_scope(name):
            W = tf.Variable(weights[name+'_W'], trainable=False, name='weights')
            b = tf.Variable(weights[name+'_b'], trainable=False, name='biases')
            conv2d = tf.nn.conv2d(prev, W, (1,1,1,1), padding='SAME')
            output = tf.nn.relu(tf.nn.bias_add(conv2d, b), name=name)
        self._nodes[name] = output
        return output


    def _pool(self, prev, name, shape):
        with tf.name_scope(name):
            output = tf.nn.max_pool(prev,
                ksize=shape, strides=shape, padding='SAME', name=name)
        self._nodes[name] = output
        return output


    def _load_image(self, filename):
        img = imread(filename, mode='RGB')
        (h, w, _), m = img.shape, self._maxsize
        if max(img.shape) > m:
            h, w = (m, round(m*w/h)) if h > w else (round(m*h/w), m)
            img = imresize(img, (h, w))
        img = img.astype(np.float32)
        img = np.flip(img, axis=2) # RGB --> BGR
        img -= NeuralStyle.VGG19_MEAN_BGR
        return img[np.newaxis,:]  # conv layers expect 4D-tensors


    def _save_image(self, filename, img):
        img = img[0,:]
        img += NeuralStyle.VGG19_MEAN_BGR
        img = np.flip(img, axis=2)
        img = np.clip(img, 0, 255)
        imsave(filename, img.astype(np.uint8))


    def _log(self, *args):
        print(*args)


class KerasVGG19:


    def __init__(self, filename):
        self._f = h5py.File(filename, 'r')


    def __getitem__(self, name):
        block, name = self._map_name(name)
        return np.array(self._f[block][name])


    def _map_name(self, name):
        "conv1_1_W --> ('block1_conv1', 'block1_conv1_W1:0')"
        block, idx, _type = name.split('_')
        block_idx = block[4:]
        return (
            'block%s_conv%s' % (block_idx, idx),
            'block%s_conv%s_%s_1:0' % (block_idx, idx, _type))


if __name__ == '__main__':
    neural_style = NeuralStyle('content.jpg', 'style.jpg', 'output.jpg',
        maxsize=300, maxiter=100)
    neural_style.run()
