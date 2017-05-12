import tensorflow as tf
from image_transform_network import *
from glob import glob
from vgg16 import vgg16
import cv2
import os
import numpy as np
import time
import datetime

FLAGS = tf.app.flags.FLAGS

def compute_gram(features):
    gram_list = []
    for feature in features:
        shape = tf.shape(feature)
        psi = tf.reshape(feature, [shape[0], shape[1] * shape[2], shape[3]])
        #psi_t = tf.transpose(psi, perm=[0, 2, 1])
        gram = tf.matmul(psi, psi, transpose_a = True)
        gram = tf.div(gram, tf.cast(shape[1] * shape[2] * shape[3], tf.float32))
        gram_list.append( gram )
    return gram_list

def train(argv=None):

    # compute gram loss from style image before building networks
    # use gram list as constant
    input_style_image = cv2.imread(FLAGS.style_image)
    input_style_image = cv2.cvtColor(input_style_image, cv2.COLOR_BGR2RGB)
    style_image = tf.placeholder(tf.float32, [1, input_style_image.shape[0], input_style_image.shape[1], 3])
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        vgg = vgg16('vgg16_weights.npz', sess)
        _, style_features = vgg.get_features(style_image)
        style_gram_list = compute_gram(style_features)
        target_gram_list = sess.run(style_gram_list, feed_dict = {style_image : [input_style_image]})


    tf.reset_default_graph()
    # build image transformation network
    input_image = tf.placeholder(tf.float32, [None, 256, 256, 3])
    net = TransformNetwork('image_transform_network')
    generated_image = net.inference(input_image)
    vgg = vgg16()
    generated_content_feature, generated_style_feature = vgg.get_features(generated_image)
    input_content_feature, _ = vgg.get_features(input_image)
    

    # feature reconstrution loss
    feature_shape = tf.shape(generated_content_feature)
    feature_size = tf.cast(feature_shape[1] * feature_shape[2] * feature_shape[3], dtype=tf.float32)
    feature_reconstruction_loss = tf.reduce_sum(tf.squared_difference(generated_content_feature, input_content_feature)) / feature_size

    
    # style reconstruction loss
    generated_style_gram_list = compute_gram(generated_style_feature)
    style_loss_list = []
    for i in range(0, len(generated_style_gram_list)): # num of layers
        shape = tf.shape(generated_style_gram_list[i])
        feature_size = tf.cast(shape[1] * shape[2], tf.float32)
        layer_style_loss = tf.reduce_sum((generated_style_gram_list[i] - tf.constant(target_gram_list[i])) ** 2) / feature_size
        #layer_style_loss = tf.norm(gen_G - style_G, ord='fro', axis=[-2, -1]) # I am not sure for frobenius norm
        style_loss_list.append(layer_style_loss)
    
    style_reconstruction_loss = tf.add_n(style_loss_list)
    #style_reconstruction_loss = tf.reduce_mean(style_loss_list)
    total_loss = FLAGS.alpha * feature_reconstruction_loss + (1 - FLAGS.alpha) * style_reconstruction_loss
    
    
    # only updates image transformation network 
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope='image_transform_network')

    train_optimizer = tf.train.AdamOptimizer(0.001).minimize(total_loss, var_list = train_vars)

    tf.summary.scalar('feature reconstruction loss', feature_reconstruction_loss)
    tf.summary.scalar('style reconstruction loss', style_reconstruction_loss)
    tf.summary.image('train image', tf.concat([input_image, generated_image], 2))


    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Create a saver.
    saver = tf.train.Saver(train_vars)

    # get train images
    train_image_batch, num_train_images = get_train_images()
    iteration = num_train_images / FLAGS.batch_size

    # open session
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess: 
        # initialize the variables
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        
        # load weight again because variables are all initialized
        vgg.load_weights('vgg16_weights.npz', sess) 

        # load pretrained model
        if FLAGS.load_pretrained_model:        
            point = tf.train.latest_checkpoint(FLAGS.summary_dir)
            print 'load last check point - ', point
            saver.restore(sess, point)        



        # write graph definition
        tf.train.write_graph(sess.graph_def, FLAGS.summary_dir, '%s_graph_def.pb' % (FLAGS.style_image.split('/')[-1].split('.')[0]))



        # summary
        summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

        # initialize the queue threads to start to shovel data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)


        count = 0
        for i in range(0, FLAGS.epoch):
            for j in range(0, iteration):
                
                # get train image batch
                train_images = sess.run(train_image_batch)
                
                # optimize network with batch
                sess.run(train_optimizer, feed_dict = {input_image : train_images})

                # write summary
                # add_summary() is computatinally expensive
                if count % 10 == 0:
                    _, output_summary, output_f_loss, output_s_loss = sess.run([train_optimizer, summary_op, feature_reconstruction_loss, style_reconstruction_loss], feed_dict = {input_image : train_images})
                
                    summary_writer.add_summary(output_summary, count)

                    ts = time.time()
                    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                    print '%s, epoch[%d] iter[%d] : feature loss - %f, style loss - %f' % (st, i, j, output_f_loss, output_s_loss)
                
                
                # save
                if count % 1000 == 0:
                    checkpoint_path = os.path.join(FLAGS.summary_dir, '%s.ckpt' % (FLAGS.style_image.split('/')[-1].split('.')[0]))
                    saver.save(sess, checkpoint_path, global_step=count, write_meta_graph = False)

                count += 1


def get_train_images():
    image_list = glob(FLAGS.train_path + '/*.jpg')
    print len(image_list)
    
    train_image_name = ops.convert_to_tensor(image_list, dtype=dtypes.string)
    train_input_queue = tf.train.slice_input_producer([train_image_name], shuffle = True)
    
    file_content = tf.read_file(train_input_queue[0])
    train_image = tf.image.decode_jpeg(file_content, channels=3)
    train_image = tf.cast(train_image, tf.float32)
    train_image = tf.image.resize_images(train_image, [FLAGS.train_size, FLAGS.train_size])
    train_image.set_shape([FLAGS.train_size, FLAGS.train_size, 3])

    # add normalization??
    min_after_dequeue = 100
    capacity = min_after_dequeue + 4 * FLAGS.batch_size
    train_image = tf.train.shuffle_batch(
        [train_image],
        batch_size=FLAGS.batch_size
        ,num_threads=4
        , capacity=capacity
        , min_after_dequeue=min_after_dequeue
    )

    return train_image, len(image_list)


if __name__ == '__main__':
    tf.app.flags.DEFINE_integer('batch_size', 4, """The batch size to use.""")
    tf.app.flags.DEFINE_float('alpha', 0.2, """weight for feature reconstruction loss.""")
    tf.app.flags.DEFINE_string('summary_dir', './summary', """summary directory.""")
    tf.app.flags.DEFINE_string('style_image', './style_images/starry_night.jpg', """target style image""")
    tf.app.flags.DEFINE_integer('train_size', 256, """image width and height""")
    tf.app.flags.DEFINE_string('train_path', './data/train2014', """path which contains train images""")
    tf.app.flags.DEFINE_integer('epoch', 2, """epoch""")
    tf.app.flags.DEFINE_bool('load_pretrained_model', False, "load pretrained model")    

    # clear summary directory
    log_files = glob(FLAGS.summary_dir + '/events*')
    for f in log_files:
        os.remove(f)

    tf.app.run(main=train)
