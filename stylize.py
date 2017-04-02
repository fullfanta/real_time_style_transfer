import tensorflow as tf
import argparse
import cv2


def main():
    # load image
    input_image = cv2.imread(args.input_image, cv2.CV_LOAD_IMAGE_COLOR)
    input_image = cv2.resize(input_image, (input_image.shape[1] / args.resize_ratio, input_image.shape[0] / args.resize_ratio))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    print 'input image - ', args.input_image, input_image.shape

    input_tensor = tf.placeholder(tf.float32, [1, input_image.shape[0], input_image.shape[1], 3])

    # load model
    with tf.gfile.GFile(args.model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            generated_image, = tf.import_graph_def(
                    graph_def, 
                    input_map={'Placeholder' : input_tensor}, # connect intensity-normalized images to graph 
                    return_elements=['image_transform_network/deconv1/output:0'], 
                    name=None, 
                    op_dict=None, 
                    producer_op_list=None
                )

    # generate
    with tf.Session() as sess:
        result = sess.run(generated_image, feed_dict = {input_tensor : [input_image]})
        result = result[0]
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
 
        if args.output_image is None:
            idx = args.input_image.rfind('.')
            args.output_image = args.input_image[:idx] + '_output.jpg'
        cv2.imwrite(args.output_image, result)
        print 'output image - ', args.output_image


if __name__=='__main__':
    parser = argparse.ArgumentParser('Stylizer')
    parser.add_argument('--model', type=str, default='models/starry_night.pb')
    parser.add_argument('--input_image', type=str, default='./test_images/Aaron_Eckhart_0001.jpg')
    parser.add_argument('--output_image', type=str)
    parser.add_argument('--resize_ratio', type=int, default=1)
    
    args = parser.parse_args()  

    main()
