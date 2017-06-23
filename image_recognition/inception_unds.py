#The code describes how to load an inception model initial phase, the code will be continuosly edited to produce image classification.

try:
    import pandas as  pd
    import numpy as np
    import tensorflow as tf
except ImportError:
    print ('One or many packages not installed')

#global variables
tensor_name_input_jpeg = "DecodeJpeg/contents:0"

# Name of the tensor for feeding the decoded input image.
# Use this for feeding images in other formats than jpeg.
tensor_name_input_image = "DecodeJpeg:0"

# Name of the tensor for the resized input image.
# This is used to retrieve the image after it has been resized.
tensor_name_resized_image = "ResizeBilinear:0"

# Name of the tensor for the output of the softmax-classifier.
# This is used for classifying images with the Inception model.
softmax = "softmax:0"

# Name of the tensor for the unscaled outputs of the softmax-classifier (aka. logits).
tensor_name_softmax_logits = "softmax/logits:0"

# Name of the tensor for the output of the Inception model.
# This is used for Transfer Learning.
tensor_name_transfer_layer = "pool_3:0"

# def lookup_clas_name():
#     path_uid_to_name = "imagenet_synset_to_human_label_map.txt"
#     path= '../model/'+path_uid_to_name
#     with open(file=path,'rb') as file:
#         #Read all lines from the file
#         lines= file.readlines()



#function for loading the inception model in graph and getting the important tensors
def load_inception():
    graph_name= 'classify_image_graph_def.pb'

    #declaring a tensorflow graph
    graph= tf.Graph()

    # Name of the tensor for feeding the input image as jpeg.



    with graph.as_default():
        path_graph= '../model/'+graph_name
        with tf.gfile.FastGFile(path_graph,'rb') as file:
            #create an empty graph

            graph_def= tf.GraphDef()
            #read the protobuf file
            graph_def.ParseFromString(file.read())
            #import the graph to tf graph and get the required tensors from the graph
            tensor_input_jpeg_1, tensor_name_input_image_1,tensor_name_resized_image_1,softmax_1, softmax_logits,transfer_layer= tf.import_graph_def(graph_def,name='inception',return_elements=[tensor_name_input_jpeg,
                                                                                tensor_name_input_image,
                                                                                tensor_name_resized_image,
                                                                                softmax, tensor_name_softmax_logits,tensor_name_transfer_layer
                                                                                ])

    print 'List down important functions associated to the graph:'
    print dir(graph)
    return tensor_input_jpeg_1, tensor_name_input_image_1,tensor_name_resized_image_1,softmax_1, softmax_logits,transfer_layer, graph

#Function to get the operation and the tensors from the graph
def get_the_ops_tensors_from_graph(GRAPH):
    #declare an open session:
    with tf.Session(graph=GRAPH) as sess:
        #get the operations
        names= [op.name for op in GRAPH.get_operations()]
        #get the tensors
        tensors_available= [op.values() for op in GRAPH.get_operations()]
    sess.close()
    return names, tensors_available


def create_feed_dict(image_path,tensor_input_jpeg_1):
    image_data= tf.gfile.FastGFile(image_path,'rb').read()
    feed_dict= {tensor_input_jpeg_1:image_data}
    return feed_dict








if __name__ == '__main__':
    input_tensor_jpeg, input_tensor_image, input_tensor_resized_image, softmax_tensor, softmax_logits_tensor, transfer_layer_tensor,graph= load_inception()
    names_ , tensors_= get_the_ops_tensors_from_graph(graph)
    feed_dict_image= create_feed_dict('/Users/vprasad/Desktop/cropped_panda.jpg',input_tensor_jpeg)
    with tf.Session(graph=graph) as sess:
        y_=sess.run(softmax_logits_tensor,feed_dict=feed_dict_image)
