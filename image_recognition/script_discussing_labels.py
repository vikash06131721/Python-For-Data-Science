import pandas as pd
import tensorflow as tf
#The code is for understanding the labels available in inception and how to get the correct labels for all the 1000 classes


def lookup():
    #label file declaring from uid to name for human labels
    path_uid_to_name = "imagenet_synset_to_human_label_map.txt"
    #lets declare a dict from uid-to-name mapping
    uid_to_name_dict={}

    path_uid_to_name_joined= '../model/'+path_uid_to_name
    with open(path_uid_to_name_joined,mode='r') as file:
        #Read all lines from the file
        lines= file.readlines()

        for line in lines:
            #remove new lines
            line= line.replace("\n","")

            #split the line on tabs
            elements= line.split("\t")

            #get the uid
            uid= elements[0]

            #get the class name:
            name= elements[1]

            #insert in the dict:
            uid_to_name_dict[uid]= name

    #uid_to_class
    path_uid_to_class='imagenet_2012_challenge_label_map_proto.pbtxt'
    uid_to_class={}
    cls_to_uid={}
    path_uid_to_class_joined= '../model/' + path_uid_to_class

    with open(path_uid_to_class_joined,mode='r') as file:
        #read all the lines
        lines1= file.readlines()

        for line in lines1:
            if line.startswith("  target_class: "):
                #split the line
                elements= line.split(": ")
                cls = int(elements[1])
            elif line.startswith("  target_class_string: "):
                elements= line.split(": ")
                #get the uid
                uid= elements[1]

                #remove the enclosing "" from the string
                uid= uid[1:-2]

                uid_to_class[uid]= cls
                cls_to_uid[cls]= uid
    return uid_to_name_dict, uid_to_class, cls_to_uid

#Get the final class name.
def cls_to_name(cls,cls_to_uid,uid_to_name_dict):
    #get the uid
    uid= cls_to_uid[cls]
    name= uid_to_name_dict[uid]
    return name

# uid_to_name_dict, uid_to_class, cls_to_uid= lookup()
