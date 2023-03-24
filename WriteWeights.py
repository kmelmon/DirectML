import numpy

import torch
import torch.nn as nn

def WriteTensor(file, tensor, name):
    print(tensor)
    nparray = tensor.detach().numpy()
    shape = nparray.shape
    if (len(shape) == 1):
        declaration = "float {0}[{1}] = \n".format(name, shape[0])
        file.write(declaration)
        file.write("{")
        for x in nparray:
            value = "{:f}f".format(x)
            file.write(value)
            file.write(",")
        file.write("};\n")
    else:
        declaration = "float {0}[{1}][{2}][{3}][{4}] = \n".format(name, shape[0], shape[1], shape[2], shape[3])
        file.write(declaration)
        file.write("{\n")
        for x in nparray:
            file.write("\t{\n")
            for y in x:
                file.write("\t\t{\n")
                for z in y:
                    file.write("\t\t\t{")
                    for w in z:
                        value = "{:f}f".format(w)
                        file.write(value)
                        file.write(",")
                    file.write("},\n")
                file.write("\t\t},\n")
            file.write("\t},\n")
        file.write("};\n")

some_model = torch.load('D:/DirectML3/RDNN_snapshot_6.pth', map_location='cpu')

tensors = [['conv1.weight', 'conv1_weights'],
           ['conv1.bias', 'conv1_biases'],
           ['RDB1.conv1.weight', 'rdb1_conv1_weights'],
           ['RDB1.conv1.bias', 'rdb1_conv1_biases'],
           ['RDB1.conv2.weight', 'rdb1_conv2_weights'],
           ['RDB1.conv2.bias', 'rdb1_conv2_biases'],
           ['RDB1.conv3.weight', 'rdb1_conv3_weights'],
           ['RDB1.conv3.bias', 'rdb1_conv3_biases'],
           ['RDB1.conv4.weight', 'rdb1_conv4_weights'],
           ['RDB1.conv4.bias', 'rdb1_conv4_biases'],
           ['RDB2.conv1.weight', 'rdb2_conv1_weights'],
           ['RDB2.conv1.bias', 'rdb2_conv1_biases'],
           ['RDB2.conv2.weight', 'rdb2_conv2_weights'],
           ['RDB2.conv2.bias', 'rdb2_conv2_biases'],
           ['RDB2.conv3.weight', 'rdb2_conv3_weights'],
           ['RDB2.conv3.bias', 'rdb2_conv3_biases'],
           ['RDB2.conv4.weight', 'rdb2_conv4_weights'],
           ['RDB2.conv4.bias', 'rdb2_conv4_biases'],
           ['conv_final.weight', 'conv_final_weights'],
           ['conv_final.bias', 'conv_final_biases'],
           ]

file = open(r"testme.h","w")
for tensor in tensors:
    WriteTensor(file, some_model[tensor[0]], tensor[1])
file.close()

