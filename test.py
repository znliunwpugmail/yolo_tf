import tensorflow as tf
import yolo_net
import pascol_read

cell_size = 7
image_size = 448
num_classes = 20
num_boxes_percell = 2

inputs = pascol_read.pascol_process_images(2)
net = yolo_net.bulid_network(inputs)

predict_class = net[:,:,:,:num_classes]
predict_box_confidence_percell = net[:,:,:,num_classes:num_classes+num_boxes_percell]
predict_box_coord = net[:,:,:,num_classes+num_boxes_percell:]
