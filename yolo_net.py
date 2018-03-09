import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
cell_size = 7
num_outputs=7*7*30
image_size = 448
num_classes = 20
num_boxes_percell = 2
lambda_coord = 5
lambda_noobj = 0.5
def bulid_network(inputs):
    with slim.arg_scope([slim.conv2d,slim.fully_connected],activation_fn=tf.nn.relu):
        with slim.arg_scope([slim.max_pool2d],padding='SAME',stride=2):
            net = slim.conv2d(inputs,64,[7,7],stride=2,scope='conv_1')
            net = slim.max_pool2d(net,2,scope='pool_1')

            net = slim.conv2d(net,192,[3,3],scope='conv_2')
            net = slim.max_pool2d(net,[2,2],scope='pool_2')

            net = slim.conv2d(net,128,[1,1],scope='conv_3')
            net = slim.conv2d(net,256,[3,3],scope='conv_4')
            net = slim.conv2d(net,256,[1,1],scope='conv_5')
            net = slim.conv2d(net,512,[3,3],scope='conv_6')
            net = slim.max_pool2d(net,[2,2],scope='pool_3')

            for i in range(4):
                net = slim.conv2d(net,256,[1,1],scope='conv_'+str(i*2+7))
                net = slim.conv2d(net,512,[3,3],scope='conv_'+str(i*2+8))
            net = slim.conv2d(net,512,[1,1],scope='conv_15')
            net = slim.conv2d(net,1024,[3,3],scope='conv_16')
            net = slim.max_pool2d(net,[2,2],scope='pool_4')

            for i in range(2):
                net = slim.conv2d(net,512,[1,1],scope='conv_'+str(i*2+17))
                net = slim.conv2d(net,1024,[3,3],scope='conv_'+str(i*2+18))
            net = slim.conv2d(net,1024,[3,3],scope='conv_21')
            net = slim.conv2d(net,1024,[3,3],stride=2,scope='conv_22')

            net = slim.conv2d(net,1024,[3,3],scope='conv_23')
            net = slim.conv2d(net,1024,[3,3],scope='conv_24')
            print(net)
            net = slim.flatten(net,scope='flatten_net')
            print(net)
            net = slim.fully_connected(net,512,scope='fc_1')
            net = slim.fully_connected(net,4096,scope='fc_2')
            net = slim.dropout(net,0.5,is_training=True,scope='dropout')
            net = slim.fully_connected(net,num_outputs,activation_fn=None,scope='logit')
            return net

def iou_calculate(box1,box2):
    boxes1 = tf.zeros_like(box1)
    boxes2 = tf.zeros_like(box2)
    boxes1[:,:,:,:,0] = box1[:,:,:,:,0]-box1[:,:,:,:,2]/2
    boxes1[:, :, :, :, 1] = box1[:, :, :, :, 0] + box1[:, :, :, :, 2] / 2
    boxes1[:, :, :, :, 2] = box1[:, :, :, :, 1] - box1[:, :, :, :, 3] / 2
    boxes1[:, :, :, :, 3] = box1[:, :, :, :, 1] + box1[:, :, :, :, 3] / 2

    boxes2[:, :, :, :, 0] = box2[:, :, :, :, 0] - box2[:, :, :, :, 2] / 2
    boxes2[:, :, :, :, 1] = box2[:, :, :, :, 0] + box2[:, :, :, :, 2] / 2
    boxes2[:, :, :, :, 2] = box2[:, :, :, :, 1] - box2[:, :, :, :, 3] / 2
    boxes2[:, :, :, :, 3] = box2[:, :, :, :, 1] + box2[:, :, :, :, 3] / 2

    lu = tf.maximum(boxes1[:,:,:,:,0:2],boxes2[:,:,:,:,0:2])
    rd = tf.minimum(boxes1[:,:,:,:,2:],boxes2[:,:,:,:,2:])

    Iou = tf.maximum(0,rd-lu)
    square_Iou = Iou[:,:,:,:,0]*Iou[:,:,:,:,1]
    square_box1 = box1[:,:,:,:,2]*box1[:,:,:,:,3]
    square_box2 = box2[:, :, :, :,2] * box2[:, :, :, :, 3]
    square_uniom = square_box1+square_box2-square_Iou
    return tf.clip_by_value(square_Iou/square_uniom,0.0,0.1)

def loss_net(net,labels):
    predict_class = net[:,:,:,:num_classes]
    predict_box_confidence_percell = net[:,:,:,num_classes:num_classes+num_boxes_percell]
    predict_box_coord = net[:,:,:,num_classes+num_boxes_percell:]
    predict_box_coord = tf.reshape(predict_box_coord,
                                   [batch_size,cell_size,cell_size,num_boxes_percell,4])

    gt_class = labels[:,:,:,5:]
    gt_box_confidence = labels[:,:,:,0]
    tf.reshape(gt_box_confidence,[batch_size,cell_size,cell_size,1,1])
    gt_box_confidence = tf.tile(gt_box_confidence,[1,1,1,2,1])
    gt_box_coord = labels[:,:,:,1:5]
    gt_box_coord = tf.reshape(gt_box_coord,[batch_size,cell_size,cell_size,1,4])
    gt_box_coord = tf.tile(gt_box_coord,[1,1,1,2,1])

    Iou = iou_calculate(predict_box_coord,gt_box_coord)
    obj_mask = tf.reduce_max(Iou,axis=3,keep_dims=True)
    obj_mask = tf.cast(Iou>obj_mask,tf.float32)*gt_box_confidence #l_{ij}^{obj}

    coord_loss1 = gt_box_coord[:, :, :, :, :2] - predict_box_coord[:, :, :, :, :2]
    coord_loss1 = obj_mask * tf.reduce_sum(tf.square(coord_loss1),axis=4)
    coord_loss1 = lambda_coord*tf.reduce_mean(coord_loss1,[1,2,3])
    gt_box_coord[:,:,:,:,2:] = tf.sqrt(gt_box_coord[:,:,:,:,2:])
    predict_box_coord[:,:,:,:,2:] = tf.sqrt(predict_box_coord[:,:,:,:,2:])
    coord_loss2 = gt_box_coord[:,:,:,:,2:]-predict_box_coord[:,:,:,:,2:]
    coord_loss2 = obj_mask*tf.reduce_sum(tf.square(coord_loss2),axis=4)
    coord_loss2 = lambda_coord*tf.reduce_mean(coord_loss2,[1,2,3])
    coord_loss = coord_loss1+coord_loss2

    noobj_mask = tf.ones_like(Iou) - obj_mask

    confidence_loss1 = obj_mask*tf.square(
        gt_box_confidence-predict_box_confidence_percell)
    confidence_loss1 = tf.reduce_mean(confidence_loss1,[1,2,3,4])
    confidence_loss2 = noobj_mask*tf.square(
        gt_box_confidence-predict_box_confidence_percell)
    confidence_loss2 = lambda_noobj*tf.reduce_mean(confidence_loss2,[1,2,3,4])
    confidence_loss = confidence_loss1+confidence_loss2

    class_loss = tf.reduce_sum(tf.square(gt_class-predict_class),axis=4)
    class_loss = tf.reduce_mean(obj_mask*class_loss,[1,2])

    total_loss = coord_loss+confidence_loss+class_loss
    return total_loss

def test_slim(inputs):
    with slim.arg_scope([slim.conv2d],padding='SAME'):
        net = slim.conv2d(inputs,64,[7,7],stride = 2,scope='conv1')
        return net

if __name__ == '__main__':
    batch_size = 4
    inputs = tf.random_uniform((batch_size,448,448,3))

    net = bulid_network(inputs)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    test_net = sess.run(net)
    # print(test_net)
    print(test_net.shape)

