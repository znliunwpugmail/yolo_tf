import xml.etree.ElementTree as ET
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
import cv2
objs_class = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
class_to_id = {}
image_size = 448
for i in range(len(objs_class)):
    class_to_id[objs_class[i]]=i
def pascol_process_labels(batch_size,cell_size,nums_boxes_per_cell):
    labels = np.zeros([batch_size, cell_size, cell_size, 5 + len(objs_class)])
    for count in range(batch_size):
        tree = ET.parse('000005.xml')
        im = read_images('000005.jpg')
        w_ratio = image_size/im.shape[0]
        h_ratio = image_size/im.shape[1]
        # root = tree.getroot()
        objs = tree.findall('object')
        # root = ET.fromstring('object')
        for obj in objs:
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin=int(float(bndbox.find('xmin').text)*w_ratio)
            ymin=int(float(bndbox.find('ymin').text)*h_ratio)
            xmax=int(float(bndbox.find('xmax').text)*w_ratio)
            ymax=int(float(bndbox.find('ymax').text)*h_ratio)
            x = (xmin+xmax)//2
            y = (ymin+ymax)//2
            w = max(0,xmax-xmin)
            h = max(0,ymax-ymin)
            if labels[count][x//(image_size//cell_size)][y//(image_size//cell_size)][0]==1:
                print('continue')
                continue
            labels[count][x // (image_size // cell_size)][y // (image_size // cell_size)][0]=1
            labels[count][x // (image_size // cell_size)][y // (image_size // cell_size)][1:5]=[x,y,w,h]
            labels[count][x // (image_size // cell_size)][y // (image_size // cell_size)][class_to_id[name]+5]=1
    return labels

def read_images(image_name):
    image = cv2.imread(image_name)
    return image

def pascol_process_images(batch_size):
    images = np.zeros([batch_size,image_size,image_size,3])
    for count in range(batch_size):
        image_name='000005.jpg'
        im = read_images(image_name)
        im = cv2.resize(im,(image_size,image_size))
        images[count]=im
    return images

if __name__ == '__main__':
    batch_size = 1;
    cell_size=7;
    nums_boxes_per_cell=2;
    labels = pascol_process_labels(batch_size,cell_size,nums_boxes_per_cell)
    # print(labels)
    # print(labels[:,:,:,0]>0)
    images = pascol_process_images(batch_size)
    for count in range(batch_size):
        # cv2.imshow(str(count),images[count])
        cv2.imwrite(str(count)+'.png',images[count])
        print(images[count].shape)

        # cv2.waitKey()