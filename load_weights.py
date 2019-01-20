import numpy as np
import time
t1=time.time()
conv1_1_W = np.load('vgg16_weights/conv1_1_W.npy')
conv1_1_b = np.load('vgg16_weights/conv1_1_b.npy')
conv1_2_W = np.load('vgg16_weights/conv1_2_W.npy')
conv1_2_b = np.load('vgg16_weights/conv1_2_b.npy')

conv2_1_W = np.load('vgg16_weights/conv2_1_W.npy')
conv2_1_b = np.load('vgg16_weights/conv2_1_b.npy')
conv2_2_W = np.load('vgg16_weights/conv2_2_W.npy')
conv2_2_b = np.load('vgg16_weights/conv2_2_b.npy')

conv3_1_W = np.load('vgg16_weights/conv3_1_W.npy')
conv3_1_b = np.load('vgg16_weights/conv3_1_b.npy')
conv3_2_W = np.load('vgg16_weights/conv3_2_W.npy')
conv3_2_b = np.load('vgg16_weights/conv3_2_b.npy')
conv3_3_W = np.load('vgg16_weights/conv3_3_W.npy')

conv4_1_b = np.load('vgg16_weights/conv4_1_b.npy')
conv4_1_W = np.load('vgg16_weights/conv4_1_W.npy')
conv4_2_b = np.load('vgg16_weights/conv4_2_b.npy')
conv4_2_W = np.load('vgg16_weights/conv4_2_W.npy')
conv4_3_b = np.load('vgg16_weights/conv4_3_b.npy')
conv4_3_W = np.load('vgg16_weights/conv4_3_W.npy')

conv5_1_b = np.load('vgg16_weights/conv5_1_b.npy')
conv5_1_W = np.load('vgg16_weights/conv5_1_W.npy')
conv5_2_b = np.load('vgg16_weights/conv5_2_b.npy')
conv5_2_W = np.load('vgg16_weights/conv5_2_W.npy')
conv5_3_b = np.load('vgg16_weights/conv5_3_b.npy')
conv5_3_W = np.load('vgg16_weights/conv5_3_W.npy')

t2=time.time()

print('Shape of conv2_1_w: ',conv2_1_W.shape)


print(f"Time took to load: {t2-t1} seconds.")