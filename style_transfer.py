import os
import time
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['axes.grid'] = False

import IPython.display

from PIL import Image

import tensorflow.contrib.eager as tfe
from tensorflow.python.keras.preprocessing import image as k_image
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

#Setup eager execution
tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))

#set image and content images, include path if not in same directory as this script
style_img = 'style.jpg'
content_img = 'content.jpg'

def load_img(img_path):
    max_size = 512
    img = Image.open(img_path)
    native_size = max(img.size)
    scale = max_size/native_size
    img = img.resize((round(img.size[0]*scale),round(img.size[1]*scale)),Image.ANTIALIAS)
    img = k_image.img_to_array(img)
    # tensorflow objects train when there is additional batch dimension, so add that 
    img = np.expand_dims(img,axis=0)
    
    return img

def imshow(img, title=None):
    #remove the batch dimension from img object
    img_disp = np.squeeze(img,axis=0)
    #normalize for display
    img_disp = img_disp.astype('uint8')
    plt.imshow(img_disp)
    if title is not None:
        plt.title(title)
    plt.imshow(img_disp)
    
plt.figure(figsize=(10,10))
content = load_img(content_img)
style = load_img(style_img)
plt.subplot(1,2,1)
imshow(content,'Content Image')
plt.subplot(1,2,2)    
imshow(style,'Style Image')
plt.show()    
    
# VGG networks preprocess inputs, e.g. they normalize by channels with 
#mean [103.939, 116.779, 123.68] for channel BGR (Blue Green Red)
def load_and_process_img(raw_img):
  img = load_img(raw_img)
  img_processed = tf.keras.applications.vgg19.preprocess_input(img)
  return img_processed

#to view images, we need to deprocess
def deprocess_img(img_processed):
    x = img_processed.copy()
    #squeeze batch dimension
    if len(x.shape) == 4:
        x = np.squeeze(x,axis=0)
    assert len(x.shape)==3
    #inverting preprocessing
    x[:,:,0] += 103.939
    x[:,:,1] += 116.779
    x[:,:,2] += 123.68
    x = x[:, :, ::-1]
    
    x =np.clip(x,0,255).astype('uint8')
    return x

#Since we are using pre-trained model from Keras, we need to specify which layers
#we are going to access in our model. I follow Gatys et al.

# Content layer where will pull our feature maps
content_layers = ['block5_conv2'] 

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)
    
def create_model():
    #load Keras model for vgg19 pre-trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False,weights='imagenet')
    vgg.trainable = False
    #Get the outputs for required layers
    # Get output layers corresponding to style and content layers 
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    # Build model 
    return models.Model(vgg.input, model_outputs)

#Define losses
def get_content_loss(curr_content, target_content):
    return tf.reduce_mean(tf.square(curr_content-target_content))

def gram_matrix(inp_tensor):
    #make image channels    
    channels = int(inp_tensor.shape[-1])
    a = tf.reshape(inp_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def get_style_loss(curr_style,target_style):
    """ Needs image of size h,w,c"""
    height,width,channels = curr_style.get_shape().as_list()
    normalization = 4.*(channels**2)*(width*height)**2
    gram_style = gram_matrix(curr_style)
    
    return tf.reduce_mean(tf.square(gram_style - target_style))#/normalization



def get_feature_representations(model, content_img, style_img):
    """Helper function to compute our content and style feature representations.

    This function will simply load and preprocess both the content and style 
    images from their path. Then it will feed them through the network to obtain
    the outputs of the intermediate layers. 
  
    Arguments:
      model: The model that we are using.
      content_path: The path to the content image.
      style_path: The path to the style image
      
    Returns:
      returns the style features and the content features. 
    """
    # Load our images in 
    content = load_and_process_img(content_img)
    style = load_and_process_img(style_img)
    
    # batch compute content and style features
    style_outputs = model(style)
    content_outputs = model(content)
    
    # Get the style and content feature representations from our model  
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    
    return style_features, content_features

def compute_loss(model, loss_weights, init_img, gram_style_features, content_features):
    """This function will compute the loss total loss.
    
    Arguments:
      model: The model that will give us access to the intermediate layers
      loss_weights: The weights of each contribution of each loss function. 
        (style weight, content weight, and total variation weight)
      init_img: Our initial base image. This image is what we are updating with 
        our optimization process. We apply the gradients wrt the loss we are 
        calculating to this image.
      gram_style_features: Precomputed gram matrices corresponding to the 
      defined style layers of interest.
      content_features: Precomputed outputs from defined content layers of 
        interest.
      
    Returns:
      returns the total loss, style loss, content loss, and total variational loss
  """
    style_weight, content_weight = loss_weights
    
    # Feed our init image through our model. This will give us the content and 
    # style representations at our desired layers. Since we're using eager
    # our model is callable just like any other function!
    model_outputs = model(init_img)
    
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]
    
    style_loss = 0
    content_loss = 0
  
    # Accumulate style losses from all layers
    # Here, we equally weight each contribution of each loss layer
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_loss += weight_per_style_layer * get_style_loss(comb_style[0], target_style)
    
    # Accumulate content losses from all layers 
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_loss += weight_per_content_layer* get_content_loss(comb_content[0], target_content)
  
    style_loss *= style_weight
    content_loss *= content_weight
  
    # Get total loss
    loss = style_loss + content_loss
    return loss, style_loss, content_loss

def compute_grads(cfg):
    #cfg is a dictionary with all required inputs, created in func perform_style_transfer
    with tf.GradientTape() as tape:
        all_loss =  compute_loss(**cfg)
        #compute gradients wrto inout image
        total_loss = all_loss[0]
    
    return tape.gradient(total_loss,cfg['init_img']), all_loss
    
def perform_style_transfer(content_img,style_img,num_iterations = 1000,content_weight = 1000,
                           style_weight=0.01):
    model = create_model()
    #we are nto training the layers of vgg19, so set trainable= false
    for layer in model.layers:
        layer.trainable = False
    
    #Get content and style features
    style_features,content_features = get_feature_representations(model,content_img,style_img)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
    
    #set initial image
    init_img = load_and_process_img(content_img)
    init_img = tfe.Variable(init_img,dtype=tf.float32)
    
    #create optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=0.1)
    
    #store the best result
    best_loss, best_img = float('inf'), None
    
    #create the config
    loss_weights = (style_weight,content_weight)
    cfg = {
        'model':model,
        'loss_weights':loss_weights,
        'init_img': init_img,
        'gram_style_features': gram_style_features,
        'content_features': content_features        
            }
    
    #For display
    num_rows = 2
    num_cols = 5
    display_interval = num_iterations/(num_rows*num_cols)
    start_time = time.time()
    global_start_time = time.time()
    
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means
    
    imgs = []
    
    for i in range(num_iterations):
        grads,all_loss = compute_grads(cfg)
        loss,style_loss,content_loss = all_loss
        optimizer.apply_gradients([(grads, init_img)])
        #clip the image values
        clipped = []
        for ii in range(3):
            clipped.append(tf.clip_by_value(init_img[:,:,:,ii], min_vals[ii], max_vals[ii]))
        
        init_img.assign(tf.stack(clipped,axis=-1))
        
        if loss < best_loss:
            #Update best loss and best image
            best_loss = loss
            #.numpy() gives concrete array
            best_img = deprocess_img(init_img.numpy())
            
        if i % display_interval==0:
            start_time = time.time()
            plot_img = init_img.numpy()
            plot_img = deprocess_img(plot_img)
            imgs.append(plot_img)
            IPython.display.clear_output(wait = True)
            IPython.display.display_png(Image.fromarray(plot_img))
            print('Iteration: {}'.format(i))
            print('Total Loss: {:.4e},'
                  'Style Loss: {:.4e},'
                  'Content Loss: {:.4e},'
                  'Time: {:.4f}s'.format(loss,style_loss,content_loss,time.time()-start_time))
        
    print('Total Time: {:.4f}s'.format(time.time()-global_start_time))
    IPython.display.clear_output(wait=True)
    plt.figure(figsize=(14,4))
    
    for i,img in enumerate(imgs):
        plt.subplot(num_rows,num_cols,i+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        
    return best_img, best_loss
    
best_img,best_loss = perform_style_transfer(content_img,style_img,num_iterations=100)
Image.fromarray(best_img)

def show_results(best_img,content_img,style_img):
    plt.figure(figsize=(10,5))
    content = load_img(content_img)
    style = load_img(style_img)
    
    plt.subplot(1, 2, 1)
    imshow(content,'Content Image')
    
    plt.subplot(1, 2, 2)
    imshow(style,'Style Image')
    
    plt.figure(figsize=(10,10))
    plt.imshow(best_img)
    plt.title('Output Image')
    plt.show()

show_results(best_img,content_img,style_img)    
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    