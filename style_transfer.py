import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['axes.grid'] = False

from PIL import Image

import tensorflow.contrib.eager as tfe
from tensorflow.python.keras.preprocessing import image as k_image
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

#Setup eager execution
#tf.enable_eager_execution()
#print("Eager execution: {}".format(tf.executing_eagerly()))

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
def deprocess(img_processed):
    x = img_processed.copy()
    #squeeze batch dimension
    if len(x.shape) == 4:
        x = np.squeeze(x,axis=0)
    assert len(x.shape)==3
    #inverting preprocessing
    x[:,:,0] += 103.939
    x[:,:,1] += 116.779
    x[:,:,2] += 123.68
    x = x[:,:,:,-1]
    
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
    
    return tf.reduce_mean(tf.square(gram_style - target_style))/normalization



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

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    """This function will compute the loss total loss.
    
    Arguments:
      model: The model that will give us access to the intermediate layers
      loss_weights: The weights of each contribution of each loss function. 
        (style weight, content weight, and total variation weight)
      init_image: Our initial base image. This image is what we are updating with 
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
    model_outputs = model(init_image)
    
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]
    
    style_score = 0
    content_score = 0
  
    # Accumulate style losses from all layers
    # Here, we equally weight each contribution of each loss layer
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)
    
    # Accumulate content losses from all layers 
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)
  
    style_score *= style_weight
    content_score *= content_weight
  
    # Get total loss
    loss = style_score + content_score 
    return loss, style_score, content_score
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    