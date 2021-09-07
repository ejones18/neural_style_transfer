import tensorflow_hub as hub
import tensorflow as tf 
import numpy as np 
import argparse
import matplotlib.pyplot as plt
import cv2

def parse_options():
    """Parse command line options."""
    parser = argparse.ArgumentParser(description=("This is the CLI for the run_gan module"),
                                     epilog="Ethan Jones, 2021-08-31")
    parser.add_argument("-c", "--content-image", dest="content_img", action="store", type=str,
                        required=True)
    parser.add_argument("-s", "--style-image", dest="style_img", action="store", type=str,
                        required=True)
    options = parser.parse_args()
    return options
    
def main(content_img_path, style_img_path):
    # Load content and style images (see example in the attached colab).
    content_image = plt.imread(content_img_path)
    style_image = plt.imread(style_img_path)
    # Convert to float32 numpy array, add batch dimension, and normalize to range [0, 1]. Example using numpy:
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
    # Optionally resize the images. It is recommended that the style image is about
    # 256 pixels (this size was used when training the style transfer network).
    # The content image can be any size.
    #style_image = tf.image.resize(style_image, (256, 256))

    # Load image stylization module.
    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    # Stylize image.
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylised_image = outputs[0]
    cv2.imwrite("./images/generated_image.jpg", cv2.cvtColor(np.squeeze(stylised_image)*255,
                                                    cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    options = parse_options()
    main(options.content_img, options.style_img)