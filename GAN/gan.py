import tensorflow_hub as hub
import tensorflow as tf 
import numpy as np 
import argparse
import cv2

MODEL = hub.load('https://tfhub.dev/google/magenta/arbitary-image-stylization-v1-256/2')

def load_img(img_path):
    """Load, decode and convert images to correct format."""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(image, tf.float32)
    return img

def parse_options():
    """Parse command line options."""
    parser = argparse.ArgumentParser(description=("This is the CLI for the run_gan module"),
                                     epilog="Ethan Jones, 2021-08-31")
    parser.add_argument("-c", "--content-image", dest="content_img", action="store", type=str,
                        requried=True)
    parser.add_argument("-s", "--style-image", dest="style_img", action="store", type=str,
                        requried=True)
    options = parser.parse_args()
    return options
                                

def main(content_img_path, style_img_path):
    content_image = load_img(content_img_path)
    style_image = load_img(style_img_path)
    stylised_image = MODEL(tf.constant(content_image), tf.constant(style_image))
    cv2.imwrite("generated_image.jpg", cv2.cvtColor(np.squeeze(stylised_image)*255, cv2.COLOR_BGR2RGB))

if __name__ == main:
    options = parse_options()
    main(options.content_img, options.style_img)