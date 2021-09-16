"""
A python module that demonstrates the nerual style transfer capabilities of Google Magenta.
- Ethan Jones <ejones18@sheffield.ac.uk>
- First authored: 2021-08-31
"""
import argparse
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

def parse_options():
    """Parse command line options."""
    parser = argparse.ArgumentParser(description=("This is the CLI for the google_magenta_example module"),
                                     epilog="Ethan Jones, 2021-08-31")
    parser.add_argument("-c", "--content-image", dest="content_img", action="store", type=str,
                        required=True, metavar="</path/to/content-image>",
                        help="Specify path to content image file - including file extension.")
    parser.add_argument("-s", "--style-image", dest="style_img", action="store", type=str,
                        required=True, metavar="</path/to/style-image>",
                        help="Specify path to style image file - including file extension.")
    parser.add_argument("-o", "--output_file", dest="output_file_path", action="store", type=str,
                        required=True, metavar="</path/to/output_file>",
                        help="Specify path to output file - including .jpg extension.")
    options = parser.parse_args()
    return options

def main(content_img_path, style_img_path, output_file_path):
    """
    Load the content and style images and then restructure the content image based on the features
    of the style image using the Google Magenta model from tensorflowhub.
    """
    # Load content and style images
    content_image = plt.imread(content_img_path)
    style_image = plt.imread(style_img_path)
    # Convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
    # Load image stylisation module.
    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    # Stylise image.
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylised_image = outputs[0]
    # Save image.
    cv2.imwrite(output_file_path, cv2.cvtColor(np.squeeze(stylised_image)*255,
                                               cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    cli_options = parse_options()
    main(cli_options.content_img, cli_options.style_img, cli_options.output_file_path)
