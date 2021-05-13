# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:13:44 2019

@author: P70002567
"""

#use json credential of private key

from google.cloud import vision
import os
import io
from google.oauth2 import service_account
import glob

my_path = "E:\\ML Learning\\google vision api"

credentials = service_account.Credentials.from_service_account_file(my_path + "\\" + "client_secrets.json")
client = vision.ImageAnnotatorClient(credentials=credentials)

path = my_path + "\\" + 'Image.jfif'
with io.open(path, 'rb') as image_file:
        content = image_file.read()

image = vision.Image(content=content)   #vision.types.Image(content=content)
print("image",image)

response = client.image_properties(image=image)

props = response.image_properties_annotation

print('Properties of the image:')

for color in props.dominant_colors.colors:
    print('Fraction: {}'.format(color.pixel_fraction))
    print('\tr: {}'.format(color.color.red))
    print('\tg: {}'.format(color.color.green))
    print('\tb: {}'.format(color.color.blue))













# from google.cloud import vision_v1


# def sample_async_batch_annotate_images(
#     input_image_uri="gs://cloud-samples-data/vision/label/wakeupcat.jpg",
#     output_uri="gs://your-bucket/prefix/",
# ):
#     """Perform async batch image annotation."""
#     client = vision_v1.ImageAnnotatorClient()

#     source = {"image_uri": input_image_uri}
#     image = {"source": source}
#     features = [
#         {"type_": vision_v1.Feature.Type.LABEL_DETECTION},
#         {"type_": vision_v1.Feature.Type.IMAGE_PROPERTIES},
#     ]

#     # Each requests element corresponds to a single image.  To annotate more
#     # images, create a request element for each image and add it to
#     # the array of requests
#     requests = [{"image": image, "features": features}]
#     gcs_destination = {"uri": output_uri}

#     # The max number of responses to output in each JSON file
#     batch_size = 2
#     output_config = {"gcs_destination": gcs_destination,
#                      "batch_size": batch_size}

#     operation = client.async_batch_annotate_images(requests=requests, output_config=output_config)

#     print("Waiting for operation to complete...")
#     response = operation.result(90)

#     # The output is written to GCS with the provided output_uri as prefix
#     gcs_output_uri = response.output_config.gcs_destination.uri
#     print("Output written to GCS with prefix: {}".format(gcs_output_uri))