model_checkpoint = "models/sam_vit_h_4b8939.pth"
device = "cuda"
model_type = "default"


import base64
import numpy as np
import os
from PIL import Image


"""
from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry[model_type](checkpoint=model_checkpoint)
sam.to(device=device)
sam_predictor = SamPredictor(sam)
"""

from segment_anything import build_sam, SamPredictor
sam_predictor = SamPredictor(build_sam(checkpoint=model_checkpoint).to(device=device))


def encode_image(image_path, output_path):
    image = np.array(Image.open(image_path).convert('RGB'))

    sam_predictor.set_image(image)
    image_embedding = sam_predictor.get_image_embedding().cpu().numpy()
    image_embedding = image_embedding[0] # Get rid of batch
    
    #np.save(output_path, )
    print(image_embedding)
    arr_bytes = image_embedding.tostring()

    # Encode the string using Base64
    arr_b64 = base64.b64encode(arr_bytes).decode()

    with open(output_path, "wb") as f:
        f.write(arr_bytes)
    
    return arr_b64