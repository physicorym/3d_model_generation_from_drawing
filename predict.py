
import os
from PIL import Image
import numpy as np

import torch
import cadquery as cq

from arch import ViTUNet3D
from config.config import DATA_DIR, SIZE, DEVICE


def predict_3d():

    list_name = os.listdir(DATA_DIR)

    for name in list_name:
        base_name = ''.join(name.split('.')[:-1])

        images = []
        list_postfix = ['f', 'r', 't']
        for postfix in list_postfix:
            img_name = f'{base_name}_{postfix}.png'
            image = Image.open(img_name).convert('L')
            image = image.resize((SIZE, SIZE))
            images.append(np.array(image, dtype=np.float32) / 255.0)
        
    drawings_arr = np.stack(images, axis=0)
    drawings_arr = torch.from_numpy(drawings_arr)

    model = ViTUNet3D()

    with torch.no_grad():
        pred_voxels = model(drawings_arr[None, ...].to(DEVICE))

    pred_voxels_binary = (torch.sigmoid(pred_voxels[0]).cpu().numpy()) > 0.5

    result = cq.Workplane("XY")

    for z in range(pred_voxels_binary.shape[0]):
        for y in range(pred_voxels_binary.shape[1]):
            for x in range(pred_voxels_binary.shape[2]):
                if pred_voxels_binary[z, y, x] > 0:
                    result = result.add(cq.Workplane("XY").transformed(offset=(x, y, z)).box(1, 1, 1))

    result.val().exportStep("voxels_model.step")


if __name__ == "__main__":
    predict_3d()