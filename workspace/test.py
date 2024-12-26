import cv2
import numpy as np
import PIL.Image
from matplotlib import pyplot as plt

import trtdepthpro


def compute_depth(W, H, fov_deg, canonical_inverse_depth):
    f_px = 0.5 * W / np.tan(0.5 * np.radians(fov_deg))
    inverse_depth = canonical_inverse_depth * (W / f_px)
    f_px = np.squeeze(f_px)
    depth = 1.0 / np.clip(inverse_depth, 1e-4, 1e4)
    return depth

def normlized(depth):
    inverse_depth = 1 / depth
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (max_invdepth_vizu - min_invdepth_vizu)
    return inverse_depth_normalized

def save_color_map(inverse_depth_normalized, save_path):
    cmap = plt.get_cmap("turbo")

    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
    PIL.Image.fromarray(color_depth).save(save_path, format="JPEG", quality=90)

def test():
    engine_path = "depth_pro.engine"
    image_path  = "images/bus.jpg"

    image = cv2.imread(image_path)
    h, w, _ = image.shape

    model = trtdepthpro.TrtDepthProInfer(engine_path, 0)
    res = model.forward(image)
    depth_map = np.array(res.depth_map)
    depth_map = depth_map.reshape(h, w)

    depth = compute_depth(w, h, depth_map, np.array([res.fov_deg]))

    inverse_depth_normalized = normlized(depth)
    save_color_map(inverse_depth_normalized, "res.jpg")

if __name__ == "__main__":
    test()
