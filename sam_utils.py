import matplotlib
import cv2
import numpy as np
import torch
from PIL import Image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# helpers
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.85)))


def display_image(path_or_array, size=(10, 10)):
  if isinstance(path_or_array, str):
    image = np.asarray(Image.open(open(image_path, 'rb')).convert("RGB"))
  else:
    image = path_or_array
  
  plt.figure(figsize=size)
  plt.imshow(image)
  plt.axis('off')
  plt.show()


DEFAULT_SAM_CONFIG = {
    'points_per_side': 64,
    'pred_iou_thresh': 0.9,
    'stability_score_thresh': 0.98,
    'crop_n_layers': 1,
    'crop_n_points_downscale_factor': 2,
    'min_mask_region_area': 10000,  # Requires open-cv to run post-processing
}


class SamInference:
    def __init__(self, backbone="vit_b", checkpoint="checkpoints/sam_vit_b_01ec64.pth", cfg=DEFAULT_SAM_CONFIG):
        self.checkpoint = checkpoint
        self.cfg = cfg
        self.backbone = backbone
        sam_model = sam_model_registry[self.backbone](checkpoint=self.checkpoint).to(device)

        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam_model,
            **self.cfg
        )

    def segment(self, image):
        if isinstance(image, str):
            # Load from path
            image = np.array(Image.open(image))
        elif isinstance(image, Image.Image):
            image = np.asarray(image)
        elif not isinstance(image, np.ndarray):
            raise ValueError("Provide either a path to load image, a numpy array or a PIL Image")

        torch.cuda.empty_cache()
        with torch.no_grad():
            masks = self.mask_generator.generate(image)
        torch.cuda.empty_cache()

        return masks
