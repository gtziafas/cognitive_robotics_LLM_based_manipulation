import clip
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image
import cv2

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# helper function
def display_image(path_or_array, size=(10, 10)):
  if isinstance(path_or_array, str):
    image = np.asarray(Image.open(open(image_path, 'rb')).convert("RGB"))
  else:
    image = path_or_array
  
  plt.figure(figsize=size)
  plt.imshow(image)
  plt.axis('off')
  plt.show()


# prompt engineering

def article(name):
  return 'an' if name[0] in 'aeiou' else 'a'

def processed_name(name, rm_dot=False):
  # _ for lvis
  # / for obj365
  res = name.replace('_', ' ').replace('/', ' or ').lower()
  if rm_dot:
    res = res.rstrip('.')
  return res

single_template = [
    'a photo of {article} {}.'
]

multiple_templates = [
    'There is {article} {} in the scene.',
    'There is the {} in the scene.',
    'a photo of {article} {} in the scene.',
    'a photo of the {} in the scene.',
    'a photo of one {} in the scene.',


    'itap of {article} {}.',
    'itap of my {}.',  # itap: I took a picture of
    'itap of the {}.',
    'a photo of {article} {}.',
    'a photo of my {}.',
    'a photo of the {}.',
    'a photo of one {}.',
    'a photo of many {}.',

    'a good photo of {article} {}.',
    'a good photo of the {}.',
    'a bad photo of {article} {}.',
    'a bad photo of the {}.',
    'a photo of a nice {}.',
    'a photo of the nice {}.',
    'a photo of a cool {}.',
    'a photo of the cool {}.',
    'a photo of a weird {}.',
    'a photo of the weird {}.',

    'a photo of a small {}.',
    'a photo of the small {}.',
    'a photo of a large {}.',
    'a photo of the large {}.',

    'a photo of a clean {}.',
    'a photo of the clean {}.',
    'a photo of a dirty {}.',
    'a photo of the dirty {}.',

    'a bright photo of {article} {}.',
    'a bright photo of the {}.',
    'a dark photo of {article} {}.',
    'a dark photo of the {}.',

    'a photo of a hard to see {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of {article} {}.',
    'a low resolution photo of the {}.',
    'a cropped photo of {article} {}.',
    'a cropped photo of the {}.',
    'a close-up photo of {article} {}.',
    'a close-up photo of the {}.',
    'a jpeg corrupted photo of {article} {}.',
    'a jpeg corrupted photo of the {}.',
    'a blurry photo of {article} {}.',
    'a blurry photo of the {}.',
    'a pixelated photo of {article} {}.',
    'a pixelated photo of the {}.',

    'a black and white photo of the {}.',
    'a black and white photo of {article} {}.',

    'a plastic {}.',
    'the plastic {}.',

    'a toy {}.',
    'the toy {}.',
    'a plushie {}.',
    'the plushie {}.',
    'a cartoon {}.',
    'the cartoon {}.',

    'an embroidered {}.',
    'the embroidered {}.',

    'a painting of the {}.',
    'a painting of a {}.',
]


# inference

class ClipInference:
    def __init__(self, vis_enc="ViT-B/32", device=device, jit=False, prompt_engineering=True, this_is=True):
        self.clip_model, self.clip_preprocess = clip.load(vis_enc, device, jit)
        self.clip_model.eval()
        self.clip_model.to(device)
        self.prompt_engineering = prompt_engineering
        self.this_is = this_is

    def build_text_embedding(self, categories, prompt_engineering=None, this_is=None):
      prompt_engineering = prompt_engineering or self.prompt_engineering
      this_is = this_is or self.this_is

      if prompt_engineering:
        templates = multiple_templates
      else:
        templates = single_template
    
      with torch.no_grad():
        all_text_embeddings = []
        # print('Building text embeddings...')
        for category in categories:
          texts = [
            template.format(processed_name(category['name'], rm_dot=True),
                            article=article(category['name']))
            for template in templates]
          if this_is:
            texts = [
                     'This is ' + text if text.startswith('a') or text.startswith('the') else text 
                     for text in texts
                     ]
          texts = clip.tokenize(texts).to(device) #tokenize
          text_embeddings = self.clip_model.encode_text(texts) #embed with text encoder
          text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
          text_embedding = text_embeddings.mean(dim=0) #average accross prompt templates
          text_embedding /= text_embedding.norm()
          all_text_embeddings.append(text_embedding)
        all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
        
      return all_text_embeddings.to(device).T

    @torch.no_grad()
    def extract_object_features(self, image, segmentation_mask):
        # extract image features from each object of the segmentation mask
        object_ids = np.unique(segmentation_mask)[2:] # first two are background
        masks = [np.where(segmentation_mask==objID, True, False) for objID in object_ids]

        object_image_features = []
        for m in masks:
            # both mask of full image resolution and cropped object image
            segm = m.copy()
            mask = np.ones_like(image) * 0xff
            mask[segm==True] = image[segm==True]
            # print(mask.shape)
            mask_input = self.clip_preprocess(Image.fromarray(mask)).to(device)
    
            # extract bounding box with opencv
            contours, _ = cv2.findContours(segm.astype(np.uint8) * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
            x, y, w, h = cv2.boundingRect(contours[0])
            crop_box = image[y:y+h, x:x+w, :]
            box_input = self.clip_preprocess(Image.fromarray(crop_box)).to(device)
            
            image_inputs = torch.stack([mask_input, box_input])
            #image_features = self.clip_model.encode_image(mask_input.unsqueeze(0))
            image_features = self.clip_model.encode_image(image_inputs)
            image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)
            object_image_features.append(image_features_norm)
            
        object_image_features = torch.stack(object_image_features)

        return {'features': object_image_features, 'IDs':object_ids.tolist(), 'masks': masks}

    @torch.no_grad()
    def get_most_similar(self, 
                         image, 
                         segmentation_mask, 
                         category_name_string, 
                         mode = "object",
                         prompt_engineering=None, 
                         this_is=None, 
                         show=False
    ):
        per_obj_data = self.extract_object_features(image, segmentation_mask) 
        # (N, 2, D), where N= #objects, 2= [mask,box], D= CLIP joint feature dim
        object_image_features = per_obj_data["features"]
        masks, object_ids = per_obj_data["masks"], per_obj_data["IDs"]
        
        # Preprocessing categories and get text embeddings
        category_names = [x.strip() for x in category_name_string.split(';')]
        categories = [{'name': item, 'id': idx+1,} for idx, item in enumerate(category_names)]
        
        # (K, D), where K= #categories, D= CLIP joint feature dim
        text_embedding = self.build_text_embedding(categories, prompt_engineering, this_is) 

        # compare each object feature with all category queries -> choose most similar
        similarities = (object_image_features @ text_embedding.T).mean(1) # (N,K)

        if mode == "object":
          # object -> most similar query for each object
          most_similar_indices =  similarities.argmax(1).cpu().numpy() # (N,)
          out = [{'mask':masks[object_ids.index(objID)],'category':category_names[index], 'objID':objID} for objID, index in zip(object_ids, most_similar_indices)]
          
        elif mode == "query":
          #query -> most similar object for each query
          most_similar_indices = similarities.argmax(0).cpu().numpy() # (K,)
          out = [{'mask':masks[index],'prompt':prompt, 'objID':object_ids[index]} for prompt, index in zip(category_names, most_similar_indices)]
          
        else:
          raise ValueError("Please set mode to either object (for recognition) or query (for grounding)")

        if show:
          if mode == "query":
            for prompt, index in zip(category_names, most_similar_indices):
                print(prompt, object_ids[index])
                mask_show = np.where(masks[index]>0, 255, 0).astype(np.uint8)[..., np.newaxis].repeat(3,2)
                display_image(np.hstack([image, mask_show]), (5,5))

          elif mode == "object":
            for objID, index in zip(object_ids, most_similar_indices):
                print(category_names[index], objID)
                mask_idx = object_ids.index(objID)
                mask_show = np.where(masks[mask_idx]>0, 255, 0).astype(np.uint8)[..., np.newaxis].repeat(3,2)
                display_image(np.hstack([image, mask_show]), (5,5))

        return out