import os
import folder_paths
import torch
import torch.amp.autocast_mode
import re
import numpy as np

from torch import nn
from huggingface_hub import InferenceClient
from transformers import AutoModel, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM
from pathlib import Path
from PIL import Image, ImageOps
from .lib.ximg import *
from .lib.xmodel import *
from comfy.utils import ProgressBar, common_upscale

class JoyModel:
    def __init__(self):
        self.clip_model = None
        self.clip_processor =None
        self.tokenizer = None
        self.text_model = None
        self.image_adapter = None
        self.parent = None
    
    def clearCache(self):
        self.clip_model = None
        self.clip_processor =None
        self.tokenizer = None
        self.text_model = None
        self.image_adapter = None 


class ImageAdapter(nn.Module):
	def __init__(self, input_features: int, output_features: int):
		super().__init__()
		self.linear1 = nn.Linear(input_features, output_features)
		self.activation = nn.GELU()
		self.linear2 = nn.Linear(output_features, output_features)
	
	def forward(self, vision_outputs: torch.Tensor):
		x = self.linear1(vision_outputs)
		x = self.activation(x)
		x = self.linear2(x)
		return x

class Joy_Model_load:

    def __init__(self):
        self.model = None
        self.pipeline = JoyModel()
        self.pipeline.parent = self
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["unsloth/Meta-Llama-3.1-8B-bnb-4bit", "meta-llama/Meta-Llama-3.1-8B"],), 
               
            }
        }

    CATEGORY = "Auto Caption"
    RETURN_TYPES = ("JoyModel",)
    FUNCTION = "gen"

    def loadCheckPoint(self):
        # 清除一波
        if self.pipeline != None:
            self.pipeline.clearCache() 
       
         # clip
        model_id = "google/siglip-so400m-patch14-384"
        CLIP_PATH = download_hg_model(model_id,"clip_vision")

        clip_processor = AutoProcessor.from_pretrained(CLIP_PATH) 
        clip_model = AutoModel.from_pretrained(
                CLIP_PATH,
                trust_remote_code=True
            )
            
        clip_model = clip_model.vision_model
        clip_model.eval()
        clip_model.requires_grad_(False)
        clip_model.to("cuda")

       
        # LLM
        MODEL_PATH = download_hg_model(self.model,"LLM")
        LORA_PATH = os.path.join(folder_paths.models_dir, "loras-LLM", "cgrkzexw-599808")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
        # tokenizer = AutoTokenizer.from_pretrained(os.path.join(CAPTION_PATH, "text_model"), use_fast=True)
        assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast), f"Tokenizer is of type {type(tokenizer)}"

        text_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto",trust_remote_code=True)
        text_model.eval()

        # Image Adapter
        adapter_path =  os.path.join(folder_paths.models_dir,"loras-LLM","image_adapter.pt")

        image_adapter = ImageAdapter(clip_model.config.hidden_size, text_model.config.hidden_size) # ImageAdapter(clip_model.config.hidden_size, 4096) 
        image_adapter.load_state_dict(torch.load(adapter_path, map_location="cpu"))
        adjusted_adapter =  image_adapter #AdjustedImageAdapter(image_adapter, text_model.config.hidden_size)
        adjusted_adapter.eval()
        adjusted_adapter.to("cuda")

        self.pipeline.clip_model = clip_model
        self.pipeline.clip_processor = clip_processor
        self.pipeline.tokenizer = tokenizer
        self.pipeline.text_model = text_model
        self.pipeline.image_adapter = adjusted_adapter
    
    def clearCache(self):
         if self.pipeline != None:
              self.pipeline.clearCache()

    def gen(self,model):
        if self.model == None or self.model != model or self.pipeline == None:
            self.model = model
            self.loadCheckPoint()
        return (self.pipeline,)

class Auto_Caption:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "JoyModel": ("JoyModel",),
                "image": ("IMAGE",),
                "prompt":   ("STRING", {"multiline": True, "default": "A descriptive caption for this image"},),
                "max_new_tokens":("INT", {"default": 1024, "min": 10, "max": 4096, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cache": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "Auto Caption"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "gen"
    def gen(self,JoyModel,image,prompt,max_new_tokens,temperature,cache): 

        if JoyModel.clip_processor == None :
            JoyModel.parent.loadCheckPoint()    

        clip_processor = JoyModel.clip_processor
        tokenizer = JoyModel.tokenizer
        clip_model = JoyModel.clip_model
        image_adapter = JoyModel.image_adapter
        text_model = JoyModel.text_model

     

        input_image = tensor2pil(image)

        # Preprocess image
        pImge = clip_processor(images=input_image, return_tensors='pt').pixel_values
        pImge = pImge.to('cuda')

        # Tokenize the prompt
        prompt = tokenizer.encode(prompt, return_tensors='pt', padding=False, truncation=False, add_special_tokens=False)
        # Embed image
        with torch.amp.autocast_mode.autocast('cuda', enabled=True):
            vision_outputs = clip_model(pixel_values=pImge, output_hidden_states=True)
            image_features = vision_outputs.hidden_states[-2]
            embedded_images = image_adapter(image_features)
            embedded_images = embedded_images.to('cuda')

        # Embed prompt
        prompt_embeds = text_model.model.embed_tokens(prompt.to('cuda'))
        assert prompt_embeds.shape == (1, prompt.shape[1], text_model.config.hidden_size), f"Prompt shape is {prompt_embeds.shape}, expected {(1, prompt.shape[1], text_model.config.hidden_size)}"
        embedded_bos = text_model.model.embed_tokens(torch.tensor([[tokenizer.bos_token_id]], device=text_model.device, dtype=torch.int64))   

        # Construct prompts
        inputs_embeds = torch.cat([
            embedded_bos.expand(embedded_images.shape[0], -1, -1),
            embedded_images.to(dtype=embedded_bos.dtype),
            prompt_embeds.expand(embedded_images.shape[0], -1, -1),
        ], dim=1)

        input_ids = torch.cat([
            torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long),
            torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
            prompt,
        ], dim=1).to('cuda')
        attention_mask = torch.ones_like(input_ids)
        
        generate_ids = text_model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=True, top_k=10, temperature=temperature, suppress_tokens=None)

        # Trim off the prompt
        generate_ids = generate_ids[:, input_ids.shape[1]:]
        if generate_ids[0][-1] == tokenizer.eos_token_id:
            generate_ids = generate_ids[:, :-1]

        caption = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        r = caption.strip()

        if cache == False:
           JoyModel.parent.clearCache()  

        return (r,)
    
class LoadImagesRezise:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 50, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING",)
    RETURN_NAMES = ("image", "mask", "count", "image_path",)
    FUNCTION = "load_images"

    CATEGORY = "Auto Caption"

    def load_images(self, folder, image_load_cap, start_index):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder '{folder}' cannot be found.")
        
        dir_files = os.listdir(folder)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{folder}'.")

        # Filter files by valid image extensions
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        # Sort files based on numeric value extracted from filename
        def extract_number(file_name):
            match = re.search(r'(\d+)', file_name)
            return int(match.group(0)) if match else float('inf')  # Use 'inf' if no number is found to push such files at the end

        dir_files = sorted(dir_files, key=extract_number)

        # Convert to full file paths
        dir_files = [os.path.join(folder, x) for x in dir_files]

        # Start at the specified start_index
        dir_files = dir_files[start_index:]

        images = []
        masks = []
        image_path_list = []

        limit_images = False
        if image_load_cap > 0:
            limit_images = True
        image_count = 0

        has_non_empty_mask = False

        for image_path in dir_files:
            if os.path.isdir(image_path):
                continue
            if limit_images and image_count >= image_load_cap:
                break
            
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)  # Handle EXIF orientation
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]  # Add a batch dimension
            
            if 'A' in i.getbands():  # Check for alpha channel
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)  # Invert the alpha mask
                has_non_empty_mask = True
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            
            images.append(image)
            masks.append(mask)
            image_path_list.append(image_path)
            image_count += 1

        if len(images) == 1:
            return (images[0], masks[0], 1)

        elif len(images) > 1:
            image1 = images[0]
            mask1 = None

            for image2 in images[1:]:
                if image1.shape[1:] != image2.shape[1:]:
                    image2 = common_upscale(image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "bilinear", "center").movedim(1, -1)
                image1 = torch.cat((image1, image2), dim=0)

            for mask2 in masks[1:]:
                if has_non_empty_mask:
                    if image1.shape[1:3] != mask2.shape:
                        mask2 = torch.nn.functional.interpolate(mask2.unsqueeze(0).unsqueeze(0), size=(image1.shape[2], image1.shape[1]), mode='bilinear', align_corners=False)
                        mask2 = mask2.squeeze(0)
                    else:
                        mask2 = mask2.unsqueeze(0)
                else:
                    mask2 = mask2.unsqueeze(0)

                if mask1 is None:
                    mask1 = mask2
                else:
                    mask1 = torch.cat((mask1, mask2), dim=0)

            return (image1, mask1, len(images), image_path_list)
        
class LoadManyImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 50, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING",)
    RETURN_NAMES = ("image", "mask", "count", "image_path",)

    OUTPUT_IS_LIST = (True, True, True, True)

    FUNCTION = "load_images"

    CATEGORY = "Auto Caption"

    def load_images(self, folder, image_load_cap, start_index):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder '{folder}' cannot be found.")
        
        dir_files = os.listdir(folder)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{folder}'.")

        # Filter files by valid image extensions
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        # Sort files based on numeric value extracted from filename
        def extract_number(file_name):
            match = re.search(r'(\d+)', file_name)
            return int(match.group(0)) if match else float('inf')  # Use 'inf' if no number is found to push such files at the end

        dir_files = sorted(dir_files, key=extract_number)
        # 

        # Convert to full file paths
        dir_files = [os.path.join(folder, x) for x in dir_files]

        # Start at the specified start_index
        dir_files = dir_files[start_index:]

        images = []
        masks = []
        image_path_list = []

        limit_images = False
        if image_load_cap > 0:
            limit_images = True
        image_count = 0

        for image_path in dir_files:
            if os.path.isdir(image_path) and os.path.ex:
                continue
            if limit_images and image_count >= image_load_cap:
                break
            
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)  # Handle EXIF orientation
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            
            
            if 'A' in i.getbands():  # Check for alpha channel
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)  # Invert the alpha mask
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            
            images.append(image)
            masks.append(mask)
            image_path_list.append(image_path)
            image_count += 1
        
        return (images, masks, image_path_list)
