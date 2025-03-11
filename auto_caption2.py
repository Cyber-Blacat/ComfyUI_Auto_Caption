import os
import torch
import torch.amp.autocast_mode
import re
import numpy as np
import shutil
from torch import nn
from huggingface_hub import InferenceClient
from transformers import AutoModel, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM
import torchvision.transforms.functional as TVF

from pathlib import Path
from PIL import Image, ImageOps
from typing import List, Union

from .lib.ximg import *
from .lib.xmodel import *
from comfy.utils import ProgressBar, common_upscale
import comfy.model_management as mm
import comfy.sd
import folder_paths



class JoyModel2:
    def __init__(self):
        self.clip_model = None
        self.clip_processor =None
        self.llm_model = None
        self.tokenizer = None
        self.image_adapter = None
        self.parent = None
    
    def clearCache(self):
        self.clip_model = None
        self.clip_processor =None
        self.tokenizer = None
        self.llm_model = None
        self.image_adapter = None 


class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int, ln1: bool, pos_emb: bool, num_image_tokens: int,
                 deep_extract: bool):
        super().__init__()
        self.deep_extract = deep_extract

        if self.deep_extract:
            input_features = input_features * 5

        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
        self.ln1 = nn.Identity() if not ln1 else nn.LayerNorm(input_features)
        self.pos_emb = None if not pos_emb else nn.Parameter(torch.zeros(num_image_tokens, input_features))

        # Other tokens (<|image_start|>, <|image_end|>, <|eot_id|>)
        self.other_tokens = nn.Embedding(3, output_features)
        self.other_tokens.weight.data.normal_(mean=0.0, std=0.02)  # Matches HF's implementation of LLaMA

    def forward(self, vision_outputs: torch.Tensor):
        if self.deep_extract:
            x = torch.cat((
                vision_outputs[-2],
                vision_outputs[3],
                vision_outputs[7],
                vision_outputs[13],
                vision_outputs[20],
            ), dim=-1)
            assert len(x.shape) == 3, f"Expected 3, got {len(x.shape)}"  # batch, tokens, features
            assert x.shape[-1] == vision_outputs[-2].shape[-1] * 5, f"Expected {vision_outputs[-2].shape[-1] * 5}, got {x.shape[-1]}"
        else:
            x = vision_outputs[-2]

        x = self.ln1(x)

        if self.pos_emb is not None:
            assert x.shape[-2:] == self.pos_emb.shape, f"Expected {self.pos_emb.shape}, got {x.shape[-2:]}"
            x = x + self.pos_emb

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        other_tokens = self.other_tokens(
            torch.tensor([0, 1], device=self.other_tokens.weight.device).expand(x.shape[0], -1))
        assert other_tokens.shape == (
            x.shape[0], 2, x.shape[2]), f"Expected {(x.shape[0], 2, x.shape[2])}, got {other_tokens.shape}"
        x = torch.cat((other_tokens[:, 0:1], x, other_tokens[:, 1:2]), dim=1)

        return x

    def get_eot_embedding(self):
        return self.other_tokens(torch.tensor([2], device=self.other_tokens.weight.device)).squeeze(0)


def llmloader(model_path, dtype, device="cuda:0", device_map=None):
    global current_device
    current_device = device  # 设置当前设备
    from transformers import AutoModel, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM
    from peft import PeftModel

    JC_lora = "text_model"
    use_lora = True if JC_lora != "none" else False
    CLIP_PATH = os.path.join(folder_paths.models_dir, "clip_vision", "siglip-so400m-patch14-384")
    CAPTION_PATH = os.path.join(folder_paths.models_dir, "loras-LLM", "cgrkzexw-599808")
    LORA_PATH = os.path.join(CAPTION_PATH, "text_model")
    
    # 加载siglip或者下载
    model_id = "google/siglip-so400m-patch14-384"
    if os.path.exists(CLIP_PATH):
        print("Start to load existing VLM")
    else:
        print("VLM not found locally. Downloading google/siglip-so400m-patch14-384...")
        try:
            # 下载clip(内含tokenzer与4bit量化版LLM一致)，snapshot函数中已构建目录
            CLIP_PATH = download_hg_model(model_id,"clip_vision")
        except Exception as e:
            print(f"Error downloading CLIP model: {e}")
            raise    


    try:
        if dtype == "nf4":
            from transformers import BitsAndBytesConfig
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, 
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            print("Loading in NF4")
            print("Loading CLIP")
            clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
            clip_model = AutoModel.from_pretrained(CLIP_PATH,trust_remote_code=True).vision_model

            print("Loading VLM's custom vision model")
            checkpoint = torch.load(os.path.join(CAPTION_PATH, "clip_model.pt"), map_location=current_device, weights_only=False)
            checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
            clip_model.load_state_dict(checkpoint)
            del checkpoint
            clip_model.eval()
            clip_model.requires_grad_(False).to(current_device)


            print(f"Loading LLM: {model_path}")
            llm_model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                quantization_config=nf4_config,
                device_map=current_device,  # 统一使用指定设备
                torch_dtype=torch.bfloat16
            ).eval()

            if use_lora and os.path.exists(LORA_PATH):
                print("Loading VLM's custom text model")
                llm_model = PeftModel.from_pretrained(
                    model=llm_model, 
                    model_id=LORA_PATH, 
                    device_map=current_device,  # 统一使用指定设备
                    quantization_config=nf4_config
                )
                llm_model = llm_model.merge_and_unload(safe_merge=True)
            else:
                print("VLM's custom text model isn't loaded")


        else:  # 选用 bf16
            print("Loading in bfloat16")
            print("Loading CLIP")
            clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
            clip_model = AutoModel.from_pretrained(CLIP_PATH,trust_remote_code=True).vision_model
            if os.path.exists(os.path.join(CAPTION_PATH, "clip_model.pt")):
                print("Loading VLM's custom vision model")
                checkpoint = torch.load(os.path.join(CAPTION_PATH, "clip_model.pt"), map_location=current_device, weights_only=False)
                checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
                clip_model.load_state_dict(checkpoint)
                del checkpoint
            clip_model.eval().requires_grad_(False)
            clip_model.to(current_device)

            # print("Loading tokenizer")
            # tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, use_fast=True)
            # assert isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)), f"Tokenizer is of type {type(tokenizer)}"

            print(f"Loading LLM: {model_path}")
            llm_model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map=current_device,  # 统一使用指定设备
                torch_dtype=torch.bfloat16
            )
            llm_model.eval()

            if use_lora and os.path.exists(LORA_PATH):
                print("Loading VLM's custom text model")
                llm_model = PeftModel.from_pretrained(
                    model=llm_model, 
                    model_id=LORA_PATH, 
                    device_map=current_device  # 统一使用指定设备
                )
                llm_model = llm_model.merge_and_unload(safe_merge=True)
            else:
                print("VLM's custom text model isn't loaded")


    except Exception as e:
        print(f"Error loading models: {e}", )
    finally:
        pass  # 可以在这里添加内存释放逻辑（如果需要）

    return (clip_model, clip_processor, llm_model)
    # return JoyModel2(clip_model,clip_processor, llm_model, None, None)



class Joy_Model2_load:

    def __init__(self):
        self.llm_model = None
        self.parent = None
        self.pipeline = JoyModel2()
        self.pipeline.parent = self
        pass

    @classmethod
    def INPUT_TYPES(cls):
        llm_model_list = ["unsloth/Meta-Llama-3.1-8B-Instruct", "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"]
        dtype_list = ['nf4', 'bf16']
        # 获取可用的GPU设备列表
        gpu_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if not gpu_devices:
            gpu_devices = ["cpu"]  # 如果没有GPU可用，则仅提供CPU选项
        
        
        #input widget
        return {
            "required": {
                "llm_model": (llm_model_list,), 
                "dtype": (dtype_list,),
                # "cache_model": ("BOOLEAN", {"default": False}),
                # 启用此项需要多卡
                # muti-GPU setting is required
                "device": (gpu_devices,),  
            }
        }

    CATEGORY = "Auto Caption"
    RETURN_TYPES = ("JoyModel2",)
    FUNCTION = "gen"

    def loadCheckPoint(self, llm_model, dtype, cache_model, device="cuda:0"):
        # cleanup
        if self.pipeline != None:
            self.pipeline.clearCache() 
       
        # LLM

        #LLM路径构造
        comfy_model_dir = os.path.join(folder_paths.models_dir, "LLM")
        print(f"comfy_model_dir: {comfy_model_dir}")
        if not os.path.exists(comfy_model_dir):
            os.mkdir(comfy_model_dir)
        leach_model_name = llm_model.split('/')[-1]
        llm_model_path = os.path.join(comfy_model_dir, leach_model_name)  
        llm_model_path_cache = os.path.join(comfy_model_dir, "cache--" + leach_model_name)

        # device chose
        selected_device = device if torch.cuda.is_available() else 'cpu'
        model_loaded_on = selected_device  # 跟踪模型加载在哪个设备上
        
        #load or download
        try:
            if os.path.exists(llm_model_path):
                print(f"Start to load existing model on {selected_device}")
            else:     #auto download from hg   
                download_hg_model(llm_model,llm_model_path_cache)
                shutil.move(llm_model_path_cache, llm_model_path)   
                print(f"Model downloaded to {llm_model_path_cache}...")  

            if self.parent is None:
                try:
                    # 尝试加载模型
                    free_vram_bytes = mm.get_free_memory()
                    free_vram_gb = free_vram_bytes / (1024 ** 3)
                    print(f"Free VRAM: {free_vram_gb:.2f} GB")
                    if dtype == 'nf4' and free_vram_gb < 10:
                        print("Free VRAM is less than 10GB when loading 'nf4' model. Performing VRAM cleanup.")
                        cleanGPU()
                    elif dtype == 'bf16' and free_vram_gb < 20:
                        print("Free VRAM is less than 20GB when loading 'bf16' model. Performing VRAM cleanup.")
                        cleanGPU()                    
                    # load LLM，使用所选设备。解包返回所需要的模型值
                    modelspackage = llmloader(
                        llm_model_path, dtype, device=selected_device, device_map=None)
                    #定义中间属性，确认模型缓存
                    # self.clip_model = clip_model
                    # self.clip_processor = clip_processor
                    # self.llm_model = llm_model
                except RuntimeError:
                    print("An error occurred while loading the model. Please check your configuration.")
            else:
                modelspackage=self.parent
                # self.pipeline.clip_model = self.clip_model
                # self.pipeline.clip_processor = self.clip_processor
                # self.pipeline.llm_model = self.llm_model

        except Exception as e:
            print(f"Error loading model: {e}")
            return None

        print(f"Model loaded on {model_loaded_on}")

        
        # # clip及lora和对应的tokenizer
        # model_id = "google/siglip-so400m-patch14-384"
        # CLIP_PATH = download_hg_model(model_id,"clip_vision")
        CAPTION_PATH = os.path.join(folder_paths.models_dir, "loras-LLM", "cgrkzexw-599808")
        

        # clip_processor = AutoProcessor.from_pretrained(CLIP_PATH) 
        # clip_model = AutoModel.from_pretrained(
        #         CLIP_PATH,
        #         trust_remote_code=True
        #     )
            
        # clip_model = clip_model.vision_model
        # clip_model.eval()
        # clip_model.requires_grad_(False)
        # clip_model.to("cuda")

    # 加载LLM
        # llm_model = AutoModelForCausalLM.from_pretrained(
        #     MODEL_PATH, 
        #     device_map="auto", 
        #     trust_remote_code=True)
        # llm_model.eval()

        tokenizer = AutoTokenizer.from_pretrained(llm_model_path, use_fast=True)        
        assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast), f"Tokenizer is of type {type(tokenizer)}"

        # load Image Adapter
        print("Loading image adapter")
        global current_device
        current_device = device  # 设置当前设备
        adapter_path =  os.path.join(CAPTION_PATH,"image_adapter.pt")

        
        #解包三个模型
        # clip_model = modelspackage[0]
        # clip_processor = modelspackage[1]
        # llm_model = modelspackage[2]
        self.clip_model, self.clip_processor, self.llm_model = modelspackage
        #统一用cuda加载imgadpter
        self.image_adapter = ImageAdapter(
                        self.clip_model.config.hidden_size, 
                        self.llm_model.config.hidden_size,
                        False, False, 38,
                        False ) # ImageAdapter(clip_model.config.hidden_size, 4096) 
        self.image_adapter.load_state_dict(torch.load(adapter_path, map_location=current_device, weights_only=False))
        adjusted_adapter =  self.image_adapter #AdjustedImageAdapter(image_adapter, llm_model.config.hidden_size)
        adjusted_adapter.eval()
        adjusted_adapter.to("cuda")
        # print("Loading image adapter")
        # image_adapter = ImageAdapter(
        #     clip_model.config.hidden_size, 
        #     llm_model.config.hidden_size, 
        #     False, False, 38,
        #     False
        # ).eval()
        # image_adapter.to(current_device)
        # image_adapter.load_state_dict(
        #     torch.load(os.path.join(CAPTION_PATH, "image_adapter.pt"), 
        #     map_location=current_device, weights_only=False)
        # )

    # pipeline ready for output
        self.pipeline.clip_model = self.clip_model
        self.pipeline.clip_processor = self.clip_processor
        self.pipeline.llm_model = self.llm_model
        self.pipeline.tokenizer = tokenizer
        self.pipeline.image_adapter = adjusted_adapter
        
    # 用于此函数内一开始的清除
    def clearCache(self):
         if self.pipeline != None:
              self.pipeline.clearCache()

    def gen(self, llm_model, dtype, device="cuda:0"):
        if self.llm_model == None or self.llm_model != llm_model or self.pipeline == None:
            self.llm_model = llm_model
            self.loadCheckPoint(llm_model, dtype, True, device)
        return (self.pipeline,)

class Auto_Caption2:

    CATEGORY = "Auto Caption"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    # OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "gen"

    def __init__(self):
        self.NODE_NAME = 'Auto Caption 2'
        self.previous_model = None

    @classmethod
    def INPUT_TYPES(cls):
        caption_type_list = [
            "Descriptive", "Descriptive (Informal)", "Training Prompt", "MidJourney",
            "Booru tag list", "Booru-like tag list", "Art Critic", "Product Listing",
            "Social Media Post"
        ]
        caption_length_list = [
            "any", "very short", "short", "medium-length", "long", "very long"
        ] + [str(i) for i in range(20, 261, 5)]

        # 获取可用的GPU设备列表
        gpu_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if not gpu_devices:
            gpu_devices = ["cpu"]  # 如果没有GPU可用，则仅提供CPU选项
        
        return {
            "required": {
                "JoyModel2": ("JoyModel2",),
                "image": ("IMAGE",),
                "caption_type": (caption_type_list,),
                "caption_length": (caption_length_list,),                
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 8, "max": 4096, "step": 1}),
                "cache": ("BOOLEAN", {"default": False}),
                "device": (gpu_devices,),  # 新增GPU设备选择 
            },
            "optional": {
                "ExtraOptionsSet": ("STRING",{"forceInput": True}),  # 接收来自 ExtraOptionsNode 的单一字符串
            },            
        }



    @classmethod
    def gen(self,JoyModel2,image,
            caption_type,caption_length, user_prompt,
            top_p, temperature, max_new_tokens, 
            cache,device,
            ExtraOptionsSet=None): 

        # if JoyModel2.clip_processor == None :
        #     JoyModel2.parent.loadCheckPoint() 


        # clip_processor = JoyModel2.clip_processor
        # clip_model = JoyModel2.clip_model
        # tokenizer = JoyModel2.tokenizer
        # image_adapter = JoyModel2.image_adapter
        # llm_model = JoyModel2.llm_model

     
        # 接收来自 ExtraOptionsNode 的额外提示
        extra = []
        if ExtraOptionsSet and ExtraOptionsSet.strip():
            extra = [ExtraOptionsSet]  # 将单一字符串包装成列表
            print(f"Extra options enabled: {ExtraOptionsSet}")
        else:
            print("No extra options provided.")

        # Preprocess image
        ret_text = [] 
        input_image = [tensor2pil(img) for img in image]
        try:
            captions = stream_chat(
                input_image, caption_type, caption_length,
                extra, "", user_prompt,
                max_new_tokens, top_p, temperature, len(input_image),
                JoyModel2, device  # 确保传递正确的设备
            )
            ret_text.extend(captions)
        except Exception as e:
            print(f"Error during stream_chat: {e}")
            return ("Error generating captions.",)

        if cache == False:
            del JoyModel2
            free_memory()

        return (ret_text,)

        # Process image
        pImge = clip_processor(images=input_image, return_tensors='pt').pixel_values
        pImge = pImge.to('cuda')

        # Tokenize the prompt
        user_prompt = tokenizer.encode(user_prompt, return_tensors='pt', padding=False, truncation=False, add_special_tokens=False)
        # Embed image
        with torch.amp.autocast_mode.autocast('cuda', enabled=True):
            vision_outputs = clip_model(pixel_values=pImge, output_hidden_states=True)
            image_features = vision_outputs.hidden_states[-2]
            embedded_images = image_adapter(image_features)
            embedded_images = embedded_images.to('cuda')

        # Embed prompt
        prompt_embeds = llm_model.model.embed_tokens(user_prompt.to('cuda'))
        assert prompt_embeds.shape == (1, user_prompt.shape[1], llm_model.config.hidden_size), f"Prompt shape is {prompt_embeds.shape}, expected {(1, prompt.shape[1], llm_model.config.hidden_size)}"
        embedded_bos = llm_model.model.embed_tokens(torch.tensor([[tokenizer.bos_token_id]], device=llm_model.device, dtype=torch.int64))   

        # Construct prompts
        inputs_embeds = torch.cat([
            embedded_bos.expand(embedded_images.shape[0], -1, -1),
            embedded_images.to(dtype=embedded_bos.dtype),
            prompt_embeds.expand(embedded_images.shape[0], -1, -1),
        ], dim=1)

        input_ids = torch.cat([
            torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long),
            torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
            user_prompt,
        ], dim=1).to('cuda')
        attention_mask = torch.ones_like(input_ids)
        
        generate_ids = llm_model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=True, top_k=10, temperature=temperature, suppress_tokens=None)

        # Trim off the prompt
        generate_ids = generate_ids[:, input_ids.shape[1]:]
        if generate_ids[0][-1] == tokenizer.eos_token_id:
            generate_ids = generate_ids[:, :-1]





        caption = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        r = caption.strip()


        if cache == False:
           JoyModel2.parent.clearCache()  

        return (r,)


class ExtraOptionsSet:
    CATEGORY = 'Auto Caption'
    FUNCTION = 'extra_options'
    RETURN_TYPES = ("STRING",)  # 改为返回单一字符串
    RETURN_NAMES = ("ExtraOptionsSet",)
    OUTPUT_IS_LIST = (False,)  # 单一字符串输出

    def __init__(self):
        self.NODE_NAME = 'ExtraOptionsSet'

    @classmethod
    def INPUT_TYPES(cls):
        # 获取 extra_option.json 的路径并加载选项
        current_dir = os.path.dirname(os.path.abspath(__file__))
        extra_option_file = os.path.join(current_dir, "lib","extra_option.json")
        extra_options_list = {}

        if os.path.isfile(extra_option_file):
            try:
                with open(extra_option_file, "r", encoding='utf-8') as f:
                    json_content = json.load(f)
                    for item in json_content:
                        option_name = item.get("name")
                        if option_name:
                            # 定义每个额外选项为布尔输入
                            extra_options_list[option_name] = ("BOOLEAN", {"default": False})
            except Exception as e:
                print(f"Error loading extra_option.json: {e}")
        else:
            print(f"extra_option.json not found at {extra_option_file}. No extra options will be available.")

        # 定义输入字段，包括开关和 character_name
        return {
            "required": {
                "enable_extra_options": ("BOOLEAN", {"default": True, "label": "启用额外选项"}),  # 开关
                **extra_options_list,  # 动态加载的额外选项
                "character_name": ("STRING", {"default": "", "multiline": False}),  # 移动 character_name
            },
        }

    def extra_options(self, enable_extra_options, character_name, **extra_options):
        """
        处理额外选项并返回已启用的提示列表。
        如果启用了替换角色名称选项，并提供了 character_name，则进行替换。
        """
        extra_prompts = []
        if enable_extra_options:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            extra_option_file = os.path.join(base_dir, "lib","extra_option.json")
            if os.path.isfile(extra_option_file):
                try:
                    with open(extra_option_file, "r", encoding='utf-8') as f:
                        json_content = json.load(f)
                        for item in json_content:
                            name = item.get("name")
                            prompt = item.get("prompt")
                            if name and prompt:
                                if extra_options.get(name):
                                    # 如果 prompt 中包含 {name}，则替换为 character_name
                                    if "{name}" in prompt:
                                        prompt = prompt.replace("{name}", character_name)
                                    extra_prompts.append(prompt)
                except Exception as e:
                    print(f"Error reading extra_option.json: {e}")
            else:
                print(f"extra_option.json not found at {extra_option_file} during processing.")

        # 将所有启用的提示拼接成一个字符串
        return (" ".join(extra_prompts),)  # 返回一个单一的合并字符串


def stream_chat(input_images: List[Image.Image], caption_type: str, caption_length: Union[str, int],
                extra_options: list[str], name_input: str, custom_prompt: str,
                max_new_tokens: int, top_p: float, temperature: float, batch_size: int, 
                model: JoyModel2, current_device=str):

    # 确定 chat_device
    if 'cuda' in current_device:
        chat_device = 'cuda'
    elif 'cpu' in current_device:
        chat_device = 'cpu'
    else:
        raise ValueError(f"Unsupported device type: {current_device}")


    CAPTION_TYPE_MAP = {
        "Descriptive": [
            "Write a descriptive caption for this image in a formal tone.",
            "Write a descriptive caption for this image in a formal tone within {word_count} words.",
            "Write a {length} descriptive caption for this image in a formal tone.",
        ],
        "Descriptive (Informal)": [
            "Write a descriptive caption for this image in a casual tone.",
            "Write a descriptive caption for this image in a casual tone within {word_count} words.",
            "Write a {length} descriptive caption for this image in a casual tone.",
        ],
        "Training Prompt": [
            "Write a stable diffusion prompt for this image.",
            "Write a stable diffusion prompt for this image within {word_count} words.",
            "Write a {length} stable diffusion prompt for this image.",
        ],
        "MidJourney": [
            "Write a MidJourney prompt for this image.",
            "Write a MidJourney prompt for this image within {word_count} words.",
            "Write a {length} MidJourney prompt for this image.",
        ],
        "Booru tag list": [
            "Write a list of Booru tags for this image.",
            "Write a list of Booru tags for this image within {word_count} words.",
            "Write a {length} list of Booru tags for this image.",
        ],
        "Booru-like tag list": [
            "Write a list of Booru-like tags for this image.",
            "Write a list of Booru-like tags for this image within {word_count} words.",
            "Write a {length} list of Booru-like tags for this image.",
        ],
        "Art Critic": [
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
        ],
        "Product Listing": [
            "Write a caption for this image as though it were a product listing.",
            "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
            "Write a {length} caption for this image as though it were a product listing.",
        ],
        "Social Media Post": [
            "Write a caption for this image as if it were being used for a social media post.",
            "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
            "Write a {length} caption for this image as if it were being used for a social media post.",
        ],
    }

    all_captions = []

    # 'any' means no length specified
    length = None if caption_length == "any" else caption_length

    if isinstance(length, str):
        try:
            length = int(length)
        except ValueError:
            pass

    # Build prompt
    if length is None:
        map_idx = 0
    elif isinstance(length, int):
        map_idx = 1
    elif isinstance(length, str):
        map_idx = 2
    else:
        raise ValueError(f"Invalid caption length: {length}")

    prompt_str = CAPTION_TYPE_MAP[caption_type][map_idx]

    # Add extra options
    if len(extra_options) > 0:
        prompt_str += " " + " ".join(extra_options)

    # Add name, length, word_count
    prompt_str = prompt_str.format(name=name_input, length=caption_length, word_count=caption_length)

    if custom_prompt.strip() != "":
        prompt_str = custom_prompt.strip()

    # For debugging
    print(f"Prompt: {prompt_str}")

    for i in range(0, len(input_images), batch_size):
        batch = input_images[i:i + batch_size]

        for input_image in batch:
            try:
                # Preprocess image
                image = input_image.resize((384, 384), Image.LANCZOS)
                pixel_values = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
                pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
                pixel_values = pixel_values.to(chat_device)
            except ValueError as e:
                print(f"Error processing image: {e}")
                print("Skipping this image and continuing...")
                continue

            # Embed image
            with torch.amp.autocast_mode.autocast(chat_device, enabled=True):
                vision_outputs = model.clip_model(pixel_values=pixel_values, output_hidden_states=True)
                image_features = vision_outputs.hidden_states
                embedded_images = model.image_adapter(image_features).to(chat_device)

            # Build the conversation
            convo = [
                {
                    "role": "system",
                    "content": "You are a helpful image captioner.",
                },
                {
                    "role": "user",
                    "content": prompt_str,
                },
            ]

            # Format the conversation
            if hasattr(model.tokenizer, 'apply_chat_template'):
                convo_string = model.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            else:
                # Fallback if apply_chat_template is not available
                convo_string = "<|eot_id|>\n"
                for message in convo:
                    if message['role'] == 'system':
                        convo_string += f"<|system|>{message['content']}<|endoftext|>\n"
                    elif message['role'] == 'user':
                        convo_string += f"<|user|>{message['content']}<|endoftext|>\n"
                    else:
                        convo_string += f"{message['content']}<|endoftext|>\n"
                convo_string += "<|eot_id|>"

            assert isinstance(convo_string, str)

            # Tokenize the conversation
            convo_tokens = model.tokenizer.encode(convo_string, return_tensors="pt", add_special_tokens=False,
                                                  truncation=False)
            prompt_tokens = model.tokenizer.encode(prompt_str, return_tensors="pt", add_special_tokens=False,
                                                   truncation=False)
            assert isinstance(convo_tokens, torch.Tensor) and isinstance(prompt_tokens, torch.Tensor)
            convo_tokens = convo_tokens.squeeze(0)
            prompt_tokens = prompt_tokens.squeeze(0)

            # Calculate where to inject the image
            eot_id_indices = (convo_tokens == model.tokenizer.convert_tokens_to_ids("<|eot_id|>")).nonzero(as_tuple=True)[
                0].tolist()
            assert len(eot_id_indices) == 2, f"Expected 2 <|eot_id|> tokens, got {len(eot_id_indices)}"

            preamble_len = eot_id_indices[1] - prompt_tokens.shape[0]

            # Embed the tokens
            convo_embeds = model.llm_model.model.embed_tokens(convo_tokens.unsqueeze(0).to(current_device))

            # Construct the input
            input_embeds = torch.cat([
                convo_embeds[:, :preamble_len],
                embedded_images.to(dtype=convo_embeds.dtype),
                convo_embeds[:, preamble_len:],
            ], dim=1).to(chat_device)

            input_ids = torch.cat([
                convo_tokens[:preamble_len].unsqueeze(0),
                torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
                convo_tokens[preamble_len:].unsqueeze(0),
            ], dim=1).to(chat_device)
            attention_mask = torch.ones_like(input_ids)

            generate_ids = model.llm_model.generate(input_ids=input_ids, inputs_embeds=input_embeds,
                                                     attention_mask=attention_mask, do_sample=True,
                                                     suppress_tokens=None, max_new_tokens=max_new_tokens, top_p=top_p,
                                                     temperature=temperature)

            # Trim off the prompt
            generate_ids = generate_ids[:, input_ids.shape[1]:]
            if generate_ids[0][-1] == model.tokenizer.eos_token_id or generate_ids[0][-1] == model.tokenizer.convert_tokens_to_ids(
                    "<|eot_id|>"):
                generate_ids = generate_ids[:, :-1]

            caption = model.tokenizer.batch_decode(generate_ids, skip_special_tokens=False,
                                                   clean_up_tokenization_spaces=False)[0]
            all_captions.append(caption.strip())

    return all_captions


def cleanGPU():
    gc.collect()
    mm.unload_all_models()
    mm.soft_empty_cache()

def free_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
def get_torch_device_patched():
    global current_device
    if (
        not torch.cuda.is_available()
        or comfy.model_management.cpu_state == comfy.model_management.CPUState.CPU
    ):
        return torch.device("cpu")

    return torch.device(current_device)

# 设置全局设备变量
current_device = "cuda:0"
# 覆盖ComfyUI的设备获取函数
comfy.model_management.get_torch_device = get_torch_device_patched