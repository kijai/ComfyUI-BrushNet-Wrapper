import os
from contextlib import nullcontext
import torch
try:
    from diffusers import (
        DPMSolverMultistepScheduler, 
        EulerDiscreteScheduler, 
        EulerAncestralDiscreteScheduler, 
        AutoencoderKL, 
        LCMScheduler, 
        DDPMScheduler, 
        DEISMultistepScheduler, 
        PNDMScheduler,
        UniPCMultistepScheduler
    )
    from diffusers.loaders.single_file_utils import (
        convert_ldm_vae_checkpoint, 
        convert_ldm_unet_checkpoint, 
        create_vae_diffusers_config, 
        create_unet_diffusers_config,
        create_text_encoder_from_ldm_clip_checkpoint
    )            
except:
    raise ImportError("Diffusers version too old. Please update to 0.26.0 minimum.")

from .brushnet.pipeline_brushnet import StableDiffusionBrushNetPipeline
from .brushnet.brushnet import BrushNetModel
from .brushnet.unet_2d_condition import UNet2DConditionModel

from omegaconf import OmegaConf
from transformers import CLIPTokenizer
import comfy.model_management as mm
import comfy.utils
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))

class brushnet_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "vae": ("VAE",),
            "brushnet_model": (
                [
                    "brushnet_segmentation_mask",
                    "brushnet_random_mask",
                ], {
                    "default": "brushnet_segmentation_mask"
                }),
            },
        }

    RETURN_TYPES = ("BRUSHNET",)
    RETURN_NAMES = ("brushnet",)
    FUNCTION = "loadmodel"
    CATEGORY = "BrushNetWrapper"

    def loadmodel(self, model, clip, vae, brushnet_model):
        mm.soft_empty_cache()
        dtype = mm.unet_dtype()
        vae_dtype = mm.vae_dtype()
        device = mm.get_torch_device()

        custom_config = {
            "model": model,
            "vae": vae,
            "clip": clip,
            "brushnet_model": brushnet_model
        }
        if not hasattr(self, "model") or self.model == None or custom_config != self.current_config:
            pbar = comfy.utils.ProgressBar(5)
            self.current_config = custom_config

            original_config = OmegaConf.load(os.path.join(script_directory, f"configs/v1-inference.yaml"))
            brushnet_config = OmegaConf.load(os.path.join(script_directory, f"configs/brushnet_config.json"))

            brushnet_model_folder = os.path.join(folder_paths.models_dir,"brushnet")
            checkpoint_path = os.path.join(brushnet_model_folder, f"{brushnet_model}_fp16.safetensors")
            print(f"Loading BrushNet from {checkpoint_path}")
            
            if not os.path.exists(checkpoint_path):
                print(f"Selected model: {checkpoint_path} not found, downloading...")
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="Kijai/BrushNet-fp16", allow_patterns=[f"*{brushnet_model}*"], local_dir=brushnet_model_folder, local_dir_use_symlinks=False)

            brushnet = BrushNetModel(**brushnet_config)
            brushnet_sd = comfy.utils.load_torch_file(checkpoint_path)
            brushnet.load_state_dict(brushnet_sd)
            brushnet.to(dtype)

            clip_sd = None
            load_models = [model]
            load_models.append(clip.load_model())
            clip_sd = clip.get_sd()
            
            comfy.model_management.load_models_gpu(load_models)
            sd = model.model.state_dict_for_saving(clip_sd, vae.get_sd(), None)

            # 1. vae
            converted_vae_config = create_vae_diffusers_config(original_config, image_size=512)
            converted_vae = convert_ldm_vae_checkpoint(sd, converted_vae_config)
            vae = AutoencoderKL(**converted_vae_config)
            vae.load_state_dict(converted_vae, strict=False)

            pbar.update(1)
            # 2. unet
            converted_unet_config = create_unet_diffusers_config(original_config, image_size=512)
            converted_unet = convert_ldm_unet_checkpoint(sd, converted_unet_config)
            
            unet = UNet2DConditionModel(**converted_unet_config)
            unet.load_state_dict(converted_unet, strict=False)
            unet = unet.to(device)

            pbar.update(1)
            # 3. text_model
            print("loading text model")
            text_encoder = create_text_encoder_from_ldm_clip_checkpoint("openai/clip-vit-large-patch14",sd)
            scheduler_config = {
                "num_train_timesteps": 1000,
                "beta_start":    0.00085,
                "beta_end":      0.012,
                "beta_schedule": "scaled_linear",
                "steps_offset": 1
            }
            # 4. tokenizer
            tokenizer_path = os.path.join(script_directory, "configs/tokenizer")
            tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)

            scheduler=DPMSolverMultistepScheduler(**scheduler_config)
            pbar.update(1)
            del sd

            pbar.update(1)

            print("creating pipeline")
          
            self.pipe = StableDiffusionBrushNetPipeline(
                unet=unet, 
                vae=vae, 
                text_encoder=text_encoder, 
                tokenizer=tokenizer, 
                scheduler=scheduler,
                brushnet=brushnet,
                requires_safety_checker=False, 
                safety_checker=None,
                feature_extractor=None
            )   
            print("pipeline created")
            pbar.update(1)
            
            brushnet = {
                "pipe": self.pipe,
            }
   
        return (brushnet,)
    
class brushnet_sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "brushnet": ("BRUSHNET",),
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "steps": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1}),
            "guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "scheduler": (
                [
                    "DPMSolverMultistepScheduler",
                    "DPMSolverMultistepScheduler_SDE_karras",
                    "DDPMScheduler",
                    "LCMScheduler",
                    "PNDMScheduler",
                    "DEISMultistepScheduler",
                    "EulerDiscreteScheduler",
                    "EulerAncestralDiscreteScheduler",
                    "UniPCMultistepScheduler"
                ], {
                    "default": "UniPCMultistepScheduler"
                }),
            "prompt": ("STRING", {"multiline": True, "default": "caption",}),
           
            },    
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "BrushNetWrapper"

    def process(self, brushnet, image, mask, prompt, steps, guidance_scale, seed, scheduler):
        device = mm.get_torch_device()
        mm.unload_all_models()
        mm.soft_empty_cache()
        dtype = mm.unet_dtype()
        pipe=brushnet["pipe"]
        pipe.to(device, dtype=dtype)

        scheduler_config = {
                "num_train_timesteps": 1000,
                "beta_start":    0.00085,
                "beta_end":      0.012,
                "beta_schedule": "scaled_linear",
                "steps_offset": 1,
            }
        if scheduler == "DPMSolverMultistepScheduler":
            noise_scheduler = DPMSolverMultistepScheduler(**scheduler_config)
        elif scheduler == "DPMSolverMultistepScheduler_SDE_karras":
            scheduler_config.update({"algorithm_type": "sde-dpmsolver++"})
            scheduler_config.update({"use_karras_sigmas": True})
            noise_scheduler = DPMSolverMultistepScheduler(**scheduler_config)
        elif scheduler == "DDPMScheduler":
            noise_scheduler = DDPMScheduler(**scheduler_config)
        elif scheduler == "LCMScheduler":
            noise_scheduler = LCMScheduler(**scheduler_config)
        elif scheduler == "PNDMScheduler":
            scheduler_config.update({"set_alpha_to_one": False})
            scheduler_config.update({"trained_betas": None})
            noise_scheduler = PNDMScheduler(**scheduler_config)
        elif scheduler == "DEISMultistepScheduler":
            noise_scheduler = DEISMultistepScheduler(**scheduler_config)
        elif scheduler == "EulerDiscreteScheduler":
            noise_scheduler = EulerDiscreteScheduler(**scheduler_config)
        elif scheduler == "EulerAncestralDiscreteScheduler":
            noise_scheduler = EulerAncestralDiscreteScheduler(**scheduler_config)
        elif scheduler == "UniPCMultistepScheduler":
            noise_scheduler = UniPCMultistepScheduler(**scheduler_config)
        pipe.scheduler = noise_scheduler

        
        image = image.permute(0, 3, 1, 2).to(device)
        mask = mask.unsqueeze(0).to(device)
        image = image * (1-mask)

        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            generator = torch.Generator(device).manual_seed(seed)
            images = pipe(
                prompt, 
                image=image, 
                mask=mask, 
                num_inference_steps=steps, 
                generator=generator,
                brushnet_conditioning_scale=guidance_scale,
                output_type="pt"
            ).images

            image_out = images.permute(0, 2, 3, 1).cpu().float()
            return (image_out,)


NODE_CLASS_MAPPINGS = {
    "brushnet_model_loader": brushnet_model_loader,
    "brushnet_sampler": brushnet_sampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "brushnet_model_loader": "BrushNet Model Loader",
    "brushnet_sampler": "BrushNet Sampler",
}
