import os
import torch
import torch.nn.functional as F

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
    from .scheduling_tcd import TCDScheduler
    from diffusers.loaders.single_file_utils import (
        convert_ldm_vae_checkpoint, 
        convert_ldm_unet_checkpoint, 
        create_vae_diffusers_config, 
        create_unet_diffusers_config,
        create_text_encoder_from_ldm_clip_checkpoint
    )            
except:
    raise ImportError("Diffusers version too old. Please update to 0.27.2 minimum.")

from .brushnet.pipeline_brushnet import StableDiffusionBrushNetPipeline
from .brushnet.brushnet import BrushNetModel
from .brushnet.unet_2d_condition import UNet2DConditionModel

import safetensors.torch
from omegaconf import OmegaConf
from transformers import CLIPTokenizer

import comfy.model_management as mm
import comfy.utils
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))
IS_MODEL_CPU_OFFLOAD_ENABLED = False
    
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

        custom_config = {
            "model": model,
            "vae": vae,
            "clip": clip,
            "brushnet_model": brushnet_model
        }
        if not hasattr(self, "model") or self.model == None or custom_config != self.current_config:
            global IS_MODEL_CPU_OFFLOAD_ENABLED
            IS_MODEL_CPU_OFFLOAD_ENABLED = False
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
                snapshot_download(repo_id="Kijai/BrushNet-fp16", 
                                  allow_patterns=[f"*{brushnet_model}*"], 
                                  local_dir=brushnet_model_folder, 
                                  local_dir_use_symlinks=False
                                  )

            brushnet = BrushNetModel(**brushnet_config)
            brushnet_sd = comfy.utils.load_torch_file(checkpoint_path)
            brushnet.load_state_dict(brushnet_sd)
            brushnet.to(dtype)

            pbar.update(1)
            
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
            vae.to(dtype)

            pbar.update(1)
            # 2. unet
            converted_unet_config = create_unet_diffusers_config(original_config, image_size=512)
            converted_unet = convert_ldm_unet_checkpoint(sd, converted_unet_config)
            
            unet = UNet2DConditionModel(**converted_unet_config)
            unet.load_state_dict(converted_unet, strict=False)
            unet = unet.to(dtype)

            pbar.update(1)
            # 3. text_model
            print("loading text model")
            text_encoder = create_text_encoder_from_ldm_clip_checkpoint("openai/clip-vit-large-patch14",sd)
            text_encoder.to(dtype)

            # 4. tokenizer
            tokenizer_path = os.path.join(script_directory, "configs/tokenizer")
            tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)

            pbar.update(1)
            del sd
          
            self.pipe = StableDiffusionBrushNetPipeline(
                unet=unet, 
                vae=vae, 
                text_encoder=text_encoder, 
                tokenizer=tokenizer, 
                scheduler=None,
                brushnet=brushnet,
                requires_safety_checker=False, 
                safety_checker=None,
                feature_extractor=None
            )   
            #self.pipe.enable_model_cpu_offload()
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
            "cfg": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.01}),
            "cfg_brushnet": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.01}),
            "control_guidance_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "control_guidance_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "guess_mode": ("BOOLEAN", {"default": False}),
            "clip_skip": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1}),
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
                    "UniPCMultistepScheduler",
                    "TCDScheduler"
                ], {
                    "default": "UniPCMultistepScheduler"
                }),
            "prompt": ("STRING", {"multiline": True, "default": "caption",}),
            "n_prompt": ("STRING", {"multiline": True, "default": "caption",}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "BrushNetWrapper"

    def process(self, brushnet, image, mask, prompt, n_prompt, steps, cfg, guess_mode, clip_skip,
                cfg_brushnet, control_guidance_start, control_guidance_end, seed, scheduler):
        device = mm.get_torch_device()
        mm.soft_empty_cache()
        pipe=brushnet["pipe"]

        global IS_MODEL_CPU_OFFLOAD_ENABLED
        if not IS_MODEL_CPU_OFFLOAD_ENABLED:
            pipe.enable_model_cpu_offload()
            IS_MODEL_CPU_OFFLOAD_ENABLED = True

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
        elif scheduler == "TCDScheduler":
            noise_scheduler = TCDScheduler(**scheduler_config)
        pipe.scheduler = noise_scheduler

        B, H, W, C = image.shape
        image = image.permute(0, 3, 1, 2).to(device)

        #handle masks
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        mask = mask.to(device)
        if mask.shape[0] < B:
            repeat_times = B // mask.shape[0]
            mask = mask.repeat(repeat_times, 1, 1, 1)
        resized_mask = F.interpolate(mask.unsqueeze(1), size=[H, W], mode='nearest').squeeze(1)

        image = image * (1-resized_mask)

        #add prompt
        prompt_list = []
        prompt_list.append(prompt)
        if len(prompt_list) < B:
            prompt_list += [prompt_list[-1]] * (B - len(prompt_list))

        n_prompt_list = []
        n_prompt_list.append(n_prompt)
        if len(n_prompt_list) < B:
            n_prompt_list += [n_prompt_list[-1]] * (B - len(n_prompt_list))

        #sample    
        generator = torch.Generator(device).manual_seed(seed)
        print(prompt_list)
        images = pipe(
            prompt_list,
            negative_prompt=n_prompt_list,
            image=image,
            ipadapter_image=None,
            mask=resized_mask, 
            num_inference_steps=steps, 
            generator=generator,
            guidance_scale=cfg,
            guess_mode=guess_mode,
            clip_skip=clip_skip if clip_skip > 0 else None,
            brushnet_conditioning_scale=cfg_brushnet,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            output_type="pt"
        ).images

        image_out = images.permute(0, 2, 3, 1).cpu().float()
        return (image_out,)


class brushnet_ella_loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "brushnet": ("BRUSHNET",),           
            },
        }

    RETURN_TYPES = ("BRUSHNET",)
    RETURN_NAMES = ("brushnet",)
    FUNCTION = "loadmodel"
    CATEGORY = "ELLA-Wrapper"

    def loadmodel(self, brushnet):
        print("loading ELLA")
        from .ella.model import ELLA
        from .ella.ella_unet import ELLAProxyUNet
        checkpoint_path = os.path.join(folder_paths.models_dir,'ella')
        ella_path = os.path.join(checkpoint_path, 'ella-sd1.5-tsc-t5xl.safetensors')
        if not os.path.exists(ella_path):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="QQGYLab/ELLA", local_dir=checkpoint_path, local_dir_use_symlinks=False)
        
        ella = ELLA()
        safetensors.torch.load_model(ella, ella_path, strict=True)
        ella_unet = ELLAProxyUNet(ella, brushnet['pipe'].unet)
        brushnet['pipe'].unet = ella_unet
        
        return (brushnet,)      
      
class brushnet_sampler_ella:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "brushnet": ("BRUSHNET",),
            "ella_embeds": ("ELLAEMBEDS",),
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "steps": ("INT", {"default": 25, "min": 1, "max": 200, "step": 1}),
            "cfg": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.01}),
            "cfg_brushnet": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.01}),
            "control_guidance_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "control_guidance_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "guess_mode": ("BOOLEAN", {"default": False}),
            "clip_skip": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1}),
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
                    "UniPCMultistepScheduler",
                    "TCDScheduler"
                ], {
                    "default": "UniPCMultistepScheduler"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "BrushNetWrapper"

    def process(self, brushnet, image, mask, steps, cfg, guess_mode, clip_skip, ella_embeds,
                cfg_brushnet, control_guidance_start, control_guidance_end, seed, scheduler):
        device = mm.get_torch_device()
        dtype = mm.unet_dtype()
        mm.soft_empty_cache()
        pipe=brushnet["pipe"].to(dtype)

        global IS_MODEL_CPU_OFFLOAD_ENABLED
        if not IS_MODEL_CPU_OFFLOAD_ENABLED:
            pipe.enable_model_cpu_offload()
            IS_MODEL_CPU_OFFLOAD_ENABLED = True
  
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
        elif scheduler == "TCDScheduler":
            noise_scheduler = TCDScheduler(**scheduler_config)
        pipe.scheduler = noise_scheduler

        B, H, W, C = image.shape
        image = image.permute(0, 3, 1, 2).to(device)

        #handle masks
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        mask = mask.to(device)
        if mask.shape[0] < B:
            repeat_times = B // mask.shape[0]
            mask = mask.repeat(repeat_times, 1, 1, 1)
        resized_mask = F.interpolate(mask.unsqueeze(1), size=[H, W], mode='nearest').squeeze(1)

        image = image * (1-resized_mask)

        #sample    
        generator = torch.Generator(device).manual_seed(seed)
        
        images = pipe(
            prompt=None,
            negative_prompt=None,
            prompt_embeds=ella_embeds["prompt_embeds"],
            negative_prompt_embeds=ella_embeds["negative_prompt_embeds"],
            image=image,
            ipadapter_image=None,
            mask=resized_mask, 
            num_inference_steps=steps, 
            generator=generator,
            guidance_scale=cfg,
            guess_mode=guess_mode,
            clip_skip=clip_skip if clip_skip > 0 else None,
            brushnet_conditioning_scale=cfg_brushnet,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            output_type="pt"
        ).images

        image_out = images.permute(0, 2, 3, 1).cpu().float()
        return (image_out,)
    
NODE_CLASS_MAPPINGS = {
    "brushnet_model_loader": brushnet_model_loader,
    "brushnet_sampler": brushnet_sampler,
    "brushnet_sampler_ella": brushnet_sampler_ella,
    "brushnet_ella_loader": brushnet_ella_loader
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "brushnet_model_loader": "BrushNet Model Loader",
    "brushnet_sampler": "BrushNet Sampler",
    "brushnet_sampler_ella": "BrushNet Sampler (ELLA)",
    "brushnet_ella_loader": "BrushNet ELLA Loader"
}
