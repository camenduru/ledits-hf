import gradio as gr
import torch
import requests
from io import BytesIO
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
from utils import *
from inversion_utils import *
from modified_pipeline_semantic_stable_diffusion import SemanticStableDiffusionPipeline
from torch import autocast, inference_mode

def invert(x0, prompt_src="", num_diffusion_steps=100, cfg_scale_src = 3.5, eta = 1):

  #  inverts a real image according to Algorihm 1 in https://arxiv.org/pdf/2304.06140.pdf, 
  #  based on the code in https://github.com/inbarhub/DDPM_inversion
   
  #  returns wt, zs, wts:
  #  wt - inverted latent
  #  wts - intermediate inverted latents
  #  zs - noise maps

  sd_pipe.scheduler.set_timesteps(num_diffusion_steps)

  # vae encode image
  w0 = (sd_pipe.vae.encode(x0).latent_dist.mode() * 0.18215).float()

  # find Zs and wts - forward process
  wt, zs, wts = inversion_forward_process(sd_pipe, w0, etas=eta, prompt=prompt_src, cfg_scale=cfg_scale_src, prog_bar=True, num_inference_steps=num_diffusion_steps)
  return wt, zs, wts



def sample(wt, zs, wts, prompt_tar="", cfg_scale_tar=15, skip=36, eta = 1):

    # reverse process (via Zs and wT)
    w0, _ = inversion_reverse_process(sd_pipe, xT=wts[skip], etas=eta, prompts=[prompt_tar], cfg_scales=[cfg_scale_tar], prog_bar=True, zs=zs[skip:])
    
    # vae decode image
    x0_dec = sd_pipe.vae.decode(1 / 0.18215 * w0).sample
    if x0_dec.dim()<4:
        x0_dec = x0_dec[None,:,:,:]
    img = image_grid(x0_dec)
    return img

# load pipelines
sd_model_id = "runwayml/stable-diffusion-v1-5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sd_pipe = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16).to(device)
sd_pipe.scheduler = DDIMScheduler.from_config(sd_model_id, subfolder = "scheduler")
sem_pipe = SemanticStableDiffusionPipeline.from_pretrained(sd_model_id, ,torch_dtype=torch.float16).to(device)


def edit(input_image, input_image_prompt='', target_prompt='', edit_prompt='', 
         negative_guidance = False, edit_warmup_steps=5,
         edit_guidance_scale=8, guidance_scale=15, skip=36, num_diffusion_steps=100,
         ):
    offsets=(0,0,0,0)
    x0 = load_512(input_image, *offsets, device)


    # invert
    wt, zs, wts = invert(x0 =x0 , prompt_src=input_image_prompt, num_diffusion_steps=num_diffusion_steps)
    latnets = wts[skip].expand(1, -1, -1, -1)

    eta = 1 
    #pure DDPM output
    pure_ddpm_out = sample(wt, zs, wts, prompt_tar=target_prompt, 
                           cfg_scale_tar=guidance_scale, skip=skip, 
                           eta = eta)
    
    editing_args = dict(
    editing_prompt = [edit_prompt],
    reverse_editing_direction = [negative_guidance],
    edit_warmup_steps=[edit_warmup_steps],
    edit_guidance_scale=[edit_guidance_scale], 
    edit_threshold=[.93],
    edit_momentum_scale=0.5, 
    edit_mom_beta=0.6 
  )
    sega_out = sem_pipe(prompt=target_prompt,eta=eta, latents=latnets, 
                        num_images_per_prompt=1,  
                        num_inference_steps=num_diffusion_steps, 
                        use_ddpm=True,  wts=wts, zs=zs[skip:], **editing_args)
    return pure_ddpm_out,sega_out.images[0]


# See the gradio docs for the types of inputs and outputs available
inputs = [
    gr.Image(label="input image", shape=(512, 512)),
    gr.Textbox(label="input prompt"),
    gr.Textbox(label="target prompt"),
    gr.Textbox(label="SEGA edit concept"),
    gr.Checkbox(label="SEGA negative_guidance"),
    gr.Slider(label="warmup steps", minimum=7, maximum=18, value=15),
    gr.Slider(label="edit guidance scale", minimum=0, maximum=15, value=3.5),
    gr.Slider(label="guidance scale", minimum=7, maximum=18, value=15),
    gr.Slider(label="skip", minimum=0, maximum=40, value=36),
    gr.Slider(label="num diffusion steps", minimum=0, maximum=300, value=100)
   
   
]
outputs = [gr.Image(label="DDPM"),gr.Image(label="DDPM+SEGA")]

# And the minimal interface
demo = gr.Interface(
    fn=edit,
    inputs=inputs,
    outputs=outputs,
)
demo.launch()  # debug=True allows you to see errors and output in Colab