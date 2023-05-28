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
import re

def invert(x0, prompt_src="", num_diffusion_steps=100, cfg_scale_src = 3.5, eta = 1):

  #  inverts a real image according to Algorihm 1 in https://arxiv.org/pdf/2304.06140.pdf, 
  #  based on the code in https://github.com/inbarhub/DDPM_inversion
   
  #  returns wt, zs, wts:
  #  wt - inverted latent
  #  wts - intermediate inverted latents
  #  zs - noise maps

  sd_pipe.scheduler.set_timesteps(num_diffusion_steps)

  # vae encode image
  with autocast("cuda"), inference_mode():
      w0 = (sd_pipe.vae.encode(x0).latent_dist.mode() * 0.18215).float()

  # find Zs and wts - forward process
  wt, zs, wts = inversion_forward_process(sd_pipe, w0, etas=eta, prompt=prompt_src, cfg_scale=cfg_scale_src, prog_bar=True, num_inference_steps=num_diffusion_steps)
  return wt, zs, wts



def sample(wt, zs, wts, prompt_tar="", cfg_scale_tar=15, skip=36, eta = 1):

    # reverse process (via Zs and wT)
    w0, _ = inversion_reverse_process(sd_pipe, xT=wts[skip], etas=eta, prompts=[prompt_tar], cfg_scales=[cfg_scale_tar], prog_bar=True, zs=zs[skip:])
    
    # vae decode image
    with autocast("cuda"), inference_mode():
        x0_dec = sd_pipe.vae.decode(1 / 0.18215 * w0).sample
    if x0_dec.dim()<4:
        x0_dec = x0_dec[None,:,:,:]
    img = image_grid(x0_dec)
    return img

# load pipelines
sd_model_id = "runwayml/stable-diffusion-v1-5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sd_pipe = StableDiffusionPipeline.from_pretrained(sd_model_id).to(device)
sd_pipe.scheduler = DDIMScheduler.from_config(sd_model_id, subfolder = "scheduler")
sem_pipe = SemanticStableDiffusionPipeline.from_pretrained(sd_model_id).to(device)

cache_examples = True

def get_example():
    case = [
        [
            'examples/source_a_man_wearing_a_brown_hoodie_in_a_crowded_street.jpeg', 
            'a man wearing a brown hoodie in a crowded street',
            'a robot wearing a brown hoodie in a crowded street',
            '+painting',
            '1'
            'examples/ddpm_a_robot_wearing_a_brown_hoodie_in_a_crowded_street.png', 
            'examples/ddpm_sega_painting_of_a_robot_wearing_a_brown_hoodie_in_a_crowded_street.png'
             ],
    [
            'examples/source_wall_with_framed_photos.jpeg', 
            '',
            '',
            '+pink drawings of muffins',
            '1'
            'examples/ddpm_wall_with_framed_photos.png', 
            'examples/ddpm_sega_plus_pink_drawings_of_muffins.png'
             ]]
    return case
    
def edit(input_image, 
                    src_prompt ="", 
                    tar_prompt="", 
                    steps=100,
                    # src_cfg_scale,
                    skip=36,
                    tar_cfg_scale=15,
                    edit_concept="",
                    sega_edit_guidance=0,
                    warm_up=None,
                    # neg_guidance=False,
                    left = 0,
                    right = 0,
                    top = 0,
                    bottom = 0):

    # offsets=(0,0,0,0)
    x0 = load_512(input_image, left,right, top, bottom, device)


    # invert
    # wt, zs, wts = invert(x0 =x0 , prompt_src=src_prompt, num_diffusion_steps=steps, cfg_scale_src=src_cfg_scale)
    wt, zs, wts = invert(x0 =x0 , prompt_src=src_prompt, num_diffusion_steps=steps)

    latnets = wts[skip].expand(1, -1, -1, -1)


    #pure DDPM output
    pure_ddpm_out = sample(wt, zs, wts, prompt_tar=tar_prompt, 
                           cfg_scale_tar=tar_cfg_scale, skip=skip)

    if not edit_concept or not sega_edit_guidance:
        return pure_ddpm_out, pure_ddpm_out
        
    # SEGA
    # parse concepts and neg guidance 
    edit_concepts = edit_concept.split(",")
    num_concepts = len(edit_concepts)
    neg_guidance =[] 
    for edit_concept in edit_concepts:
        edit_concept=edit_concept.strip(" ")
        if edit_concept.startswith("-"):
            neg_guidance.append(True)
        else:
            neg_guidance.append(False)
    edit_concepts = [concept.strip("+|-") for concept in edit_concepts]
                        
    # parse warm-up steps
    default_warm_up_steps = [1]*num_concepts
    if warm_up:
        digit_pattern = re.compile(r"^\d+$")
        warm_up_steps_str = warm_up.split(",")
        for i,num_steps in enumerate(warm_up_steps_str[:num_concepts]):
            if not digit_pattern.match(num_steps):
                raise gr.Error("Invalid value for warm-up steps, using 1 instead")
            else:
                default_warm_up_steps[i] = int(num_steps)
        
        
    editing_args = dict(
    editing_prompt = edit_concepts,
    reverse_editing_direction = neg_guidance,
    edit_warmup_steps=default_warm_up_steps,
    edit_guidance_scale=[sega_edit_guidance]*num_concepts, 
    edit_threshold=[.93]*num_concepts,
    edit_momentum_scale=0.5, 
    edit_mom_beta=0.6 
  )
    sega_out = sem_pipe(prompt=tar_prompt,eta=1, latents=latnets, guidance_scale = tar_cfg_scale,
                        num_images_per_prompt=1,  
                        num_inference_steps=steps, 
                        use_ddpm=True,  wts=wts, zs=zs[skip:], **editing_args)
    return pure_ddpm_out,sega_out.images[0]

########
# demo #
########
                        
intro = """
<h1 style="font-weight: 1400; text-align: center; margin-bottom: 7px;">
   Edit Friendly DDPM X Semantic Guidance: Editing Real Images
</h1>
<p style="font-size: 0.9rem; margin: 0rem; line-height: 1.2em; margin-top:1em">
For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings.
<a href="https://huggingface.co/spaces/LinoyTsaban/ddpm_sega?duplicate=true">
<img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
<p/>"""
with gr.Blocks() as demo:
    gr.HTML(intro)
   
    with gr.Row():
        src_prompt = gr.Textbox(lines=1, label="Source Prompt", interactive=True)
        tar_prompt = gr.Textbox(lines=1, label="Target Prompt", interactive=True)
        edit_concept = gr.Textbox(lines=1, label="SEGA Edit Concepts", interactive=True)
         
    with gr.Row():
        input_image = gr.Image(label="Input Image", interactive=True)
        ddpm_edited_image = gr.Image(label=f"DDPM Reconstructed Image", interactive=False)
        sega_edited_image = gr.Image(label=f"DDPM + SEGA Edited Image", interactive=False)
        input_image.style(height=512, width=512)
        ddpm_edited_image.style(height=512, width=512)
        sega_edited_image.style(height=512, width=512)

    with gr.Row():
        with gr.Column(scale=1, min_width=100):
            generate_button = gr.Button("Run")

    with gr.Accordion("Advanced Options", open=False):
        with gr.Row():
            with gr.Column():
                #inversion
                steps = gr.Number(value=100, precision=0, label="Num Diffusion Steps", interactive=True)
                # src_cfg_scale = gr.Number(value=3.5, label=f"Source CFG", interactive=True)
      
                # reconstruction
                skip = gr.Slider(minimum=0, maximum=40, value=36, precision=0, label="Skip Steps", interactive=True)
                tar_cfg_scale = gr.Slider(minimum=7, maximum=18,value=15, label=f"Guidance Scale", interactive=True)
                
            with gr.Column():    
                sega_edit_guidance = gr.Slider(value=10, label=f"SEGA Edit Guidance Scale", interactive=True)
                warm_up = gr.Textbox(label=f"SEGA Warm-up Steps", interactive=True)

            #shift
            with gr.Column():
                left = gr.Number(value=0, precision=0, label="Left Shift", interactive=True)
                right = gr.Number(value=0, precision=0, label="Right Shift", interactive=True)
            with gr.Column():
                top = gr.Number(value=0, precision=0, label="Top Shift", interactive=True)
                bottom = gr.Number(value=0, precision=0, label="Bottom Shift", interactive=True)

            
            # neg_guidance = gr.Checkbox(label="SEGA Negative Guidance")
          

    # gr.Markdown(help_text)

    generate_button.click(
        fn=edit,
        inputs=[input_image, 
                    src_prompt, 
                    tar_prompt, 
                    steps,
                    # src_cfg_scale,
                    skip,
                    tar_cfg_scale,
                    edit_concept,
                    sega_edit_guidance,
                    warm_up,
                    # neg_guidance,
                    left,
                    right,
                    top,
                    bottom
        ],
        outputs=[ddpm_edited_image, sega_edited_image],
    )

    gr.Examples(
        label='Examples', 
        examples=get_example(), 
        inputs=[input_image, src_prompt, tar_prompt, edit_concept, warm_up, ddpm_edited_image, sega_edited_image],
        outputs=[ddpm_edited_image, sega_edited_image])



demo.queue()
demo.launch(share=False)



