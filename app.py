import gradio as gr
import torch
import numpy as np
import requests
import random
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
  return zs, wts



def sample(zs, wts, prompt_tar="", cfg_scale_tar=15, skip=36, eta = 1):

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
sd_model_id = "stabilityai/stable-diffusion-2-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sd_pipe = StableDiffusionPipeline.from_pretrained(sd_model_id).to(device)
sd_pipe.scheduler = DDIMScheduler.from_config(sd_model_id, subfolder = "scheduler")
sem_pipe = SemanticStableDiffusionPipeline.from_pretrained(sd_model_id).to(device)


def get_example():
    case = [
        [
            'examples/source_a_cat_sitting_next_to_a_mirror.jpeg', 
            'a cat sitting next to a mirror',
            'watercolor painting of a cat sitting next to a mirror',
            100,
            36,
            15,
            'Schnauzer dog', 'cat',
            5.5,
            1,
            'examples/ddpm_sega_watercolor_painting_a_cat_sitting_next_to_a_mirror_plus_dog_minus_cat.png'
             ],
        [
            'examples/source_a_man_wearing_a_brown_hoodie_in_a_crowded_street.jpeg', 
            'a man wearing a brown hoodie in a crowded street',
            'a robot wearing a brown hoodie in a crowded street',
            100,
            36,
            15,
            'painting','',
            10,
            1,
            'examples/ddpm_sega_painting_of_a_robot_wearing_a_brown_hoodie_in_a_crowded_street.png'
             ],
    [
            'examples/source_wall_with_framed_photos.jpeg', 
            '',
            '',
            100,
            36,
            15,
            'pink drawings of muffins','',
            10,
            1,
            'examples/ddpm_sega_plus_pink_drawings_of_muffins.png'
             ],
    [
            'examples/source_an_empty_room_with_concrete_walls.jpg', 
            'an empty room with concrete walls',
            'glass walls',
            100,
            36,
            17,
            'giant elephant','',
            10,
            1,
            'examples/ddpm_sega_glass_walls_gian_elephant.png'
             ]]
    return case

def randomize_seed_fn(seed, randomize_seed):
    if randomize_seed:
        seed = random.randint(0, np.iinfo(np.int32).max)
    torch.manual_seed(seed)
    return seed


    

def reconstruct(tar_prompt, 
                tar_cfg_scale, 
                skip, 
                wts, zs, 
                # do_reconstruction, 
                # reconstruction
               ):
    

    # if do_reconstruction:
    reconstruction = sample(zs.value, wts.value, prompt_tar=tar_prompt, skip=skip, cfg_scale_tar=tar_cfg_scale)
    return reconstruction

    
def load_and_invert(
                    input_image, 
                    do_inversion,
                    seed, randomize_seed,
                    wts, zs, 
                    src_prompt ="", 
                    tar_prompt="", 
                    steps=100,
                    src_cfg_scale = 3.5,
                    skip=36,
                    tar_cfg_scale=15
                    
):

    
    x0 = load_512(input_image, device=device)
    
    if do_inversion or randomize_seed:
        # invert and retrieve noise maps and latent
        zs_tensor, wts_tensor = invert(x0 =x0 , prompt_src=src_prompt, num_diffusion_steps=steps, cfg_scale_src=src_cfg_scale)
        wts = gr.State(value=wts_tensor)
        zs = gr.State(value=zs_tensor)
        do_inversion = False
        
    return wts, zs, do_inversion

    
def edit(input_image,
            wts, zs, 
            tar_prompt, 
            steps,
            skip,
            tar_cfg_scale,
            edit_concept_1,edit_concept_2,edit_concept_3,
            guidnace_scale_1,guidnace_scale_2,guidnace_scale_3,
            warmup_1, warmup_2, warmup_3,
            neg_guidance_1, neg_guidance_2, neg_guidance_3,
            threshold_1, threshold_2, threshold_3

   ):
       
    # SEGA
    # parse concepts and neg guidance 

    
    
    editing_args = dict(
    editing_prompt = [edit_concept_1,edit_concept_2,edit_concept_3],
    reverse_editing_direction = [ neg_guidance_1, neg_guidance_2, neg_guidance_3,],
    edit_warmup_steps=[warmup_1, warmup_2, warmup_3,],
    edit_guidance_scale=[guidnace_scale_1,guidnace_scale_2,guidnace_scale_3], 
    edit_threshold=[threshold_1, threshold_2, threshold_3],
    edit_momentum_scale=0.3, 
    edit_mom_beta=0.6,
    eta=1,
  )
    latnets = wts.value[skip].expand(1, -1, -1, -1)
    sega_out = sem_pipe(prompt=tar_prompt, latents=latnets, guidance_scale = tar_cfg_scale,
                        num_images_per_prompt=1,  
                        num_inference_steps=steps, 
                        use_ddpm=True,  wts=wts.value, zs=zs.value[skip:], **editing_args)
    return sega_out.images[0], True




########
# demo #
########
                        
intro = """
<h1 style="font-weight: 1400; text-align: center; margin-bottom: 7px;">
   Edit Friendly DDPM X Semantic Guidance
</h1>
<p style="font-size: 0.9rem; text-align: center; margin: 0rem; line-height: 1.2em; margin-top:1em">
<a href="https://arxiv.org/abs/2301.12247" style="text-decoration: underline;" target="_blank">An Edit Friendly DDPM Noise Space:
Inversion and Manipulations </a> X
<a href="https://arxiv.org/abs/2301.12247" style="text-decoration: underline;" target="_blank">SEGA: Instructing Diffusion using Semantic Dimensions</a>
<p/>
<p style="font-size: 0.9rem; margin: 0rem; line-height: 1.2em; margin-top:1em">
For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings.
<a href="https://huggingface.co/spaces/LinoyTsaban/ddpm_sega?duplicate=true">
<img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
<p/>"""

help_text = """
- **Getting Started - edit images with DDPM X SEGA:**
    
    The are 3 general setting options you can play with - 
    
    1. **Pure DDPM Edit -** Describe the desired edited output image in detail
    2. **Pure SEGA Edit -** Keep the target prompt empty ***or*** with a description of the original image and add editing concepts for Semantic Gudiance editing 
    3. **Combined -** Describe the desired edited output image in detail and add additional SEGA editing concepts on top 
- **Getting Started - Tips**
    
    While the best approach depends on your editing objective and source image,  we can layout a few guiding tips to use as a starting point -
    
    1. **DDPM** is usually more suited for scene/style changes and major subject changes (for example ) while **SEGA** allows for more fine grained control, changes are more delicate, more suited for adding details (for example facial expressions and attributes, subtle style modifications, object adding/removing)
    2. The more you describe the scene in the target prompt (both the parts and details you wish to keep the same and those you wish to change), the better the result 
    3. **Combining DDPM Edit with SEGA -** 
    Try dividing your editing objective to more significant scene/style/subject changes and detail adding/removing and more moderate changes. Then describe the major changes in a detailed target prompt and add the more fine grained details as SEGA concepts. 
    4. **Reconstruction:** Using an empty source prompt + target prompt will lead to a perfect reconstruction
- **Fidelity vs creativity**:
    
    Bigger values → more fidelity, smaller values → more creativity
    
    1. `Skip Steps` 
    2. `Warmup` (SEGA)
    3. `Threshold`  (SEGA)
    
    Bigger values → more creativity, smaller values → more fidelity
    
    1. `Guidance Scale`
    2. `Concept Guidance Scale` (SEGA)
"""

with gr.Blocks(css='style.css') as demo:
    
    def add_concept(sega_concepts_counter):
      if sega_concepts_counter == 1:
        return row2.update(visible=True), row3.update(visible=False), plus.update(visible=True), 2
      else:
        return row2.update(visible=True), row3.update(visible=True), plus.update(visible=False), 3

    def show_reconstruction_button():
        return reconstruct_button.update(visible=True)
            
    def hide_reconstruction_button():
        return reconstruct_button.update(visible=False)

    def show_reconstruction():
        return ddpm_edited_image.update(visible=True)
        
        
    def reset_do_inversion():
        do_inversion = True
        return do_inversion

        
    gr.HTML(intro)
    wts = gr.State()
    zs = gr.State()
    do_inversion = gr.State(value=True)
    sega_concepts_counter = gr.State(1)

    

    
    with gr.Row():
        input_image = gr.Image(label="Input Image", interactive=True)
        ddpm_edited_image = gr.Image(label=f"DDPM Reconstructed Image", interactive=False, visible=False)
        sega_edited_image = gr.Image(label=f"DDPM + SEGA Edited Image", interactive=False)
        input_image.style(height=365, width=365)
        ddpm_edited_image.style(height=365, width=365)
        sega_edited_image.style(height=365, width=365)

    with gr.Tabs() as tabs:
          with gr.TabItem('1. Describe the desired output', id=0):
            with gr.Row().style(mobile_collapse=False, equal_height=True):
              tar_prompt = gr.Textbox(
                                label="Edit Concept",
                                show_label=False,
                                max_lines=1,
                                placeholder="Enter your 1st edit prompt",
                            )
          with gr.TabItem('2. Add SEGA edit concepts', id=1):
              # 1st SEGA concept
              with gr.Row().style(mobile_collapse=False, equal_height=True):
                  neg_guidance_1 = gr.Checkbox(
                      label='Negative Guidance')
                  warmup_1 = gr.Slider(label='Warmup', minimum=0, maximum=50, value=1, step=1, interactive=True)
                  guidnace_scale_1 = gr.Slider(label='Concept Guidance Scale', minimum=1, maximum=15, value=5, step=0.25, interactive=True)
                  threshold_1 = gr.Slider(label='Threshold', minimum=0.5, maximum=0.99, value=0.95, steps=0.01, interactive=True)
                  edit_concept_1 = gr.Textbox(
                                  label="Edit Concept",
                                  show_label=False,
                                  max_lines=1,
                                  placeholder="Enter your 1st edit prompt",
                              )
                  
              # 2nd SEGA concept
              with gr.Row(visible=False) as row2:
                  neg_guidance_2 = gr.Checkbox(
                      label='Negative Guidance',visible=True)
                  warmup_2 = gr.Slider(label='Warmup', minimum=0, maximum=50, value=1, step=1, visible=True,interactive=True)
                  guidnace_scale_2 = gr.Slider(label='Concept Guidance Scale', minimum=1, maximum=15, value=5, step=0.25,visible=True, interactive=True)
                  threshold_2 = gr.Slider(label='Threshold', minimum=0.5, maximum=0.99, value=0.95, steps=0.01,visible=True, interactive=True)
                  edit_concept_2 = gr.Textbox(
                                  label="Edit Concept",
                                  show_label=False,visible=True,
                                  max_lines=1,
                                  placeholder="Enter your 2st edit prompt",
                              )
              # 3rd SEGA concept
              with gr.Row(visible=False) as row3:
                  neg_guidance_3 = gr.Checkbox(
                      label='Negative Guidance',visible=True)
                  warmup_3 = gr.Slider(label='Warmup', minimum=0, maximum=50, value=1, step=1, visible=True,interactive=True)
                  guidnace_scale_3 = gr.Slider(label='Concept Guidance Scale', minimum=1, maximum=15, value=5, step=0.25,visible=True, interactive=True)
                  threshold_3 = gr.Slider(label='Threshold', minimum=0.5, maximum=0.99, value=0.95, steps=0.01,visible=True, interactive=True)
                  edit_concept_3 = gr.Textbox(
                                  label="Edit Concept",
                                  show_label=False,visible=True,
                                  max_lines=1,
                                  placeholder="Enter your 3rd edit prompt",
                              )
              
              with gr.Row().style(mobile_collapse=False, equal_height=True):
                add_concept_button = gr.Button("+")

                      
    with gr.Row():
        run_button = gr.Button("Run")
        reconstruct_button = gr.Button("Show Reconstruction", visible=False)

    with gr.Accordion("Advanced Options", open=False):
            with gr.Row():
                with gr.Column():
                    src_prompt = gr.Textbox(lines=1, label="Source Prompt", interactive=True, placeholder="")
                    steps = gr.Number(value=100, precision=0, label="Num Diffusion Steps", interactive=True)
                    src_cfg_scale = gr.Number(value=3.5, label=f"Source Guidance Scale", interactive=True)
                    seed = gr.Number(value=0, precision=0, label="Seed", interactive=True)
                    randomize_seed = gr.Checkbox(label='Randomize seed', value=False)
                with gr.Column():    
                    skip = gr.Slider(minimum=0, maximum=60, value=36, label="Skip Steps", interactive=True)
                    tar_cfg_scale = gr.Slider(minimum=7, maximum=18,value=15, label=f"Guidance Scale", interactive=True)  



    with gr.Accordion("Help", open=False):
        gr.Markdown(help_text)
    

    
    add_concept_button.click(fn = add_concept, inputs=sega_concepts_counter,
               outputs= [row2, row3, add_concept_button, sega_concepts_counter], queue = False)

    
    run_button.click(
        fn = randomize_seed_fn,
        inputs = [seed, randomize_seed],
        outputs = [seed], 
        queue = False).then(
        fn=load_and_invert,
        inputs=[input_image, 
                do_inversion,
                seed, randomize_seed,
                wts, zs, 
                src_prompt, 
                tar_prompt, 
                steps,
                src_cfg_scale,
                skip,
                tar_cfg_scale         
        ],
        outputs=[wts, zs, do_inversion],
    ).success(
        fn=edit,
        inputs=[input_image, 
                wts, zs, 
                tar_prompt, 
                steps,
                skip,
                tar_cfg_scale,
                edit_concept_1,edit_concept_2,edit_concept_3,
                guidnace_scale_1,guidnace_scale_2,guidnace_scale_3,
                warmup_1, warmup_2, warmup_3,
                neg_guidance_1, neg_guidance_2, neg_guidance_3,
                threshold_1, threshold_2, threshold_3

        ],
        outputs=[sega_edited_image, is_show_reconstruction],     
    ).success( 
        fn = show_reconstruction_button,
        outputs = [reconstruct_button]
    )

    reconstruct_button.click(
        fn = show_reconstruction,
        outputs = [ddpm_edited_image]
    ).then(
        fn = reconstruct,
        inputs = [tar_prompt, 
                  tar_cfg_scale, 
                  skip, 
                  wts, zs],
        outputs = [ddpm_edited_image]
    )


    # Automatically start inverting upon input_image change
    input_image.change(
        fn = reset_do_inversion,
        outputs = [do_inversion], 
        queue = False).then(
        fn = hide_reconstruction_button, 
           outputs = [reconstruct_button], 
        queue=False).then(
        fn=load_and_invert,
        inputs=[input_image, 
                do_inversion,
                seed, randomize_seed,
                wts, zs, 
                src_prompt, 
                tar_prompt, 
                steps,
                src_cfg_scale,
                skip,
                tar_cfg_scale,          
        ],
        # outputs=[ddpm_edited_image, wts, zs, do_inversion],
        outputs=[wts, zs, do_inversion],
    )

    # Repeat inversion when these params are changed:
    src_prompt.change(
        fn = reset_do_inversion,
        outputs = [do_inversion], queue = False
    )
    steps.change(fn = reset_do_inversion,
        outputs = [do_inversion], queue = False)

    src_cfg_scale.change(fn = reset_do_inversion,
        outputs = [do_inversion], queue = False)
    

    gr.Examples(
        label='Examples', 
        examples=get_example(), 
        inputs=[input_image, src_prompt, tar_prompt, steps,
                    # src_cfg_scale,
                    skip,
                    tar_cfg_scale,
                    edit_concept_1,
                    edit_concept_2,
                    guidnace_scale_1,
                    warmup_1,
                    # neg_guidance,
                    sega_edited_image
               ],
        outputs=[sega_edited_image],
        # fn=edit,
        # cache_examples=True
    )



demo.queue()
demo.launch(share=False)



