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



def randomize_seed_fn(seed, randomize_seed):
    if randomize_seed:
        seed = random.randint(0, np.iinfo(np.int32).max)
    torch.manual_seed(seed)
    return seed


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
sd_model_id = "runwayml/stable-diffusion-v1-5"
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
            '+Schnauzer dog, -cat',
            5.5,
            1,
            'examples/ddpm_watercolor_painting_a_cat_sitting_next_to_a_mirror.png', 
            'examples/ddpm_sega_watercolor_painting_a_cat_sitting_next_to_a_mirror_plus_dog_minus_cat.png'
             ],
        [
            'examples/source_a_man_wearing_a_brown_hoodie_in_a_crowded_street.jpeg', 
            'a man wearing a brown hoodie in a crowded street',
            'a robot wearing a brown hoodie in a crowded street',
            100,
            36,
            15,
            '+painting',
            10,
            1,
            'examples/ddpm_a_robot_wearing_a_brown_hoodie_in_a_crowded_street.png', 
            'examples/ddpm_sega_painting_of_a_robot_wearing_a_brown_hoodie_in_a_crowded_street.png'
             ],
    [
            'examples/source_wall_with_framed_photos.jpeg', 
            '',
            '',
            100,
            36,
            15,
            '+pink drawings of muffins',
            10,
            1,
            'examples/ddpm_wall_with_framed_photos.png', 
            'examples/ddpm_sega_plus_pink_drawings_of_muffins.png'
             ],
    [
            'examples/source_an_empty_room_with_concrete_walls.jpg', 
            'an empty room with concrete walls',
            'glass walls',
            100,
            36,
            17,
            '+giant elephant',
            10,
            1,
            'examples/ddpm_glass_walls.png', 
            'examples/ddpm_sega_glass_walls_gian_elephant.png'
             ]]
    return case


def invert_and_reconstruct(
                    input_image, 
                    do_inversion,
                    seed, randomize_seed,
                    wts, zs, 
                    src_prompt ="", 
                    tar_prompt="", 
                    steps=100,
                    src_cfg_scale = 3.5,
                    skip=36,
                    tar_cfg_scale=15,
                    # neg_guidance=False,
                    
):

    
    x0 = load_512(input_image, device=device)
    
    if do_inversion or randomize_seed:
        # invert and retrieve noise maps and latent
        zs_tensor, wts_tensor = invert(x0 =x0 , prompt_src=src_prompt, num_diffusion_steps=steps, cfg_scale_src=src_cfg_scale)
        wts = gr.State(value=wts_tensor)
        zs = gr.State(value=zs_tensor)
        do_inversion = False

    output = sample(zs.value, wts.value, prompt_tar=tar_prompt, skip=skip, cfg_scale_tar=tar_cfg_scale)

    return output, wts, zs, do_inversion

def update_sega_concept_table(edit_concept, neg_guidance, concepts_table):
    if edit_concept:
        new_rows = concepts_table.value.append([edit_concept, neg_guidance])
        new_concepts_table = gr.DataFrame(value=new_rows)
        return new_concepts_table
    return concepts_table
    
def edit(input_image,
            wts, zs, 
            tar_prompt="", 
            steps=100,
            skip=36,
            tar_cfg_scale=15,
            edit_concept="",
            sega_edit_guidance=10,
            warm_up=None,
            # neg_guidance=False,

   ):
       
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
    edit_threshold=[.95]*num_concepts,
    edit_momentum_scale=0.5, 
    edit_mom_beta=0.6 
  )
    latnets = wts.value[skip].expand(1, -1, -1, -1)
    sega_out = sem_pipe(prompt=tar_prompt,eta=1, latents=latnets, guidance_scale = tar_cfg_scale,
                        num_images_per_prompt=1,  
                        num_inference_steps=steps, 
                        use_ddpm=True,  wts=wts.value, zs=zs.value[skip:], **editing_args)
    return sega_out.images[0]




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
with gr.Blocks(css='style.css') as demo:
    
    def reset_do_inversion():
        do_inversion = True
        return do_inversion


    gr.HTML(intro)
    wts = gr.State()
    zs = gr.State()
    do_inversion = gr.State(value=True)
         
    with gr.Row():
        input_image = gr.Image(label="Input Image", interactive=True)
        ddpm_edited_image = gr.Image(label=f"DDPM Reconstructed Image", interactive=False)
        sega_edited_image = gr.Image(label=f"DDPM + SEGA Edited Image", interactive=False)
        input_image.style(height=512, width=512)
        ddpm_edited_image.style(height=512, width=512)
        sega_edited_image.style(height=512, width=512)

    with gr.Row():
        tar_prompt = gr.Textbox(lines=1, label="Target Prompt", interactive=True, placeholder="")
        with gr.Accordion("SEGA Concepts", open=False, visible=True):
            with gr.Column(scale=1):
                edit_concept = gr.Textbox(lines=1, label="Enter SEGA Edit Concept", visible = True, interactive=True)
            with gr.Column(scale=1):
                neg_guidance = gr.Checkbox(label="Negative Guidance", value=False)
                submit_concept = gr.Button(label="Add Concept")
                concepts_table = gr.Dataframe(
                                headers=["Concepts", "Negative Guidance"],
                                datatype=["str", "bool"],
                                label="SEGA Concepts",
                            )
                
         
    with gr.Row():
        with gr.Column(scale=1, min_width=100):
            invert_button = gr.Button("Invert")
        with gr.Column(scale=1, min_width=100):
            edit_button = gr.Button("Edit")

    with gr.Accordion("Advanced Options", open=False):
         with gr.Tabs() as tabs:
                with gr.TabItem('SEGA Guidance', id=0):
                    with gr.Row().style(mobile_collapse=False, equal_height=True):
                        edit_1 = gr.Textbox(
                            label="Edit Prompt 1",
                            show_label=False,
                            max_lines=1,
                            placeholder="Enter your 1st edit prompt",
                        ).style(
                            border=(True, False, True, True),
                            rounded=(True, False, False, True),
                            container=False,
                        )
                        with gr.Group():
                            with gr.Row().style(mobile_collapse=False, equal_height=True):
                                rev_1 = gr.Checkbox(
                                    label='Negative Guidance')
                                warmup_1 = gr.Slider(label='Warmup', minimum=0, maximum=50, value=10, step=1, interactive=True)
                                scale_1 = gr.Slider(label='Scale', minimum=1, maximum=10, value=5, step=0.25, interactive=True)
                                threshold_1 = gr.Slider(label='Threshold', minimum=0.5, maximum=0.99, value=0.95, steps=0.01, interactive=True)
                    with gr.TabItem('DDPM Guidance', id=0):
                        with gr.Row():
                            with gr.Column():
                                src_prompt = gr.Textbox(lines=1, label="Source Prompt", interactive=True, placeholder="")
                                steps = gr.Number(value=100, precision=0, label="Num Diffusion Steps", interactive=True)
                                src_cfg_scale = gr.Number(value=3.5, label=f"Source Guidance Scale", interactive=True)
                                seed = gr.Number(value=0, precision=0, label="Seed", interactive=True)
                                randomize_seed = gr.Checkbox(label='Randomize seed', value=False)
                            with gr.Column():    
                                skip = gr.Slider(minimum=0, maximum=40, value=36, label="Skip Steps", interactive=True)
                                tar_cfg_scale = gr.Slider(minimum=7, maximum=18,value=15, label=f"Guidance Scale", interactive=True)  
                                # sega_edit_guidance = gr.Slider(value=10, label=f"SEGA Edit Guidance Scale", interactive=True)
                                # warm_up = gr.Textbox(label=f"SEGA Warm-up Steps", interactive=True, placeholder="type #warm-up steps for each concpets (e.g. 2,7,5...")

            
            # neg_guidance = gr.Checkbox(label="SEGA Negative Guidance")
          

    # gr.Markdown(help_text)


    submit_concept.click(
        fn = update_sega_concept_table,
        inputs = [edit_concept, neg_guidance, concepts_table],
        outputs = [concepts_table]
    )
    
    invert_button.click(
        fn = randomize_seed_fn,
        inputs = [seed, randomize_seed],
        outputs = [seed], 
        queue = False).then(
        fn=invert_and_reconstruct,
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
        outputs=[ddpm_edited_image, wts, zs, do_inversion],
    )

    edit_button.click(
        fn=edit,
        inputs=[input_image, 
                wts, zs, 
                tar_prompt, 
                steps,
                skip,
                tar_cfg_scale,
                edit_concept,
                sega_edit_guidance,
                warm_up,
                # neg_guidance,

        ],
        outputs=[sega_edited_image],
        
    )

    input_image.change(
        fn = reset_do_inversion,
        outputs = [do_inversion]
    )

    gr.Examples(
        label='Examples', 
        examples=get_example(), 
        inputs=[input_image, src_prompt, tar_prompt, steps,
                    # src_cfg_scale,
                    skip,
                    tar_cfg_scale,
                    edit_concept,
                    sega_edit_guidance,
                    warm_up,
                    # neg_guidance,
                    ddpm_edited_image, sega_edited_image
               ],
        outputs=[ddpm_edited_image, sega_edited_image],
        # fn=edit,
        # cache_examples=True
    )



demo.queue()
demo.launch(share=False)



