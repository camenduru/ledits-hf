import gradio as gr
import torch
import numpy as np
import requests
import random
from io import BytesIO
from utils import *
from constants import *
from inversion_utils import *
from modified_pipeline_semantic_stable_diffusion import SemanticStableDiffusionPipeline
from torch import autocast, inference_mode
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
from transformers import AutoProcessor, BlipForConditionalGeneration

# load pipelines
sd_model_id = "stabilityai/stable-diffusion-2-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sd_pipe = StableDiffusionPipeline.from_pretrained(sd_model_id).to(device)
sd_pipe.scheduler = DDIMScheduler.from_config(sd_model_id, subfolder = "scheduler")
sem_pipe = SemanticStableDiffusionPipeline.from_pretrained(sd_model_id).to(device)
blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)



## IMAGE CPATIONING ##
def caption_image(input_image):

  inputs = blip_processor(images=input_image, return_tensors="pt").to(device)
  pixel_values = inputs.pixel_values

  generated_ids = blip_model.generate(pixel_values=pixel_values, max_length=50)
  generated_caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
  return generated_caption


## DDPM INVERSION AND SAMPLING ##
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


def reconstruct(tar_prompt,
                tar_cfg_scale,
                skip,
                wts, zs,
                do_reconstruction,
                reconstruction,
                reconstruct_button
               ):

    if reconstruct_button == "Hide Reconstruction":
      return reconstruction.value, reconstruction, ddpm_edited_image.update(visible=False), do_reconstruction, "Show Reconstruction"

    else:
      if do_reconstruction:
          reconstruction_img = sample(zs.value, wts.value, prompt_tar=tar_prompt, skip=skip, cfg_scale_tar=tar_cfg_scale)
          reconstruction = gr.State(value=reconstruction_img)
          do_reconstruction = False
      return reconstruction.value, reconstruction, ddpm_edited_image.update(visible=True), do_reconstruction, "Hide Reconstruction"


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
                    tar_cfg_scale=15,
                    progress=gr.Progress(track_tqdm=True)

):


    x0 = load_512(input_image, device=device)

    if do_inversion or randomize_seed:
        # invert and retrieve noise maps and latent
        zs_tensor, wts_tensor = invert(x0 =x0 , prompt_src=src_prompt, num_diffusion_steps=steps, cfg_scale_src=src_cfg_scale)
        wts = gr.State(value=wts_tensor)
        zs = gr.State(value=zs_tensor)
        do_inversion = False

    return wts, zs, do_inversion, inversion_progress.update(visible=False)

## SEGA ##

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
            threshold_1, threshold_2, threshold_3):


    editing_args = dict(
    editing_prompt = [edit_concept_1,edit_concept_2,edit_concept_3],
    reverse_editing_direction = [ neg_guidance_1, neg_guidance_2, neg_guidance_3,],
    edit_warmup_steps=[warmup_1, warmup_2, warmup_3,],
    edit_guidance_scale=[guidnace_scale_1,guidnace_scale_2,guidnace_scale_3],
    edit_threshold=[threshold_1, threshold_2, threshold_3],
    edit_momentum_scale=0.3,
    edit_mom_beta=0.6,
    eta=1,)

    latnets = wts.value[skip].expand(1, -1, -1, -1)
    sega_out = sem_pipe(prompt=tar_prompt, latents=latnets, guidance_scale = tar_cfg_scale,
                        num_images_per_prompt=1,
                        num_inference_steps=steps,
                        use_ddpm=True,  wts=wts.value, zs=zs.value[skip:], **editing_args)

    return sega_out.images[0], reconstruct_button.update(visible=True)
                



def randomize_seed_fn(seed, randomize_seed):
    if randomize_seed:
        seed = random.randint(0, np.iinfo(np.int32).max)
    torch.manual_seed(seed)
    return seed

    
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



########
# demo #
########

intro = """
<h1 style="font-weight: 1400; text-align: center; margin-bottom: 7px;">
   Edit Friendly DDPM X Semantic Guidance
</h1>
<p style="font-size: 0.9rem; text-align: center; margin: 0rem; line-height: 1.2em; margin-top:1em">
<a href="https://arxiv.org/abs/2304.06140" style="text-decoration: underline;" target="_blank">An Edit Friendly DDPM Noise Space:
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

with gr.Blocks(css=css) as demo:

    def add_concept(sega_concepts_counter):
      if sega_concepts_counter == 1:
        return row2.update(visible=True), row2_advanced.update(visible=True), row3.update(visible=False), row3_advanced.update(visible=False), add_concept_button.update(visible=True), 2
      else:
        return row2.update(visible=True), row2_advanced.update(visible=True), row3.update(visible=True), row3_advanced.update(visible=False), add_concept_button.update(visible=False), 3

    def update_display_concept_1(add_1, edit_concept_1):
      if add_1 == 'Add':
        return box1.update(visible=True), edit_concept_1, concept_1.update(visible=True), edit_concept_1, guidnace_scale_1.update(visible=True), "Clear"
      else: # remove
        return box1.update(visible=False),"", concept_1.update(visible=False), "", guidnace_scale_1.update(visible=False), "Add"

    def update_display_concept_2(add_2, edit_concept_2):
      if add_2 == 'Add':
        return box2.update(visible=True), edit_concept_2, concept_2.update(visible=True),edit_concept_2, guidnace_scale_2.update(visible=True), "Clear"
      else: # remove
        return box2.update(visible=False),"", concept_2.update(visible=False), "", guidnace_scale_2.update(visible=False), "Add"

    def update_display_concept_3(add_3, edit_concept_3):
      if add_3 == 'Add':
        return box3.update(visible=True), edit_concept_3, concept_3.update(visible=True), edit_concept_3, guidnace_scale_3.update(visible=True), "Clear"
      else: # remove
        return box3.update(visible=False), "", concept_3.update(visible=False), "", guidnace_scale_3.update(visible=False), "Add"

    def display_editing_options(run_button, clear_button, sega_tab):
      return run_button.update(visible=True), clear_button.update(visible=True), sega_tab.update(visible=True)

    # def update_gallery_display(prev_output_image, sega_edited_image):
    #   if prev_output_image is None:
    #     return sega_edited_image, gallery.update(visible=True), sega_edited_image
    #   else:
    #     return prev_output_image, gallery.update(visible=True), sega_edited_image



    def reset_do_inversion():
        do_inversion = True
        return do_inversion

    def reset_do_reconstruction():
      do_reconstruction = True
      return  do_reconstruction

    def update_inversion_progress_visibility(input_image, do_inversion):
      if do_inversion and not input_image is None:
          return inversion_progress.update(visible=True)
      else:
        return inversion_progress.update(visible=False)

    def undo():
      return


    gr.HTML(intro)
    wts = gr.State()
    zs = gr.State()
    reconstruction = gr.State()
    do_inversion = gr.State(value=True)
    do_reconstruction = gr.State(value=True)
    sega_concepts_counter = gr.State(1)


    #Undo / gallery
    # prev_wts = gr.State()
    # prev_zs = gr.State()
    # prev_src_prompt = gr.State()
    # prev_tar_prompt = gr.State()
    # prev_tar_guidance_scale = gr.State()
    # prev_src_guidance_scale = gr.State()
    # prev_num_steps = gr.State()
    # prev_skip = gr.State()
    # prev_output_image = gr.Image(visible=False)
    # prev_input_image = gr.State()



    with gr.Row():
        input_image = gr.Image(label="Input Image", interactive=True)
        ddpm_edited_image = gr.Image(label=f"DDPM Reconstructed Image", interactive=False, visible=False)
        sega_edited_image = gr.Image(label=f"DDPM + SEGA Edited Image", interactive=False)
        input_image.style(height=365, width=365)
        ddpm_edited_image.style(height=365, width=365)
        sega_edited_image.style(height=365, width=365)

    with gr.Row():
      with gr.Box(visible=False) as box1:
        concept_1 = gr.Button(visible=False)
        guidnace_scale_1 = gr.Slider(label='Concept Guidance Scale', minimum=1, maximum=30,
                                                  value=DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,
                                                  step=0.5, interactive=True,visible=False)
      with gr.Box(visible=False) as box2:
       concept_2 = gr.Button(visible=False)
       guidnace_scale_2 = gr.Slider(label='Concept Guidance Scale', minimum=1, maximum=30,
                                                value=DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,
                                                step=0.5, interactive=True,visible=False)
      with gr.Box(visible=False) as box3:
       concept_3 = gr.Button(visible=False)
       guidnace_scale_3 = gr.Slider(label='Concept Guidance Scale', minimum=1, maximum=30,
                                                value=DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,
                                                step=0.5, interactive=True,visible=False)



    # with gr.Row():
    #   gallery =  gr.Gallery(label = "History", visible = True).style(
    #                                  columns=1,rows=1,
    #                                  object_fit='contain')



    with gr.Row():
        inversion_progress = gr.Textbox(visible=False, label="Inversion progress")

    with gr.Tabs() as tabs:
          with gr.TabItem('1. Describe the desired output', id=0):
            with gr.Row().style(mobile_collapse=False, equal_height=True):
                tar_prompt = gr.Textbox(
                                label="Edit Concept",
                                show_label=False,
                                max_lines=1,
                                placeholder="Enter your 1st edit prompt",  elem_classes="feedback"
                            )
                caption_button = gr.Button("Caption Image")
          with gr.TabItem('2. Add SEGA edit concepts', id=1):
              # 1st SEGA concept
              with gr.Row().style(mobile_collapse=False, equal_height=True):
                  edit_concept_1 = gr.Textbox(
                                  label="Edit Concept",
                                  show_label=False,
                                  max_lines=1,
                                  placeholder="Enter your 1st edit prompt",
                              )
                  neg_guidance_1 = gr.Checkbox(
                      label='Negative Guidance')

                  # guidnace_scale_1 = gr.Slider(label='Concept Guidance Scale', minimum=1, maximum=30,
                  #                              value=DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,
                  #                              step=0.5, interactive=True)


                  add_1 = gr.Button('Add')

              # 2nd SEGA concept
              with gr.Row(visible=False) as row2:
                  edit_concept_2 = gr.Textbox(
                                  label="Edit Concept",
                                  show_label=False,
                                  max_lines=1,
                                  placeholder="Enter your 2st edit prompt",
                              )
                  neg_guidance_2 = gr.Checkbox(
                      label='Negative Guidance',visible=True)
                  # guidnace_scale_2 = gr.Slider(label='Concept Guidance Scale', minimum=1, maximum=30,
                  #                              value=DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,
                  #                              step=0.5, interactive=True)
                  add_2 = gr.Button('Add')

              # 3rd SEGA concept
              with gr.Row(visible=False) as row3:
                 edit_concept_3 = gr.Textbox(
                                  label="Edit Concept",
                                  show_label=False,
                                  max_lines=1,
                                  placeholder="Enter your 3rd edit prompt",
                              )
                 neg_guidance_3 = gr.Checkbox(
                      label='Negative Guidance',visible=True)
                #  guidnace_scale_3 = gr.Slider(label='Concept Guidance Scale', minimum=1, maximum=30,
                #                                value=DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,
                #                                step=0.5, interactive=True)
                 add_3 = gr.Button('Add')




              with gr.Row().style(mobile_collapse=False, equal_height=True):
                add_concept_button = gr.Button("+")


    with gr.Row():
        run_button = gr.Button("Edit", visible=True)
        reconstruct_button = gr.Button("Show Reconstruction", visible=False)
        clear_button = gr.Button("Clear", visible=True)

    with gr.Accordion("Advanced Options", open=False):
      with gr.Tabs() as tabs:

          with gr.TabItem('General options', id=2):
            with gr.Row():
                with gr.Column():
                    src_prompt = gr.Textbox(lines=1, label="Source Prompt", interactive=True, placeholder="")
                    steps = gr.Number(value=100, precision=0, label="Num Diffusion Steps", interactive=True)
                    src_cfg_scale = gr.Number(value=3.5, label=f"Source Guidance Scale", interactive=True)
                    seed = gr.Number(value=0, precision=0, label="Seed", interactive=True)
                    randomize_seed = gr.Checkbox(label='Randomize seed', value=False)

                with gr.Column():
                    skip = gr.Slider(minimum=0, maximum=60, value=36, label="Skip Steps", interactive=True)
                    tar_cfg_scale = gr.Slider(minimum=7, maximum=30,value=15, label=f"Guidance Scale", interactive=True)

          with gr.TabItem('SEGA options', id=3) as sega_advanced_tab:
             # 1st SEGA concept
              with gr.Row().style(mobile_collapse=False, equal_height=True):
                  warmup_1 = gr.Slider(label='Warmup', minimum=0, maximum=50,
                                       value=DEFAULT_WARMUP_STEPS,
                                       step=1, interactive=True)
                  threshold_1 = gr.Slider(label='Threshold', minimum=0.5, maximum=0.99,
                                          value=DEFAULT_THRESHOLD, steps=0.01, interactive=True)

              # 2nd SEGA concept
              with gr.Row(visible=False) as row2_advanced:
                  warmup_2 = gr.Slider(label='Warmup', minimum=0, maximum=50,
                                       value=DEFAULT_WARMUP_STEPS,
                                       step=1, interactive=True)
                  threshold_2 = gr.Slider(label='Threshold', minimum=0.5, maximum=0.99,
                                          value=DEFAULT_THRESHOLD,
                                          steps=0.01, interactive=True)
              # 3rd SEGA concept
              with gr.Row(visible=False) as row3_advanced:
                  warmup_3 = gr.Slider(label='Warmup', minimum=0, maximum=50,
                                       value=DEFAULT_WARMUP_STEPS, step=1,
                                       interactive=True)
                  threshold_3 = gr.Slider(label='Threshold', minimum=0.5, maximum=0.99,
                                          value=DEFAULT_THRESHOLD, steps=0.01,
                                          interactive=True)

    caption_button.click(
        fn = caption_image,
        inputs = [input_image],
        outputs = [tar_prompt]
    )

    add_1.click(fn = update_display_concept_1, inputs=[add_1, edit_concept_1],  outputs=[box1, concept_1, concept_1, edit_concept_1, guidnace_scale_1, add_1])
    add_2.click(fn = update_display_concept_2, inputs=[add_2, edit_concept_2],  outputs=[box2, concept_2, concept_2, edit_concept_2, guidnace_scale_2, add_2])
    add_3.click(fn = update_display_concept_3, inputs=[add_3, edit_concept_3],  outputs=[box3, concept_3, concept_3, edit_concept_3, guidnace_scale_3, add_3])


    add_concept_button.click(fn = add_concept, inputs=sega_concepts_counter,
               outputs= [row2, row2_advanced, row3, row3_advanced, add_concept_button, sega_concepts_counter], queue = False)

    run_button.click(fn = update_inversion_progress_visibility, inputs =[input_image,do_inversion], outputs=[inversion_progress],queue=False).then(
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
        outputs=[wts, zs, do_inversion, inversion_progress],
    ).then(fn = update_inversion_progress_visibility, inputs =[input_image,do_inversion], outputs=[inversion_progress],queue=False).success(
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
        outputs=[sega_edited_image, reconstruct_button])
    # .success(fn=update_gallery_display, inputs= [prev_output_image, sega_edited_image], outputs = [gallery, gallery, prev_output_image])



    # Automatically start inverting upon input_image change
    input_image.change(
        fn = reset_do_inversion,
        outputs = [do_inversion],
        queue = False).then(fn = update_inversion_progress_visibility, inputs =[input_image,do_inversion],
                            outputs=[inversion_progress],queue=False).then(
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
        outputs=[wts, zs, do_inversion, inversion_progress],
    ).then(fn = update_inversion_progress_visibility, inputs =[input_image,do_inversion],
           outputs=[inversion_progress],queue=False).then(
              lambda: reconstruct_button.update(visible=False),
              outputs=[reconstruct_button]).then(
        fn = reset_do_reconstruction,
        outputs = [do_reconstruction],
        queue = False)


    # Repeat inversion (and reconstruction) when these params are changed:
    src_prompt.change(
        fn = reset_do_inversion,
        outputs = [do_inversion], queue = False).then(
        fn = reset_do_reconstruction,
        outputs = [do_reconstruction], queue = False)

    steps.change(
        fn = reset_do_inversion,
        outputs = [do_inversion], queue = False).then(
        fn = reset_do_reconstruction,
        outputs = [do_reconstruction], queue = False)


    src_cfg_scale.change(
        fn = reset_do_inversion,
        outputs = [do_inversion], queue = False).then(
        fn = reset_do_reconstruction,
        outputs = [do_reconstruction], queue = False)

    # Repeat only reconstruction these params are changed:

    tar_prompt.change(
        fn = reset_do_reconstruction,
        outputs = [do_reconstruction], queue = False)

    tar_cfg_scale.change(
        fn = reset_do_reconstruction,
        outputs = [do_reconstruction], queue = False)

    skip.change(
        fn = reset_do_reconstruction,
        outputs = [do_reconstruction], queue = False)



    clear_components = [input_image,ddpm_edited_image,ddpm_edited_image,sega_edited_image, do_inversion,
                                   src_prompt, steps, src_cfg_scale, seed,
                                  tar_prompt, skip, tar_cfg_scale, reconstruct_button,reconstruct_button,
                                  edit_concept_1, guidnace_scale_1,guidnace_scale_1,warmup_1,  threshold_1, neg_guidance_1, concept_1, concept_1, 
                                  edit_concept_2, guidnace_scale_2,guidnace_scale_2,warmup_2,  threshold_2, neg_guidance_2, concept_2, concept_2, row2, row2_advanced, 
                                  edit_concept_3, guidnace_scale_3,guidnace_scale_3,warmup_3,  threshold_3, neg_guidance_3, concept_3,concept_3, row3, row3_advanced ]

    clear_components_output_vals = [None, None,ddpm_edited_image.update(visible=False), None, True,
                     "", DEFAULT_DIFFUSION_STEPS, DEFAULT_SOURCE_GUIDANCE_SCALE, DEFAULT_SEED,
                     "", DEFAULT_SKIP_STEPS, DEFAULT_TARGET_GUIDANCE_SCALE, reconstruct_button.update(value="Show Reconstruction"),reconstruct_button.update(visible=False),
                     "", DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,guidnace_scale_1.update(visible=False), DEFAULT_WARMUP_STEPS, DEFAULT_THRESHOLD, DEFAULT_NEGATIVE_GUIDANCE, "", concept_1.update(visible=False),
                     "", DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,guidnace_scale_2.update(visible=False), DEFAULT_WARMUP_STEPS, DEFAULT_THRESHOLD, DEFAULT_NEGATIVE_GUIDANCE, "", concept_2.update(visible=False), row2.update(visible=False), row2_advanced.update(visible=False), 
                     "", DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,guidnace_scale_3.update(visible=False), DEFAULT_WARMUP_STEPS, DEFAULT_THRESHOLD, DEFAULT_NEGATIVE_GUIDANCE, "",concept_3.update(visible=False), row3.update(visible=False), row3_advanced.update(visible=False)
                         ]


    clear_button.click(lambda: clear_components_output_vals, outputs =clear_components)

    reconstruct_button.click(lambda: ddpm_edited_image.update(visible=True), outputs=[ddpm_edited_image]).then(fn = reconstruct,
                inputs = [tar_prompt,
                tar_cfg_scale,
                skip,
                wts, zs,
                do_reconstruction,
                reconstruction,
                          reconstruct_button],
                outputs = [ddpm_edited_image,reconstruction, ddpm_edited_image, do_reconstruction, reconstruct_button])

    randomize_seed.change(
        fn = randomize_seed_fn,
        inputs = [seed, randomize_seed],
        outputs = [seed],
        queue = False)
    
    # gr.Examples(
    #     label='Examples',
    #     examples=get_example(),
    #     inputs=[input_image, src_prompt, tar_prompt, steps,
    #                 # src_cfg_scale,
    #                 skip,
    #                 tar_cfg_scale,
    #                 edit_concept_1,
    #                 edit_concept_2,
    #                 guidnace_scale_1,
    #                 warmup_1,
    #                 # neg_guidance,
    #                 sega_edited_image
    #            ],
    #     outputs=[sega_edited_image],
    #     # fn=edit,
    #     # cache_examples=True
    # )





demo.queue()
demo.launch(share=False)



