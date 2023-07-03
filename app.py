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
sd_model_id = "stabilityai/stable-diffusion-2-1-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sd_pipe = StableDiffusionPipeline.from_pretrained(sd_model_id,torch_dtype=torch.float16).to(device)
sd_pipe.scheduler = DDIMScheduler.from_config(sd_model_id, subfolder = "scheduler")
sem_pipe = SemanticStableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16).to(device)
blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",torch_dtype=torch.float16).to(device)



## IMAGE CPATIONING ##
def caption_image(input_image):
    inputs = blip_processor(images=input_image, return_tensors="pt").to(device, torch.float16)
    pixel_values = inputs.pixel_values

    generated_ids = blip_model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption, generated_caption



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
  with inference_mode():
    w0 = (sd_pipe.vae.encode(x0).latent_dist.mode() * 0.18215)

  # find Zs and wts - forward process
  wt, zs, wts = inversion_forward_process(sd_pipe, w0, etas=eta, prompt=prompt_src, cfg_scale=cfg_scale_src, prog_bar=True, num_inference_steps=num_diffusion_steps)
  return zs, wts


def sample(zs, wts, prompt_tar="", cfg_scale_tar=15, skip=36, eta = 1):

    # reverse process (via Zs and wT)
    w0, _ = inversion_reverse_process(sd_pipe, xT=wts[skip], etas=eta, prompts=[prompt_tar], cfg_scales=[cfg_scale_tar], prog_bar=True, zs=zs[skip:])

    # vae decode image
    with inference_mode():
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


    x0 = load_512(input_image, device=device).to(torch.float16)

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
            image_caption,
            steps,
            skip,
            tar_cfg_scale,
            edit_concept_1,edit_concept_2,edit_concept_3,
            guidnace_scale_1,guidnace_scale_2,guidnace_scale_3,
            warmup_1, warmup_2, warmup_3,
            neg_guidance_1, neg_guidance_2, neg_guidance_3,
            threshold_1, threshold_2, threshold_3,
         do_reconstruction,
         reconstruction,
         
         # for inversion in case it needs to be re computed (and avoid delay):
         do_inversion,
         seed, 
         randomize_seed,
         src_prompt,
         src_cfg_scale):

    if do_inversion or randomize_seed:
        x0 = load_512(input_image, device=device).to(torch.float16)
        # invert and retrieve noise maps and latent
        zs_tensor, wts_tensor = invert(x0 =x0 , prompt_src=src_prompt, num_diffusion_steps=steps, cfg_scale_src=src_cfg_scale)
        wts = gr.State(value=wts_tensor)
        zs = gr.State(value=zs_tensor)
        do_inversion = False    
    
    if image_caption.lower() == tar_prompt.lower(): # if image caption was not changed, run pure sega
          tar_prompt = ""
        
    if edit_concept_1 != "" or edit_concept_2 != "" or edit_concept_3 != "":
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
      
      return sega_out.images[0], reconstruct_button.update(visible=True), do_reconstruction, reconstruction, wts, zs, do_inversion
    
    else: # if sega concepts were not added, performs regular ddpm sampling
      
      if do_reconstruction: # if ddpm sampling wasn't computed
          pure_ddpm_img = sample(zs.value, wts.value, prompt_tar=tar_prompt, skip=skip, cfg_scale_tar=tar_cfg_scale)
          reconstruction = gr.State(value=pure_ddpm_img)
          do_reconstruction = False
          return pure_ddpm_img, reconstruct_button.update(visible=False), do_reconstruction, reconstruction, wts, zs, do_inversion
      
      return reconstruction.value, reconstruct_button.update(visible=False), do_reconstruction, reconstruction, wts, zs, do_inversion
        

def randomize_seed_fn(seed, randomize_seed):
    if randomize_seed:
        seed = random.randint(0, np.iinfo(np.int32).max)
    torch.manual_seed(seed)
    return seed

    
    

def get_example():
    case = [
        [
            'examples/lemons_input.jpg', 
            # '',
            'apples', 'lemons',
            'a ceramic bowl',
             'examples/lemons_output.jpg',
            
            
            7,7,
            1,1,
            False, True,
            100,
            36,
            15,
           
             ],
        [
            'examples/girl_with_pearl_earring_input.png', 
            # '',
            'glasses', '',
            '',
             'examples/girl_with_pearl_earring_output.png',
            
            
            3,'',
            3,'',
            False,'' ,
            100,
            36,
            15,
           
             ],
        [
            'examples/rockey_shore_input.jpg', 
            # '',
            'sea turtle', '',
            'watercolor painting',
            'examples/rockey_shore_output.jpg',
            
            
            7,7,
            1,2,
            False,False,
            100,
            36,
            15,
             ],
                 [
            'examples/flower_field_input.jpg', 
            # '',
            'wheat', 'red flowers',
            'oil painting',
             'examples/flower_field_output_2.jpg',


            20,7,
            1,1,
                     False,True,
                      100,
            36,
            15,
            
             ],
        [
            'examples/butterfly_input.jpg', 
            # '',
             'bee', 'butterfly',
            'oil painting',
            'examples/butterfly_output.jpg',      
            7, 7,
            1,1,
            False, True,
                        100,
            36,
            15,
             ]
 ]
    return case


def swap_visibilities(input_image,  
                    edit_concept_1,
                    edit_concept_2,
                    tar_prompt,  
                    sega_edited_image,
                    guidnace_scale_1,
                    guidnace_scale_2,
                    warmup_1,
                    warmup_2,
                    neg_guidance_1,
                    neg_guidance_2,
                    steps,
                    skip,
                    tar_cfg_scale,
                    sega_concepts_counter
                    
):
    sega_concepts_counter=0
    concept1_update = update_display_concept("Remove" if neg_guidance_1 else "Add", edit_concept_1, neg_guidance_1, sega_concepts_counter)
    if(edit_concept_2 != ""):
        concept2_update = update_display_concept("Remove" if neg_guidance_2 else "Add", edit_concept_2, neg_guidance_2, sega_concepts_counter+1)
    else:
        concept2_update = gr.update(visible=False), gr.update(visible=False),gr.update(visible=False), gr.update(value=neg_guidance_2),gr.update(visible=True),gr.update(visible=False),sega_concepts_counter+1
    return (*concept1_update[:-1], *concept2_update)
    


########
# demo #
########


intro = """
<h1 style="font-weight: 1400; text-align: center; margin-bottom: 7px;">
   LEDITS - Pipeline for editing images
</h1>
<h3 style="font-weight: 600; text-align: center;">
    Real Image Latent Editing with Edit Friendly DDPM and Semantic Guidance
</h3>
<h4 style="text-align: center; margin-bottom: 7px;">
    <a href="https://editing-images-project.hf.space/" style="text-decoration: underline;" target="_blank">Project Page</a> | <a href="#" style="text-decoration: underline;" target="_blank">ArXiv</a>
</h4>

<p style="font-size: 0.9rem; margin: 0rem; line-height: 1.2em; margin-top:1em">
<a href="https://huggingface.co/spaces/editing-images/edit_friendly_ddpm_x_sega?duplicate=true">
<img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3CWLGkA" alt="Duplicate Space"></a>
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

with gr.Blocks(css="style.css") as demo:
    def update_counter(sega_concepts_counter, concept1, concept2, concept3):
        if sega_concepts_counter == "":
            sega_concepts_counter = sum(1 for concept in (concept1, concept2, concept3) if concept != '')
        return sega_concepts_counter
    def remove_concept(sega_concepts_counter, row_triggered):
      sega_concepts_counter -= 1
      rows_visibility = [gr.update(visible=False) for _ in range(4)]
      
      if(row_triggered-1 > sega_concepts_counter):
            rows_visibility[sega_concepts_counter] = gr.update(visible=True)
      else:
            rows_visibility[row_triggered-1] = gr.update(visible=True)
      
      row1_visibility, row2_visibility, row3_visibility, row4_visibility = rows_visibility

      guidance_scale_label = "Concept Guidance Scale"
      enable_interactive =  gr.update(interactive=True)
      return (gr.update(visible=False),
              gr.update(visible=False, value="",),
              gr.update(interactive=True, value=""),
              gr.update(visible=False,label = guidance_scale_label),
              gr.update(interactive=True, value =False),
              gr.update(value=DEFAULT_WARMUP_STEPS),
              gr.update(value=DEFAULT_THRESHOLD),
              gr.update(visible=True),
              enable_interactive,
              row1_visibility,
              row2_visibility,
              row3_visibility,
              row4_visibility,
              sega_concepts_counter
             ) 
    
    
    
    def update_display_concept(button_label, edit_concept, neg_guidance, sega_concepts_counter):
      sega_concepts_counter += 1
      guidance_scale_label = "Concept Guidance Scale"
      if(button_label=='Remove'):
        neg_guidance = True
        guidance_scale_label = "Negative Guidance Scale" 
      
      return (gr.update(visible=True), #boxn
             gr.update(visible=True, value=edit_concept), #concept_n
             gr.update(visible=True,label = guidance_scale_label), #guidance_scale_n
             gr.update(value=neg_guidance),#neg_guidance_n
             gr.update(visible=False), #row_n
             gr.update(visible=True), #row_n+1
             sega_concepts_counter
             ) 


    def display_editing_options(run_button, clear_button, sega_tab):
      return run_button.update(visible=True), clear_button.update(visible=True), sega_tab.update(visible=True)
    
    def update_interactive_mode(add_button_label):
      if add_button_label == "Clear":
        return gr.update(interactive=False), gr.update(interactive=False)
      else:
        return gr.update(interactive=True), gr.update(interactive=True)
    
    def update_dropdown_parms(dropdown):
        if dropdown == 'custom':
          return DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,DEFAULT_WARMUP_STEPS, DEFAULT_THRESHOLD
        elif dropdown =='style':
          return STYLE_SEGA_CONCEPT_GUIDANCE_SCALE,STYLE_WARMUP_STEPS, STYLE_THRESHOLD
        elif dropdown =='object':
          return OBJECT_SEGA_CONCEPT_GUIDANCE_SCALE,OBJECT_WARMUP_STEPS, OBJECT_THRESHOLD
        elif dropdown =='faces':
          return FACE_SEGA_CONCEPT_GUIDANCE_SCALE,FACE_WARMUP_STEPS, FACE_THRESHOLD


    def reset_do_inversion():
        return True

    def reset_do_reconstruction():
      do_reconstruction = True
      return  do_reconstruction

    def reset_image_caption():
        return ""

    def update_inversion_progress_visibility(input_image, do_inversion):
      if do_inversion and not input_image is None:
          return inversion_progress.update(visible=True)
      else:
        return inversion_progress.update(visible=False)

    def update_edit_progress_visibility(input_image, do_inversion):
      # if do_inversion and not input_image is None:
      #     return inversion_progress.update(visible=True)
      # else:
        return inversion_progress.update(visible=True)


    gr.HTML(intro)
    wts = gr.State()
    zs = gr.State()
    reconstruction = gr.State()
    do_inversion = gr.State(value=True)
    do_reconstruction = gr.State(value=True)
    sega_concepts_counter = gr.State(0)
    image_caption = gr.State(value="")



    with gr.Row():
        input_image = gr.Image(label="Input Image", interactive=True)
        ddpm_edited_image = gr.Image(label=f"Pure DDPM Inversion Image", interactive=False, visible=False)
        sega_edited_image = gr.Image(label=f"LEDITS Edited Image", interactive=False)
        input_image.style(height=365, width=365)
        ddpm_edited_image.style(height=365, width=365)
        sega_edited_image.style(height=365, width=365)

    with gr.Row():
      with gr.Box(visible=False) as box1:
        with gr.Row():
          concept_1 = gr.Button(scale=3)
          remove_concept1 = gr.Button("x", scale=1, min_width=10)
        with gr.Row():
            guidnace_scale_1 = gr.Slider(label='Concept Guidance Scale', minimum=1, maximum=30,
                            info="How strongly the concept should modify the image",
                                                  value=DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,
                                                  step=0.5, interactive=True)
      with gr.Box(visible=False) as box2:
        with gr.Row():
          concept_2 = gr.Button(scale=3)
          remove_concept2 = gr.Button("x", scale=1, min_width=10)
        with gr.Row():
          guidnace_scale_2 = gr.Slider(label='Concept Guidance Scale', minimum=1, maximum=30,
                              info="How strongly the concept should modify the image",
                                                    value=DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,
                                                    step=0.5, interactive=True)
      with gr.Box(visible=False) as box3:
        with gr.Row():
          concept_3 = gr.Button(visible=False, scale=3)
          remove_concept3 = gr.Button("x", scale=1, min_width=10)
        with gr.Row():
          guidnace_scale_3 = gr.Slider(label='Concept Guidance Scale', minimum=1, maximum=30,
                              info="How strongly the concept should modify the image",
                                                    value=DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,
                                                    step=0.5, interactive=True,visible=False)


    with gr.Row():
        inversion_progress = gr.Textbox(visible=False, label="Inversion progress")
        

    with gr.Box():
        intro_segs = gr.Markdown("Add/Remove Concepts from your Image <span style=\"font-size: 12px; color: rgb(156, 163, 175)\">with Semantic Guidance</span>")
                  # 1st SEGA concept
        with gr.Row().style(mobile_collapse=False) as row1:
              with gr.Column(scale=3, min_width=100):
                  with gr.Row().style(mobile_collapse=True):
                      with gr.Column(scale=3, min_width=100):
                              edit_concept_1 = gr.Textbox(
                                              label="Concept",
                                              show_label=True,
                                              max_lines=1, value="",
                                              placeholder="E.g.: Sunglasses",
                                          )
                      with gr.Column(scale=2, min_width=100):
                              dropdown1 = gr.Dropdown(label = "Edit Type", value ='custom' , choices=['custom','style', 'object', 'faces'])
    

              with gr.Column(scale=1, min_width=100, visible=False):
                      neg_guidance_1 = gr.Checkbox(
                          label='Remove Concept?')
              
              with gr.Column(scale=1, min_width=100):
                      add_1 = gr.Button('Add')
                      remove_1 = gr.Button('Remove')
             
    
                  # 2nd SEGA concept
        with gr.Row(visible=False).style(equal_height=True) as row2:
            with gr.Column(scale=3, min_width=100):
                with gr.Row().style(mobile_collapse=True): #better mobile UI
                    with gr.Column(scale=3, min_width=100):
                              edit_concept_2 = gr.Textbox(
                                              label="Concept",
                                              show_label=True,
                                              max_lines=1,
                                              placeholder="E.g.: Realistic",
                                          )
                    with gr.Column(scale=2, min_width=100):
                              dropdown2 = gr.Dropdown(label = "Edit Type", value ='custom' , choices=['custom','style', 'object', 'faces'])

            with gr.Column(scale=1, min_width=100, visible=False):
                      neg_guidance_2 = gr.Checkbox(
                          label='Remove Concept?')
                
            with gr.Column(scale=1, min_width=100):
                      add_2 = gr.Button('Add')
                      remove_2 = gr.Button('Remove')
    
                  # 3rd SEGA concept
        with gr.Row(visible=False).style(equal_height=True) as row3:
            with gr.Column(scale=3, min_width=100):
                with gr.Row().style(mobile_collapse=True): #better mobile UI  
                    with gr.Column(scale=3, min_width=100):
                             edit_concept_3 = gr.Textbox(
                                              label="Concept",
                                              show_label=True,
                                              max_lines=1,
                                              placeholder="E.g.: orange",
                                          )
                    with gr.Column(scale=2, min_width=100):
                             dropdown3 = gr.Dropdown(label = "Edit Type", value ='custom' , choices=['custom','style', 'object', 'faces'])
            
            with gr.Column(scale=1, min_width=100, visible=False):
                             neg_guidance_3 = gr.Checkbox(
                              label='Remove Concept?',visible=True)
            with gr.Column(scale=1, min_width=100):
                     add_3 = gr.Button('Add')
                     remove_3 = gr.Button('Remove')
    
        with gr.Row(visible=False).style(equal_height=True) as row4:
            gr.Markdown("### Max of 3 concepts reached. Remove a concept to add more")
    
        #with gr.Row(visible=False).style(mobile_collapse=False, equal_height=True):
        #            add_concept_button = gr.Button("+1 concept")


    
    with gr.Row().style(mobile_collapse=False, equal_height=True):
                tar_prompt = gr.Textbox(
                                label="Describe your edited image (optional)",
                                # show_label=False,
                                max_lines=1, value="", scale=3,
                                placeholder="Target prompt, DDPM Inversion", info = "DDPM Inversion Prompt. Can help with global changes, modify to what you would like to see"
                            )
                # caption_button = gr.Button("Caption Image", scale=1)
        
    
    with gr.Row():
        run_button = gr.Button("Edit your image!", visible=True)
        

    with gr.Accordion("Advanced Options", open=False):
      with gr.Tabs() as tabs:

          with gr.TabItem('General options', id=2):
            with gr.Row():
                with gr.Column(min_width=100):
                   clear_button = gr.Button("Clear", visible=True)
                   src_prompt = gr.Textbox(lines=1, label="Source Prompt", interactive=True, placeholder="")
                   steps = gr.Number(value=100, precision=0, label="Num Diffusion Steps", interactive=True)
                   src_cfg_scale = gr.Number(value=3.5, label=f"Source Guidance Scale", interactive=True)
                   

                with gr.Column(min_width=100):
                    reconstruct_button = gr.Button("Show Reconstruction", visible=False)
                    skip = gr.Slider(minimum=0, maximum=60, value=36, label="Skip Steps", interactive=True, info = "At which step to start denoising. Bigger values increase fidelity to input image")
                    tar_cfg_scale = gr.Slider(minimum=7, maximum=30,value=15, label=f"Guidance Scale", interactive=True)
                    seed = gr.Number(value=0, precision=0, label="Seed", interactive=True)
                    randomize_seed = gr.Checkbox(label='Randomize seed', value=False)

          with gr.TabItem('SEGA options', id=3) as sega_advanced_tab:
             # 1st SEGA concept
              gr.Markdown("1st concept")
              with gr.Row().style(mobile_collapse=False, equal_height=True):
                  warmup_1 = gr.Slider(label='Warmup', minimum=0, maximum=50,
                                       value=DEFAULT_WARMUP_STEPS,
                                       step=1, interactive=True, info="At which step to start applying semantic guidance. Bigger values reduce edit concept's effect")
                  threshold_1 = gr.Slider(label='Threshold', minimum=0.5, maximum=0.99,
                                          value=DEFAULT_THRESHOLD, step=0.01, interactive=True, 
                                          info = "Lower the threshold for more effect (e.g. ~0.9 for style transfer)")

              # 2nd SEGA concept
              gr.Markdown("2nd concept")
              with gr.Row() as row2_advanced:
                  warmup_2 = gr.Slider(label='Warmup', minimum=0, maximum=50,
                                       value=DEFAULT_WARMUP_STEPS,
                                       step=1, interactive=True, info="At which step to start applying semantic guidance. Bigger values reduce edit concept's effect")
                  threshold_2 = gr.Slider(label='Threshold', minimum=0.5, maximum=0.99,
                                          value=DEFAULT_THRESHOLD,
                                          step=0.01, interactive=True,
                                         info = "Lower the threshold for more effect (e.g. ~0.9 for style transfer)")
              # 3rd SEGA concept
              gr.Markdown("3rd concept")
              with gr.Row() as row3_advanced:
                  warmup_3 = gr.Slider(label='Warmup', minimum=0, maximum=50,
                                       value=DEFAULT_WARMUP_STEPS, step=1,
                                       interactive=True, info="At which step to start applying semantic guidance. Bigger values reduce edit concept's effect")
                  threshold_3 = gr.Slider(label='Threshold', minimum=0.5, maximum=0.99,
                                          value=DEFAULT_THRESHOLD, step=0.01,
                                          interactive=True,
                                         info = "Lower the threshold for more effect (e.g. ~0.9 for style transfer)")

    # caption_button.click(
    #     fn = caption_image,
    #     inputs = [input_image],
    #     outputs = [tar_prompt]
    # )
    #neg_guidance_1.change(fn = update_label, inputs=[neg_guidance_1], outputs=[add_1])
    #neg_guidance_2.change(fn = update_label, inputs=[neg_guidance_2], outputs=[add_2])
    #neg_guidance_3.change(fn = update_label, inputs=[neg_guidance_3], outputs=[add_3])
    add_1.click(fn=update_counter,inputs=[sega_concepts_counter,edit_concept_1,edit_concept_2,edit_concept_3], outputs=sega_concepts_counter).then(fn = update_display_concept, inputs=[add_1, edit_concept_1, neg_guidance_1, sega_concepts_counter],  outputs=[box1, concept_1, guidnace_scale_1,neg_guidance_1,row1, row2, sega_concepts_counter],queue=False)
    add_2.click(fn=update_counter,inputs=[sega_concepts_counter,edit_concept_1,edit_concept_2,edit_concept_3], outputs=sega_concepts_counter).then(fn = update_display_concept, inputs=[add_2, edit_concept_2, neg_guidance_2, sega_concepts_counter],  outputs=[box2, concept_2, guidnace_scale_2,neg_guidance_2,row2, row3, sega_concepts_counter],queue=False)
    add_3.click(fn=update_counter,inputs=[sega_concepts_counter,edit_concept_1,edit_concept_2,edit_concept_3], outputs=sega_concepts_counter).then(fn = update_display_concept, inputs=[add_3, edit_concept_3, neg_guidance_3, sega_concepts_counter],  outputs=[box3, concept_3, guidnace_scale_3,neg_guidance_3,row3, row4, sega_concepts_counter],queue=False)
    
    remove_1.click(fn = update_display_concept, inputs=[remove_1, edit_concept_1, neg_guidance_1, sega_concepts_counter],  outputs=[box1, concept_1, guidnace_scale_1,neg_guidance_1,row1, row2, sega_concepts_counter],queue=False)
    remove_2.click(fn = update_display_concept, inputs=[remove_2, edit_concept_2, neg_guidance_2 ,sega_concepts_counter],  outputs=[box2, concept_2, guidnace_scale_2,neg_guidance_2,row2, row3,sega_concepts_counter],queue=False)
    remove_3.click(fn = update_display_concept, inputs=[remove_3, edit_concept_3, neg_guidance_3, sega_concepts_counter],  outputs=[box3, concept_3, guidnace_scale_3,neg_guidance_3, row3, row4, sega_concepts_counter],queue=False)
    
    remove_concept1.click(
        fn=update_counter,inputs=[sega_concepts_counter,edit_concept_1,edit_concept_2,edit_concept_3], outputs=sega_concepts_counter).then(
        fn = remove_concept, inputs=[sega_concepts_counter,gr.State(1)], outputs= [box1, concept_1, edit_concept_1, guidnace_scale_1,neg_guidance_1,warmup_1, threshold_1, add_1, dropdown1, row1, row2, row3, row4, sega_concepts_counter],queue=False)
    remove_concept2.click(
        fn=update_counter,inputs=[sega_concepts_counter,edit_concept_1,edit_concept_2,edit_concept_3], outputs=sega_concepts_counter).then(
        fn = remove_concept,  inputs=[sega_concepts_counter,gr.State(2)], outputs=[box2, concept_2, edit_concept_2, guidnace_scale_2,neg_guidance_2, warmup_2, threshold_2, add_2 , dropdown2, row1, row2, row3, row4, sega_concepts_counter],queue=False)
    remove_concept3.click(
        fn=update_counter,inputs=[sega_concepts_counter,edit_concept_1,edit_concept_2,edit_concept_3], outputs=sega_concepts_counter).then(
        fn = remove_concept,inputs=[sega_concepts_counter,gr.State(3)], outputs=[box3, concept_3, edit_concept_3, guidnace_scale_3,neg_guidance_3,warmup_3, threshold_3,  add_3, dropdown3, row1, row2, row3, row4, sega_concepts_counter],queue=False)

    #add_concept_button.click(fn = update_display_concept, inputs=sega_concepts_counter,
    #           outputs= [row2, row2_advanced, row3, row3_advanced, add_concept_button, sega_concepts_counter], queue = False)

    run_button.click(
        fn=edit,
        inputs=[input_image,
                wts, zs,
                tar_prompt,
                image_caption,
                steps,
                skip,
                tar_cfg_scale,
                edit_concept_1,edit_concept_2,edit_concept_3,
                guidnace_scale_1,guidnace_scale_2,guidnace_scale_3,
                warmup_1, warmup_2, warmup_3,
                neg_guidance_1, neg_guidance_2, neg_guidance_3,
                threshold_1, threshold_2, threshold_3, do_reconstruction, reconstruction,
                do_inversion,
                seed, 
                randomize_seed,
                src_prompt,
                src_cfg_scale


        ],
        outputs=[sega_edited_image, reconstruct_button, do_reconstruction, reconstruction, wts, zs, do_inversion])
    # .success(fn=update_gallery_display, inputs= [prev_output_image, sega_edited_image], outputs = [gallery, gallery, prev_output_image])


    input_image.change(
        fn = reset_do_inversion,
        outputs = [do_inversion],
        queue = False)
    # Automatically start inverting upon input_image change
    input_image.upload(
        fn = reset_do_inversion,
        outputs = [do_inversion],
        queue = False).then(fn = caption_image,
        inputs = [input_image],
        outputs = [tar_prompt, image_caption]).then(fn = update_inversion_progress_visibility, inputs =[input_image,do_inversion],
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

    dropdown1.change(fn=update_dropdown_parms, inputs = [dropdown1], outputs = [guidnace_scale_1,warmup_1,  threshold_1])
    dropdown2.change(fn=update_dropdown_parms, inputs = [dropdown2], outputs = [guidnace_scale_2,warmup_2,  threshold_2])
    dropdown3.change(fn=update_dropdown_parms, inputs = [dropdown3], outputs = [guidnace_scale_3,warmup_3,  threshold_3])

    clear_components = [input_image,ddpm_edited_image,ddpm_edited_image,sega_edited_image, do_inversion,
                                   src_prompt, steps, src_cfg_scale, seed,
                                  tar_prompt, skip, tar_cfg_scale, reconstruct_button,reconstruct_button,
                                  edit_concept_1, guidnace_scale_1,guidnace_scale_1,warmup_1,  threshold_1, neg_guidance_1,dropdown1, concept_1, concept_1,
                                  edit_concept_2, guidnace_scale_2,guidnace_scale_2,warmup_2,  threshold_2, neg_guidance_2,dropdown2, concept_2, concept_2, row2, row2_advanced,
                                  edit_concept_3, guidnace_scale_3,guidnace_scale_3,warmup_3,  threshold_3, neg_guidance_3,dropdown3, concept_3,concept_3, row3, row3_advanced ]

    clear_components_output_vals = [None, None,ddpm_edited_image.update(visible=False), None, True,
                     "", DEFAULT_DIFFUSION_STEPS, DEFAULT_SOURCE_GUIDANCE_SCALE, DEFAULT_SEED,
                     "", DEFAULT_SKIP_STEPS, DEFAULT_TARGET_GUIDANCE_SCALE, reconstruct_button.update(value="Show Reconstruction"),reconstruct_button.update(visible=False),
                     "", DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,guidnace_scale_1.update(visible=False), DEFAULT_WARMUP_STEPS, DEFAULT_THRESHOLD, DEFAULT_NEGATIVE_GUIDANCE, "","default", concept_1.update(visible=False),
                     "", DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,guidnace_scale_2.update(visible=False), DEFAULT_WARMUP_STEPS, DEFAULT_THRESHOLD, DEFAULT_NEGATIVE_GUIDANCE, "","default", concept_2.update(visible=False), row2.update(visible=False), row2_advanced.update(visible=False),
                     "", DEFAULT_SEGA_CONCEPT_GUIDANCE_SCALE,guidnace_scale_3.update(visible=False), DEFAULT_WARMUP_STEPS, DEFAULT_THRESHOLD, DEFAULT_NEGATIVE_GUIDANCE, "","default",concept_3.update(visible=False), row3.update(visible=False), row3_advanced.update(visible=False)
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


    
    gr.Examples(
        label='Examples',
        fn=swap_visibilities,
        run_on_click=True,
        examples=get_example(),
        inputs=[input_image,
                    edit_concept_1,
                    edit_concept_2,
                    tar_prompt,
                    sega_edited_image,
                    guidnace_scale_1,
                    guidnace_scale_2,
                    warmup_1,
                    warmup_2,
                    neg_guidance_1,
                    neg_guidance_2,
                    steps,
                    skip,
                    tar_cfg_scale,
                    sega_concepts_counter
               ],
        outputs=[box1, concept_1, guidnace_scale_1,neg_guidance_1, row1, row2,box2, concept_2, guidnace_scale_2,neg_guidance_2,row2, row3,sega_concepts_counter],
        cache_examples=True
    )


demo.queue()
demo.launch()