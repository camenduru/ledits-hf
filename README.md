---
title: LEDITS
emoji: ✏️
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 3.35.2
app_file: app.py
pinned: true
---

This is the repository for LEDITS - Real Image Latent Editing with Edit Friendly DDPM and Semantic Guidance.  More information about the technique [here](https://editing-images-project.hf.space)

This repository contains the following relevant files: 
- `app.py` - Gradio application for the inversion technique combining uploading an image, captioning it, doing the DDPM Inversion and applying SEGA concepts to the editing.
- `constants.py` - default config values for the `app.py`
- `inversion_utils.py` - utilities for providing the DDPM Inversion
- `modified_pipeline_semantic_stable_diffusion.py` - modified pipeline of SEGA for the purposes of LEDIS
- `utils.py` - generic useful utils for the app
