import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import base64
from io import BytesIO
import os
from volta_accelerate import DemoDiffusion
import tensorrt as trt
from utilities import Engine, DPMScheduler, LMSDiscreteScheduler, save_image, TRT_LOGGER
from PIL import Image

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
    MODEL_NAME = os.getenv("MODEL_NAME")
    engine_dir = f'engine/{MODEL_NAME}'
    onnx_dir = "onnx"
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    # Initialize demo
    model = DemoDiffusion(
        model_path=MODEL_NAME,
        denoising_steps=10,
        denoising_fp16=True,
        output_dir="output",
        scheduler="LMSD",
        hf_token=HF_AUTH_TOKEN,
        verbose=False,
        nvtx_profile=False,
        max_batch_size=16,
    )

    model.loadEngines(engine_dir, onnx_dir, 16, 
        opt_batch_size=1, opt_image_height=512, opt_image_width=512, \
        force_export=False, force_optimize=False, \
        force_build=False, minimal_optimization=False, \
        static_batch=False, static_shape=True, \
        enable_preview=False)
    model.loadModules()
    

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    negative_prompt = model_inputs.get('negative_prompt', None)
    height = model_inputs.get('height', 512)
    width = model_inputs.get('width', 512)
    num_inference_steps = 10
    guidance_scale = model_inputs.get('guidance_scale', 7.5)
    input_seed = model_inputs.get("seed",None)
    
    if prompt == None:
        return {'message': "No prompt provided"}
    
    outputs = model.infer(prompt, negative_prompt, height, width, guidance_scale,num_inference_steps,verbose=False, seed=input_seed)
    image = Image.fromarray(outputs[0])
    buffered = BytesIO()
    image.save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return {'image_base64': image_base64}
