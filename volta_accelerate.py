import argparse 
from cuda import cudart
from models import CLIP, UNet, VAE
import numpy as np
import nvtx
import os
import onnx
from polygraphy import cuda
import time
import tqdm
import torch
from transformers import CLIPTokenizer
import uuid
import tensorrt as trt
from utilities import Engine, DPMScheduler, LMSDiscreteScheduler, save_image, TRT_LOGGER
import gc

class DemoDiffusion:
    """
    Application showcasing the acceleration of Stable Diffusion v1.4 pipeline using NVidia TensorRT w/ Plugins.
    """
    def __init__(
        self,

        denoising_steps,
        denoising_fp16=True,
        scheduler="LMSD",
        guidance_scale=7.5,
        device='cuda',
        output_dir='.',
        hf_token=None,
        verbose=False,
        nvtx_profile=False,
        max_batch_size=16,
        model_path="CompVis/stable-diffusion-v1-4"
    ):
        """
        Initializes the Diffusion pipeline.
        Args:
            denoising_steps (int):
                The number of denoising steps.
                More denoising steps usually lead to a higher quality image at the expense of slower inference.
            denoising_fp16 (bool):
                Run the denoising loop (UNet) in fp16 precision.
                When enabled image quality will be lower but generally results in higher throughput.
            guidance_scale (float):
                Guidance scale is enabled by setting as > 1.
                Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality.
            device (str):
                PyTorch device to run inference. Default: 'cuda'
            output_dir (str):
                Output directory for log files and image artifacts
            hf_token (str):
                HuggingFace User Access Token to use for downloading Stable Diffusion model checkpoints.
            verbose (bool):
                Enable verbose logging.
            nvtx_profile (bool):
                Insert NVTX profiling markers.
            max_batch_size (int):
                Max batch size for dynamic batch engines.
        """
        # Only supports single image per prompt.
        self.num_images = 1

        self.denoising_steps = denoising_steps
        self.denoising_fp16 = denoising_fp16
        assert guidance_scale > 1.0
        self.guidance_scale = guidance_scale
        self.model_path = model_path
        self.output_dir = output_dir
        self.hf_token = hf_token
        self.device = device
        self.verbose = verbose
        self.nvtx_profile = nvtx_profile

        # A scheduler to be used in combination with unet to denoise the encoded image latens.
        # This demo uses an adaptation of LMSDiscreteScheduler or DPMScheduler:
        sched_opts = {'num_train_timesteps': 1000, 'beta_start': 0.00085, 'beta_end': 0.012}
        if scheduler == "DPM":
            self.scheduler = DPMScheduler(device=self.device, **sched_opts)
        elif scheduler == "LMSD":
            self.scheduler = LMSDiscreteScheduler(device=self.device, **sched_opts)
        else:
            raise ValueError(f"Scheduler should be either DPM or LMSD")

        self.tokenizer = None

        self.unet_model_key = 'unet_fp16' if denoising_fp16 else 'unet'
        self.models = {
            'clip': CLIP(hf_token=hf_token, device=device, verbose=verbose, max_batch_size=max_batch_size),
            self.unet_model_key: UNet(model_path=model_path, hf_token=hf_token, fp16=denoising_fp16, device=device, verbose=verbose, max_batch_size=max_batch_size),
            'vae': VAE(hf_token=hf_token, device=device, verbose=verbose, max_batch_size=max_batch_size)
        }

        self.engine = {}
        self.stream = cuda.Stream()

    def teardown(self):
        for engine in self.engine.values():
            del engine
        self.stream.free()
        del self.stream

    def getModelPath(self, name, onnx_dir, opt=True):
        return os.path.join(onnx_dir, name+('.opt' if opt else '')+'.onnx')
    
    def buildOnlyEngines(
        self,
        engine_dir,
        onnx_dir,
        onnx_opset,
        opt_batch_size,
        opt_image_height,
        opt_image_width,
        force_export=False,
        force_optimize=False,
        force_build=False,
        minimal_optimization=False,
        static_batch=False,
        static_shape=True,
        enable_preview=False,
    ):
        print("[I] Compile only mode")
        for model_name, obj in self.models.items():
            engine = Engine(model_name, engine_dir)
            onnx_path = self.getModelPath(model_name, onnx_dir, opt=False)
            onnx_opt_path = self.getModelPath(model_name, onnx_dir)
            print(f"Exporting model: {onnx_path}")
            model = obj.get_model()
            with torch.inference_mode(), torch.autocast("cuda"):
                inputs = obj.get_sample_input(opt_batch_size, opt_image_height, opt_image_width)
                torch.onnx.export(model,
                        inputs,
                        onnx_path,
                        export_params=True,
                        opset_version=onnx_opset,
                        do_constant_folding=True,
                        input_names = obj.get_input_names(),
                        output_names = obj.get_output_names(),
                        dynamic_axes=obj.get_dynamic_axes(),
                )
            del model
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Generating optimizing model: {onnx_opt_path}")
            onnx_opt_graph = obj.optimize(onnx.load(onnx_path), minimal_optimization=minimal_optimization)
            onnx.save(onnx_opt_graph, onnx_opt_path)

            # Build engine
            engine.build(onnx_opt_path, fp16=True, \
                input_profile=obj.get_input_profile(opt_batch_size, opt_image_height, opt_image_width, \
                    static_batch=static_batch, static_shape=static_shape), \
                enable_preview=enable_preview)
            engine.__del__()
            del engine
            gc.collect()
            torch.cuda.empty_cache()


    def loadEngines(
        self,
        engine_dir,
        onnx_dir,
        onnx_opset,
        opt_batch_size,
        opt_image_height,
        opt_image_width,
        force_export=False,
        force_optimize=False,
        force_build=False,
        minimal_optimization=False,
        static_batch=False,
        static_shape=True,
        enable_preview=False,
    ):
        """
        Build and load engines for TensorRT accelerated inference.
        Export ONNX models first, if applicable.
        Args:
            engine_dir (str):
                Directory to write the TensorRT engines.
            onnx_dir (str):
                Directory to write the ONNX models.
            onnx_opset (int):
                ONNX opset version to export the models.
            opt_batch_size (int):
                Batch size to optimize for during engine building.
            opt_image_height (int):
                Image height to optimize for during engine building. Must be a multiple of 8.
            opt_image_width (int):
                Image width to optimize for during engine building. Must be a multiple of 8.
            force_export (bool):
                Force re-exporting the ONNX models.
            force_optimize (bool):
                Force re-optimizing the ONNX models.
            force_build (bool):
                Force re-building the TensorRT engine.
            minimal_optimization (bool):
                Apply minimal optimizations during build (no plugins).
            static_batch (bool):
                Build engine only for specified opt_batch_size.
            static_shape (bool):
                Build engine only for specified opt_image_height & opt_image_width. Default = True.
            enable_preview (bool):
                Enable TensorRT preview features.
        """
        # Build engines
        for model_name, obj in self.models.items():
            engine = Engine(model_name, engine_dir)
            if force_build or not os.path.exists(engine.engine_path):
                onnx_path = self.getModelPath(model_name, onnx_dir, opt=False)
                onnx_opt_path = self.getModelPath(model_name, onnx_dir)
                if not os.path.exists(onnx_opt_path):
                    # Export onnx
                    if force_export or not os.path.exists(onnx_path):
                        print(f"Exporting model: {onnx_path}")
                        model = obj.get_model()
                        with torch.inference_mode(), torch.autocast("cuda"):
                            inputs = obj.get_sample_input(opt_batch_size, opt_image_height, opt_image_width)
                            torch.onnx.export(model,
                                    inputs,
                                    onnx_path,
                                    export_params=True,
                                    opset_version=onnx_opset,
                                    do_constant_folding=True,
                                    input_names = obj.get_input_names(),
                                    output_names = obj.get_output_names(),
                                    dynamic_axes=obj.get_dynamic_axes(),
                            )
                        del model
                        gc.collect()
                    else:
                        print(f"Found cached model: {onnx_path}")
                    # Optimize onnx
                    if force_optimize or not os.path.exists(onnx_opt_path):
                        print(f"Generating optimizing model: {onnx_opt_path}")
                        onnx_opt_graph = obj.optimize(onnx.load(onnx_path), minimal_optimization=minimal_optimization)
                        onnx.save(onnx_opt_graph, onnx_opt_path)
                    else:
                        print(f"Found cached optimized model: {onnx_opt_path} ")
                # Build engine
                engine.build(onnx_opt_path, fp16=True, \
                    input_profile=obj.get_input_profile(opt_batch_size, opt_image_height, opt_image_width, \
                        static_batch=static_batch, static_shape=static_shape), \
                    enable_preview=enable_preview)
            self.engine[model_name] = engine

        # Separate iteration to activate engines
        for model_name, obj in self.models.items():
            self.engine[model_name].activate()
        gc.collect()
        

    def loadModules(
        self,
    ):
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.scheduler.set_timesteps(self.denoising_steps)
        # Pre-compute latent input scales and linear multistep coefficients
        self.scheduler.configure()

    def runEngine(self, model_name, feed_dict):
        engine = self.engine[model_name]
        return engine.infer(feed_dict, self.stream)

    def infer(
        self,
        prompt,
        negative_prompt,
        image_height,
        image_width,
        guidance_scale,
        inference_steps,
        warmup = False,
        verbose = False,
        seed=None
    ):
        """
        Run the diffusion pipeline.
        Args:
            prompt (str):
                The text prompt to guide image generation.
            negative_prompt (str):
                The prompt not to guide the image generation.
            image_height (int):
                Height (in pixels) of the image to be generated. Must be a multiple of 8.
            image_width (int):
                Width (in pixels) of the image to be generated. Must be a multiple of 8.
            warmup (bool):
                Indicate if this is a warmup run.
            verbose (bool):
                Enable verbose logging.
        """
        # Process inputs
        batch_size = 1
        #assert len(prompt) == len(negative_prompt)

        self.scheduler.set_timesteps(inference_steps)

        # Spatial dimensions of latent tensor
        latent_height = image_height // 8
        latent_width = image_width // 8

        # Create profiling events
        events = {}
        for stage in ['clip', 'denoise', 'vae']:
            for marker in ['start', 'stop']:
                events[stage+'-'+marker] = cudart.cudaEventCreate()[1]

        # Allocate buffers for TensorRT engine bindings
        for model_name, obj in self.models.items():
            self.engine[model_name].allocate_buffers(shape_dict=obj.get_shape_dict(batch_size, image_height, image_width), device=self.device)

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed) 

        # Run Stable Diffusion pipeline
        with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER) as runtime:
            # latents need to be generated on the target device
            unet_channels = 4 # unet.in_channels
            latents_shape = (batch_size * self.num_images, unet_channels, latent_height, latent_width)
            latents_dtype = torch.float32 # text_embeddings.dtype
            latents = torch.randn(latents_shape, device=self.device, dtype=latents_dtype, generator=generator)

            # Scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma

            torch.cuda.synchronize()
            e2e_tic = time.perf_counter()

            if self.nvtx_profile:
                nvtx_clip = nvtx.start_range(message='clip', color='green')
            cudart.cudaEventRecord(events['clip-start'], 0)
            # Tokenize input
            text_input_ids = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.type(torch.int32).to(self.device)

            # CLIP text encoder
            text_input_ids_inp = cuda.DeviceView(ptr=text_input_ids.data_ptr(), shape=text_input_ids.shape, dtype=np.int32)
            text_embeddings = self.runEngine('clip', {"input_ids": text_input_ids_inp})['text_embeddings']

            # Duplicate text embeddings for each generation per prompt
            bs_embed, seq_len, _ = text_embeddings.shape
            text_embeddings = text_embeddings.repeat(1, self.num_images, 1)
            text_embeddings = text_embeddings.view(bs_embed * self.num_images, seq_len, -1)

            max_length = text_input_ids.shape[-1]
            uncond_input_ids = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.type(torch.int32).to(self.device)
            uncond_input_ids_inp = cuda.DeviceView(ptr=uncond_input_ids.data_ptr(), shape=uncond_input_ids.shape, dtype=np.int32)
            uncond_embeddings = self.runEngine('clip', {"input_ids": uncond_input_ids_inp})['text_embeddings']

            # Duplicate unconditional embeddings for each generation per prompt
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, self.num_images, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * self.num_images, seq_len, -1)

            # Concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes for classifier free guidance
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            if self.denoising_fp16:
                text_embeddings = text_embeddings.to(dtype=torch.float16)

            cudart.cudaEventRecord(events['clip-stop'], 0)
            if self.nvtx_profile:
                nvtx.end_range(nvtx_clip)

            cudart.cudaEventRecord(events['denoise-start'], 0)
            for step_index, timestep in enumerate(tqdm.tqdm(self.scheduler.timesteps)):
                if self.nvtx_profile:
                    nvtx_latent_scale = nvtx.start_range(message='latent_scale', color='pink')
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2)
                # LMSDiscreteScheduler.scale_model_input()
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, step_index)
                if self.nvtx_profile:
                    nvtx.end_range(nvtx_latent_scale)

                # predict the noise residual
                if self.nvtx_profile:
                    nvtx_unet = nvtx.start_range(message='unet', color='blue')
                dtype = np.float16 if self.denoising_fp16 else np.float32
                if timestep.dtype != torch.float32:
                    timestep_float = timestep.float()
                else:
                    timestep_float = timestep
                sample_inp = cuda.DeviceView(ptr=latent_model_input.data_ptr(), shape=latent_model_input.shape, dtype=np.float32)
                timestep_inp = cuda.DeviceView(ptr=timestep_float.data_ptr(), shape=timestep_float.shape, dtype=np.float32)
                embeddings_inp = cuda.DeviceView(ptr=text_embeddings.data_ptr(), shape=text_embeddings.shape, dtype=dtype)
                noise_pred = self.runEngine(self.unet_model_key, {"sample": sample_inp, "timestep": timestep_inp, "encoder_hidden_states": embeddings_inp})['latent']
                if self.nvtx_profile:
                    nvtx.end_range(nvtx_unet)

                if self.nvtx_profile:
                    nvtx_latent_step = nvtx.start_range(message='latent_step', color='pink')
                # Perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, latents, step_index, timestep)

                if self.nvtx_profile:
                    nvtx.end_range(nvtx_latent_step)

            latents = 1. / 0.18215 * latents
            cudart.cudaEventRecord(events['denoise-stop'], 0)

            if self.nvtx_profile:
                nvtx_vae = nvtx.start_range(message='vae', color='red')
            cudart.cudaEventRecord(events['vae-start'], 0)
            sample_inp = cuda.DeviceView(ptr=latents.data_ptr(), shape=latents.shape, dtype=np.float32)
            images = self.runEngine('vae', {"latent": sample_inp})['images']
            cudart.cudaEventRecord(events['vae-stop'], 0)
            if self.nvtx_profile:
                nvtx.end_range(nvtx_vae)

            torch.cuda.synchronize()
            e2e_toc = time.perf_counter()

            # Save image
            #image_name_prefix = 'sd-'+('fp16' if self.denoising_fp16 else 'fp32')+''.join(set(['-'+prompt[i].replace(' ','_')[:10] for i in range(batch_size)]))+'-'
            images = ((images + 1) * 255 / 2).clamp(0, 255).detach().permute(0, 2, 3, 1).round().type(torch.uint8).cpu().numpy()
            return images


def compile_trt(model_path,hf_auth_token):

    engine_dir = f'engine/{model_path}'
    onnx_dir = "onnx"
    output_dir = "output"
    isExist = os.path.exists(engine_dir.split('/')[0])
    if not isExist:
        os.makedirs(engine_dir.split('/')[0])
    isExist = os.path.exists(os.path.join(engine_dir.split('/')[0],engine_dir.split('/')[1]))
    if not isExist:
        os.makedirs(os.path.join(engine_dir.split('/')[0],engine_dir.split('/')[1]))
    isExist = os.path.exists(engine_dir)
    if not isExist:
        os.makedirs(engine_dir)
    isExist = os.path.exists(onnx_dir)
    if not isExist:
        os.makedirs(onnx_dir)
    isExist = os.path.exists(output_dir)
    if not isExist:
        os.makedirs(output_dir)
    # Register TensorRT plugins
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    # Initialize demo
    demo = DemoDiffusion(
        model_path=model_path,
        denoising_steps=10,
        denoising_fp16=True,
        output_dir="output",
        scheduler="LMSD",
        hf_token=hf_auth_token,
        verbose=False,
        nvtx_profile=False,
        max_batch_size=16,
    )

    demo.buildOnlyEngines(engine_dir, onnx_dir, 16, 
        opt_batch_size=1, opt_image_height=512, opt_image_width=512, \
        force_export=False, force_optimize=False, \
        force_build=False, minimal_optimization=False, \
        static_batch=False, static_shape=True, \
        enable_preview=False)


def infer_trt(saving_path, model, prompt, neg_prompt, img_height, img_width, num_inference_steps, guidance_scale, num_images_per_prompt, seed=None):
    
    print("[I] Initializing StableDiffusion with TensorRT Plugins")
    args = parseArgs()

    args.output_dir=saving_path
    args.prompt=[prompt]
    args.model_path=model
    args.height=img_height
    args.width=img_width
    args.repeat_prompt=num_images_per_prompt
    args.denoising_steps=num_inference_steps
    args.seed=seed
    args.guidance_scale=guidance_scale
    args.negative_prompt=[neg_prompt]
    
    print('Seed :', args.seed)
    
    args.engine_dir = os.path.join(args.engine_dir, args.model_path)

    isExist = os.path.exists(args.engine_dir.split('/')[0])
    if not isExist:
        os.makedirs(args.engine_dir.split('/')[0])
    isExist = os.path.exists(os.path.join(args.engine_dir.split('/')[0],args.engine_dir.split('/')[1]))
    if not isExist:
        os.makedirs(os.path.join(args.engine_dir.split('/')[0],args.engine_dir.split('/')[1]))
    isExist = os.path.exists(args.engine_dir)
    if not isExist:
        os.makedirs(args.engine_dir)
    isExist = os.path.exists(args.onnx_dir)
    if not isExist:
        os.makedirs(args.onnx_dir)
    isExist = os.path.exists(args.output_dir)
    if not isExist:
        os.makedirs(args.output_dir)
    
    
    # Process prompt
    # if not isinstance(args.prompt, list):
    #     raise ValueError(f"`prompt` must be of type `str` or `str` list, but is {type(args.prompt)}")
    print('String :', args.prompt, type(args.prompt))
    prompt = args.prompt * args.repeat_prompt

    if not isinstance(args.negative_prompt, list):
        raise ValueError(f"`--negative-prompt` must be of type `str` or `str` list, but is {type(args.negative_prompt)}")
    if len(args.negative_prompt) == 1:
        negative_prompt = args.negative_prompt * len(prompt)
    else:
        negative_prompt = args.negative_prompt

    max_batch_size = 16
    if args.build_dynamic_shape:
        max_batch_size = 4

    if len(prompt) > max_batch_size:
        raise ValueError(f"Batch size {len(prompt)} is larger than allowed {max_batch_size}. If dynamic shape is used, then maximum batch size is 4")

    # Validate image dimensions
    image_height = args.height
    image_width = args.width
    if image_height % 8 != 0 or image_width % 8 != 0:
        raise ValueError(f"Image height and width have to be divisible by 8 but specified as: {image_height} and {image_width}.")

    # Register TensorRT plugins
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    # Initialize demo
    demo = DemoDiffusion(
        model_path=args.model_path,
        denoising_steps=args.denoising_steps,
        denoising_fp16=(args.denoising_prec == 'fp16'),
        output_dir=args.output_dir,
        scheduler=args.scheduler,
        hf_token=args.hf_token,
        verbose=args.verbose,
        nvtx_profile=args.nvtx_profile,
        max_batch_size=max_batch_size,
    )

    # Load TensorRT engines and pytorch modules
    alreadyCompiled = os.path.exists("engine/CompVis/stable-diffusion-v1-4/clip.plan")
    if not alreadyCompiled:
        demo.buildOnlyEngines(args.engine_dir, args.onnx_dir, args.onnx_opset, 
        opt_batch_size=1, opt_image_height=image_height, opt_image_width=image_width, \
        force_export=args.force_onnx_export, force_optimize=args.force_onnx_optimize, \
        force_build=args.force_engine_build, minimal_optimization=args.onnx_minimal_optimization, \
        static_batch=args.build_static_batch, static_shape=not args.build_dynamic_shape, \
        enable_preview=args.build_preview_features)
    
    
    
    demo.loadEngines(args.engine_dir, args.onnx_dir, args.onnx_opset, 
        opt_batch_size=len(prompt), opt_image_height=image_height, opt_image_width=image_width, \
        force_export=args.force_onnx_export, force_optimize=args.force_onnx_optimize, \
        force_build=args.force_engine_build, minimal_optimization=args.onnx_minimal_optimization, \
        static_batch=args.build_static_batch, static_shape=not args.build_dynamic_shape, \
        enable_preview=args.build_preview_features)
    demo.loadModules()

    print("[I] Warming up ..")
    for _ in range(args.num_warmup_runs):
        images = demo.infer(prompt, negative_prompt, args.height, args.width, warmup=True, verbose=False, seed=args.seed)

    print("[I] Running StableDiffusion pipeline")
    if args.nvtx_profile:
        cudart.cudaProfilerStart()
    pipeline_time = demo.infer(prompt, negative_prompt, args.height, args.width, verbose=args.verbose, seed=args.seed)
    if args.nvtx_profile:
        cudart.cudaProfilerStop()

    demo.teardown()
    gc.collect()
    return pipeline_time