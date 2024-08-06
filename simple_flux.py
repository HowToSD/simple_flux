"""
This script provides an ImageGenerator class to generate and navigate through images using Flux.1-schnell.
It includes a Gradio interface for interacting with the image generation and navigation functionalities.

Classes:
    ImageGenerator

Functions:
    main()

Copyright (c) 2024 Hideyuki Inada

# Credit
* Code to interact with the Flux model is based on a github user's code. Link to the code is available below:
  https://www.reddit.com/r/StableDiffusion/comments/1ehl4as/how_to_run_flux_8bit_quantized_locally_on_your_16/

* Further memory footprint reduction idea is from [3]

# References
[1] black-forest-labs/FLUX.1-schnell. https://huggingface.co/black-forest-labs/FLUX.1-schnell
[2] Sayak Paul, David Corvoysier. Memory-efficient Diffusion Transformers with Quanto and Diffusers. https://huggingface.co/blog/quanto-diffusers
[3] https://huggingface.co/black-forest-labs/FLUX.1-schnell/discussions/5
"""

import os
import logging
import time
import json
import gradio as gr
import torch
import optimum.quanto
from optimum.quanto import freeze, qfloat8, quantize
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast

from PIL.PngImagePlugin import PngInfo
from image_navigator import ImageNavigator
from image_utils import read_image
from file_utils import get_image_file_list_in_directory
from typing import List, Optional, Tuple

SKIP_NUM = 10
OUTPUTS_DIR = "outputs"
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
MODEL_REVISION = "refs/pr/1"
TEXT_MODEL_ID = "openai/clip-vit-large-patch14"
MODEL_DATA_TYPE = torch.bfloat16

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

class ImageGenerator:
    """
    A class to generate and navigate through images using Stable Diffusion.

    Methods:
        rescan_output_directory() -> None
        generate_button_handler(prompt: str, steps: int, guidance_scale: float) -> Tuple
        setup_ui(file_paths: Optional[List[str]] = None) -> gr.Blocks
        __call__() -> None
    """

    def quantize_and_freeze(self,
                            model:torch.nn.Module,
                            weights:optimum.quanto.tensor=qfloat8) -> torch.nn.Module:
        """
        Quantizes and freezes the model to reduce memory footprint.

        Args:
            model (torch.nn.Module): Model to quantize and freeze.
            weights (optimum.quanto.tensor.qtype, optional): Target data type. Defaults to quanto_tensor.qfloat8.

        Returns:
            torch.nn.Module: The quantized and frozen model.
        """
        quantize(model, weights)
        freeze(model)
        return model

    def __init__(self, outputs_dir: Optional[str] = None,
                 low_mem: Optional[bool]=False):
        """
        Initializes the ImageGenerator with the specified checkpoint and output directories.

        Args:
            outputs_dir (Optional[str]): The directory to save generated images.
            low_mem (Optional[bool]): Use less GPU memory.
        """
        self.outputs_dir = outputs_dir
        self.file_paths = None
        self.current_index = 0
        self.pipe = None

        # Initialize required models
        # Text encoders
        # 1
        logger.info("Instantiating CLIP text tokenizer and model")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            TEXT_MODEL_ID , torch_dtype=MODEL_DATA_TYPE)
        self.text_encoder = CLIPTextModel.from_pretrained(
            TEXT_MODEL_ID , torch_dtype=MODEL_DATA_TYPE)

        # 2
        logger.info("Instantiating T5 text tokenizer and model")
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(
            MODEL_ID, subfolder="tokenizer_2", torch_dtype=MODEL_DATA_TYPE,
            revision=MODEL_REVISION)
        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            MODEL_ID, subfolder="text_encoder_2", torch_dtype=MODEL_DATA_TYPE,
            revision=MODEL_REVISION)

        # Transformers
        logger.info("Instantiating scheduler")
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            MODEL_ID, subfolder="scheduler", revision=MODEL_REVISION)
        logger.info("Instantiating transformer")
        self.transformer = FluxTransformer2DModel.from_pretrained(
            MODEL_ID, subfolder="transformer", torch_dtype=MODEL_DATA_TYPE,
            revision=MODEL_REVISION)

        # VAE
        logger.info("Instantiating VAE")
        self.vae = AutoencoderKL.from_pretrained(
            MODEL_ID, subfolder="vae", torch_dtype=MODEL_DATA_TYPE,
            revision=MODEL_REVISION)

        if low_mem is False:
            logger.info("Quantizing T5 to 8 bits")
            self.quantize_and_freeze(self.text_encoder_2)

            logger.info("Quantizing transformer to 8 bits")
            self.quantize_and_freeze(self.transformer)

        # Create a pipeline without T5 and transformer
        self.pipe = FluxPipeline(
            scheduler=self.scheduler,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            text_encoder_2=None,
            tokenizer_2=self.tokenizer_2,
            vae=self.vae,
            transformer=None
        )

        # Set to the quantized version
        self.pipe.text_encoder_2 = self.text_encoder_2
        self.pipe.transformer = self.transformer

        if low_mem:
            logging.info("Using low memory mode")
            self.pipe.vae.enable_tiling()
            self.pipe.vae.enable_slicing()
            self.pipe.enable_sequential_cpu_offload()
        else:
            logging.info("Using standard memory mode")
            self.pipe.enable_model_cpu_offload()

        if outputs_dir is None:
            raise ValueError("Output directory is not specified.")
        os.makedirs(outputs_dir, exist_ok=True)
 
        # Full path, file name, creation time
        self.file_paths = get_image_file_list_in_directory(self.outputs_dir)
        self.nav = ImageNavigator(
            app=self,
            file_paths=self.file_paths,
            skip_num=SKIP_NUM,
            callback=read_image
        )
        self.gradio_app = self.setup_ui()

    def rescan_output_directory(self) -> None:
        """
        Rescans the output directory. This is to be called by the navigation class when an image is deleted.
        """
        self.file_paths = get_image_file_list_in_directory(self.outputs_dir)

    def generate_button_handler(self, prompt: str, steps: int = 4, guidance_scale: float = 0.0) -> Tuple:
        """
        Generates an image based on the provided prompts, steps, and guidance scale.

        Args:
            prompt (str): The positive prompt for image generation.
            steps (int): The number of inference steps.
            guidance_scale (float): The guidance scale for image generation.

        Returns:
            Tuple: A tuple containing the generated image, file information, and metadata.
        """
        image = self.pipe(
            prompt=prompt,
            width=1024,
            height=1024,
            num_inference_steps=steps,
            guidance_scale=guidance_scale
        ).images[0]

        logging.info(
            "Image generated. Max GPU memory allocated (GB): " + \
            f"{torch.cuda.max_memory_allocated() / (1024 ** 3)}")

        file_name = str(time.time())
        output_path = os.path.join(self.outputs_dir, file_name + ".png")

        generation_parameters = {
            "time": time.time(),
            "positive_prompt": prompt,
            "sampling_iterations": steps,
            "cfg": guidance_scale
        }
        str_generation_params = json.dumps(generation_parameters)
        metadata = PngInfo()
        metadata.add_text("generation_data", str_generation_params)
        image.save(output_path, pnginfo=metadata)

        # Update paths
        self.file_paths.append(output_path)
        self.current_index = len(self.file_paths) - 1  # Set to last
        self.nav.update_file_paths(self.file_paths, new_current_index=self.current_index)
        return read_image(output_path)  # This is to return consistent meta data

    def setup_ui(self, file_paths: Optional[List[str]] = None) -> gr.Blocks:
        """
        Sets up the Gradio user interface for the image generator.

        Args:
            file_paths (Optional[List[str]]): The list of image file paths.

        Returns:
            gr.Blocks: The Gradio Blocks object representing the UI.
        """
        css = "#output_image {height:800px}"
        with gr.Blocks(analytics_enabled=False, css=css) as app:
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        prompt = gr.TextArea(label="Positive prompt:", elem_id="small-textarea", lines=4, max_lines=4)
                    with gr.Row():
                        guidance_scale = gr.Slider(value=0.0, minimum=0.0, maximum=30.0, step=0.1, label="Guidance scale")
                    with gr.Row():
                        steps = gr.Slider(value=4, minimum=1, maximum=100, step=1, label="Steps")
                    with gr.Row():
                        generate_btn = gr.Button("Generate")
                with gr.Column():
                    with gr.Row():
                        global image_field
                        image_field = gr.Image(label="Output Image", elem_id="output_image")
                    with gr.Row():
                        skip_backward_btn = gr.Button("<<")
                        prev_btn = gr.Button("<")
                        next_btn = gr.Button(">")
                        skip_forward_btn = gr.Button(">>")
                        first_btn = gr.Button("First")
                        last_btn = gr.Button("Last")
                    with gr.Row():
                        delete_btn = gr.Button("Delete")
                    with gr.Row():
                        gr.Markdown("File info:")
                        file_info = gr.Markdown()
                    with gr.Row():
                        gr.Markdown("File metadata:")
                        metadata = gr.HTML()

            # Define the list of fields where output of a function is set.
            # This needs to match the return value for the target function.
            output = [
                image_field, file_info, metadata
            ]

            # Event handlers
            generate_btn.click(fn=self.generate_button_handler,
                            inputs=[prompt, steps, guidance_scale],
                            outputs=output,
                            api_name="generate")

            # Navigation
            skip_backward_btn.click(fn=self.nav.skip_backward_button_handler,
                            inputs=None,
                            outputs=output,
                            api_name="skip_backward")
            prev_btn.click(fn=self.nav.prev_button_handler,
                            inputs=None,
                            outputs=output,
                            api_name="prev")
            next_btn.click(fn=self.nav.next_button_handler,
                            inputs=None,
                            outputs=output,
                            api_name="next")
            skip_forward_btn.click(fn=self.nav.skip_forward_button_handler,
                            inputs=None,
                            outputs=output,
                            api_name="skip_forward")
            first_btn.click(fn=self.nav.first_button_handler,
                            inputs=None,
                            outputs=output,
                            api_name="first")
            last_btn.click(fn=self.nav.last_button_handler,
                            inputs=None,
                            outputs=output,
                            api_name="last")
            delete_btn.click(fn=self.nav.delete_button_handler,
                            inputs=None,
                            outputs=output,
                            api_name="delete")
            
        return app

    def __call__(self) -> None:
        """
        Launches the Gradio application.
        """
        self.gradio_app.launch()

def main() -> None:
    """
    The main function to create an instance of ImageGenerator and launch the application.

    To run in normal memory mode:
    ```
    python simple_flux.py --low_mem
    ```

    To run in low memory mode:
    ```
    python simple_flux.py --low_mem
    ```
    """
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--low_mem', action='store_true', help='Use less GPU memory')
    args = parser.parse_args()

    image_generator = ImageGenerator(
        outputs_dir=OUTPUTS_DIR,
        low_mem=args.low_mem
    )
    image_generator()


if __name__ == "__main__":
    main()
