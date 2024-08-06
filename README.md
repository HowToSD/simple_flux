# Simple Flux
Simple Flux offers a lightweight web interface to Flux using Gradio.


<figure>
  <img src="docs/resources/main_ui.jpg" alt="UI">
  <figcaption>Main UI</figcaption>
</figure>

# Hardware & software requirements
This was only tested on Ubuntu with NVIVIA RTX 4090.
Using nvidia-smi command, I see 13986MiB for the Python process during image generation.
So you can try if you have 16GB GPU RAM.

# How to install Simple Flux
Currently, only systems with NVIDIA GPU and CUDA are supported.
I am using Linux and these steps have not been tested on Windows yet.

### Overview
1. Copy files from github
1. Set up conda environment
1. Install Pytorch
1. Install other python packages

### Steps
1. Copy files from github

   Open the terminal and go to a directory that you want to install Simple Flux.
   Type:
   ```
   git clone https://github.com/HowToSD/simple_flux.git
   ```

2. Set up conda environment

    Run the following. Note that if your system does not have conda, you need to install it first.

    ```
    conda create -n simpleflux python=3.10
    conda activate simpleflux
    ```

3. Install Pytorch
   
    ```
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```
    For more information, refer to the installation section at https://pytorch.org/.


4. Install other python packages

    ```
    pip install -r requirements.txt
    ```

5. Start Simple Flux
   Type:
   ```
   python simple_flux.py
   ```

   If you have low GPU memory, type:
   ```
   python simple_flux.py --low_mem
   ```

   This should start Gradio.

5.  Using your browser, access the URL displayed in the above.
