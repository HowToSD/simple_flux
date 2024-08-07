# Simple Flux
Simple Flux offers a lightweight web interface to Flux using Gradio.


<figure>
  <img src="docs/resources/main_ui.jpg" alt="UI">
  <figcaption>Main UI</figcaption>
</figure>

# Hardware & software requirements
Tested on Ubuntu with NVIVIA RTX 4090 and GTX 1070 (8GB GPU RAM).

## GPU Memory
If your GPU has less than 16GB GPU memory, you need to specify --low_mem when you start the script.
Max GPU memory consumption that was observed when --low_mem was specified was 5.88GB, so 6GB GPU RAM may work.

## System RAM
Tested on 64GB RAM hosts.
I learned from a user that he/she was able to run it with 40GB RAM on Windows 11. It is unclear if you can run it on a machine with less RAM than that.

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
