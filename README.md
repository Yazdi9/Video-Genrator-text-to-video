# Video Generation from text

**[Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation](https://arxiv.org/abs/2212.11565)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/showlab/Tune-A-Video/blob/main/notebooks/Tune-A-Video.ipynb)

## Setup

### Requirements

```shell
pip install -r requirements.txt
```

### Weights

**[Stable Diffusion]** [Stable Diffusion](https://arxiv.org/abs/2112.10752) is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input. The pre-trained Stable Diffusion models can be downloaded from Hugging Face (e.g., [Stable Diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4), [v2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)). You can also use fine-tuned Stable Diffusion models trained on different styles (e.g, [Modern Disney](https://huggingface.co/nitrosocke/mo-di-diffusion), [Redshift](https://huggingface.co/nitrosocke/redshift-diffusion), etc.).

**[DreamBooth]** [DreamBooth](https://dreambooth.github.io/) is a method to personalize text-to-image models like Stable Diffusion given just a few images (3~5 images) of a subject. Tuning a video on DreamBooth models allows personalized text-to-video generation of a specific subject. There are some public DreamBooth models available on [Hugging Face](https://huggingface.co/sd-dreambooth-library) (e.g., [mr-potato-head](https://huggingface.co/sd-dreambooth-library/mr-potato-head)). You can also train your own DreamBooth model following [this training example](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth). 


## Usage

### Training

To fine-tune the text-to-image diffusion models for text-to-video generation, run this command:

```bash
accelerate launch train_tuneavideo.py --config="configs/man-skiing.yaml"
```

### Inference

Once the training is done, run inference:

```python
from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid
import torch

pretrained_model_path = "./checkpoints/stable-diffusion-v1-4"
my_model_path = "./outputs/man-skiing"
unet = UNet3DConditionModel.from_pretrained(my_model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')
pipe = TuneAVideoPipeline.from_pretrained(pretrained_model_path, unet=unet, torch_dtype=torch.float16).to("cuda")
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()

prompt = "spider man is skiing"
ddim_inv_latent = torch.load(f"{my_model_path}/inv_latents/ddim_latent-500.pt").to(torch.float16)
video = pipe(prompt, latents=ddim_inv_latent, video_length=24, height=512, width=512, num_inference_steps=50, guidance_scale=12.5).videos

save_videos_grid(video, f"./{prompt}.gif")
```

## Results

### Pretrained T2I (Stable Diffusion)

<table class="center">
<tr>
  <td style="text-align:center;"><b>Input Video</b></td>
  <td style="text-align:center;" colspan="2"><b>Output Video</b></td>
</tr>
  
<tr>
  <td><img src="https://user-images.githubusercontent.com/33378412/227791032-301ad016-baa5-48e7-bee6-53eecff5c39f.gif"></td>
  <td><img src="https://user-images.githubusercontent.com/33378412/227790574-41828540-0a0e-4617-9751-4cacd83212f5.gif"></td>
  <td><img src="https://user-images.githubusercontent.com/33378412/227790190-a51b3632-e3d6-402d-ac48-f458e28ee3bb.gif"></td>
</tr>

<tr>
  <td width=25% style="text-align:center;color:gray;">"A man is skiing"</td>
  <td width=25% style="text-align:center;">"Wonder Woman, wearing a cowboy hat, is skiing"</td>
  <td width=25% style="text-align:center;">"A little girl is skiing "</td>
</tr>
     
<tr>
  <td><img src="https://tuneavideo.github.io/assets/data/rabbit-watermelon.gif"></td>
  <td><img src="https://user-images.githubusercontent.com/33378412/227790611-0a788ac2-9b2b-4267-b436-13bd1324b774.gif"></td>
  <td><img src="https://user-images.githubusercontent.com/33378412/227790211-964f8fda-133c-4bca-b623-fa49f4f37caf.gif"></td>              
 
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A rabbit is eating a watermelon"</td>
  <td width=25% style="text-align:center;">"A cat is eating a watermelon on the table"</td>
  <td width=25% style="text-align:center;">"A puppy is eating a cheeseburger on the table, comic style"</td>
 </tr>

<tr>
  <td><img src="https://tuneavideo.github.io/assets/data/car-turn.gif"></td>
  <td><img src="https://user-images.githubusercontent.com/33378412/227790590-c1c13d51-7409-4f3c-914f-9d1ad422bc30.gif"></td>              
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/car-turn/car-snow.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A jeep car is moving on the road"</td>
  <td width=25% style="text-align:center;">"A car is moving on the road, cartoon style"</td>
  <td width=25% style="text-align:center;">"A car is moving on the snow"</td>

 
</tr>


<!-- <tr>
  <td><img src="https://tuneavideo.github.io/assets/data/lion-roaring.gif"></td>
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/lion-roaring/tiger-roar.gif"></td>
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/lion-roaring/lion-vangogh.gif"></td>              
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/lion-roaring/wolf-nyc.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A lion is roaring"</td>
  <td width=25% style="text-align:center;">"A tiger is roaring"</td>
  <td width=25% style="text-align:center;">"A lion is roaring, Van Gogh style"</td>
  <td width=25% style="text-align:center;">"A wolf is roaring in New York City"</td>
</tr> -->

</table>

### Pretrained T2I (personalized DreamBooth)

<img src="https://tuneavideo.github.io/assets/results/tuneavideo/modern-disney/modern-disney.png" width="240px"/>  

<table class="center">
<tr>
  <td style="text-align:center;"><b>Input Video</b></td>
  <td style="text-align:center;" colspan="3"><b>Output Video</b></td>
</tr>
<tr>
  <td><img src="https://tuneavideo.github.io/assets/data/bear-guitar.gif"></td>
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/modern-disney/bear-guitar/rabbit.gif"></td>      
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/modern-disney/bear-guitar/prince.gif"></td>
  <td><img src="https://tuneavideo.github.io/assets/results/tuneavideo/modern-disney/bear-guitar/princess.gif"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"A bear is playing guitar"</td>
  <td width=25% style="text-align:center;">"A rabbit is playing guitar, modern disney style"</td>
  <td width=25% style="text-align:center;">"A handsome prince is playing guitar, modern disney style"</td>
  <td width=25% style="text-align:center;">"A magic princess with sunglasses is playing guitar on the stage, modern disney style"</td>
</tr>
</table>



