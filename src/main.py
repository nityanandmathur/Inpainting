import argparse
import os
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

warnings.filterwarnings("ignore")

class inpainter:
    '''
    Class for inpainting images
    Functions:
        1. generate_mask - Function to generate mask for images.
        2. plot - Function to plot the image and mask
        3. inpaint - Inpaint function
    '''
    def __init__(self, prompt, image_path, upscale=False):
        self.prompt = prompt
        self.image_path = image_path
        self.image = Image.open(self.image_path)
        self.model = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float32
        )
        self.mask = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save_dir(self):
        '''
        Function to save the inpainted image

        @Returns:
            1. Path to save image, mask, generated image
        '''
        image_name = self.image_path.split("/")[-1]
        image_name = image_name.replace(".jpg" if ".jpg" in image_name else ".png", "")
        image_folder = f"../data/generated/{image_name}"

        if not os.path.exists(image_folder):
            os.mkdir(image_folder)
        return image_folder + "/"
    
    def upscale(self):
        '''
        Function to add extra whitespace to the image.
        "FOR IMAGES COVERING MOST OF THE PARTS OF THE IMAGE"

        @Returns:
            1. Upscaled image with size (2*max(image_size), 2*max(image_size))
        '''
        size = (2*max(self.image.size),)*2
        upscaled_image = Image.new('RGB', size, (255,255,255))
        upscaled_image.paste(self.image, tuple(map(lambda x:int((x[0]-x[1])/2), zip(size, self.image.size))))
        return upscaled_image

    def generate_mask(self):
        '''
        Function to generate mask for images. 
        "WORKS ONLY FOR WHITE BACKGROUND IMAGES"

        @Arguments:
            1. Image - Image to be inpainted

        @Returns:
            1. Mask - Mask for the image
        '''

        #lower and upper bounds for white color -> region having colour except than white will be black
        lower = np.array([220, 220, 220])
        upper = np.array([255, 255, 255])

        image_cv2 = np.array(self.image)
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

        mask = cv2.inRange(image_cv2, lower, upper)
        mask = Image.fromarray(mask)

        return mask

    def plot(self):
        '''
        Function to plot the image and mask

        @Arguments:
            1. Image - Image to be inpainted
            2. Mask - Mask for the image

        @Returns:
            1. Combined image path
        '''
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(self.image)
        ax[0].set_title("Image")
        ax[0].set_xticks([])  # remove x-axis labels
        ax[0].set_yticks([])  # remove y-axis labels
        ax[1].imshow(self.mask, cmap="gray")
        ax[1].set_title("Mask")
        ax[1].set_xticks([])  # remove x-axis labels
        ax[1].set_yticks([])  # remove y-axis labels

        path = self.save_dir()
        plt.savefig(f"{path}im_mask.png")

        print(f"Image and Mask saved at: {path}im_mask.png")

    def inpaint(self):
        '''
        Inpaint function

        @Arguments:
            1. Prompt - Text prompt describing the image to be inpainted
            2. Image path - Path to the image to be inpainted

        @Returns:
            1. Inpainted image Path
        '''
        if self.upscale:
            self.image = self.upscale()

        path = self.save_dir()

        self.mask = self.generate_mask()
        self.plot()

        image = self.model(prompt=self.prompt, image=self.image, mask_image=self.mask, torch_device=self.device).images[0]

        image.save(f"{path}{self.prompt}.png")
        print(f'Inpainted image saved at: {path}{self.prompt}.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--upscale", type=bool, default=False)

    args = parser.parse_args()

    inpainter = inpainter(args.prompt, args.image_path, args.upscale)
    inpainter.inpaint()
