import argparse
import os
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from tqdm import tqdm

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
        image_folder = f"../data/video/{image_name}"

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
        size = (512+50, 512+50)
        upscaled_image = Image.new('RGB', size, (255,255,255))
        upscaled_image.paste(self.image, tuple(map(lambda x:int((x[0]-x[1])/2), zip(size, self.image.size))))
        resized_image = upscaled_image.resize((512,512))
        return resized_image

    def generate_mask(self):
        '''
        Function to generate mask for images. 
        "WORKS ONLY FOR WHITE BACKGROUND IMAGES"

        @Arguments:
            1. Image - Image to be inpainted

        @Returns:
            1. Mask - Mask for the image
        '''
        # Create a black image of the same size
        mask = np.ones(self.image.size)*255

        # Define points for rectangle
        pt1 = (25, 25)
        pt2 = (mask.shape[1] - 25, mask.shape[0] - 25)

        # Draw a white rectangle on the black image
        cv2.rectangle(mask, pt1, pt2, (0, 0, 0), -1)
        mask = Image.fromarray(mask)

        return mask

    def plot(self, i):
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
        plt.savefig(f"{path}{i}_im_mask.png")

        print(f"Image and Mask saved at: {path}{i}_im_mask.png")

    def inpaint(self):
        '''
        Inpaint function

        @Arguments:
            1. Prompt - Text prompt describing the image to be inpainted
            2. Image path - Path to the image to be inpainted

        @Returns:
            1. Inpainted image Path
        '''
        path = self.save_dir()

        for i in tqdm(range(50,500,50)):
            self.image = self.upscale()
            self.mask = self.generate_mask()
            self.plot(i//50)
            self.image = self.model(prompt=self.prompt, image=self.image, mask_image=self.mask, torch_device=self.device).images[0]
            self.image.save(f"{path}{i//50}.png")
            print(f'Inpainted frame saved at: {path}{i}.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)

    args = parser.parse_args()

    inpainter = inpainter(args.prompt, args.image_path)
    inpainter.inpaint()
