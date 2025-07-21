import requests
import base64
import io
from PIL import Image
import torch
import logging
import os
import hashlib
import folder_paths
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_unique_hash(string):
    hash_object = hashlib.sha1(string.encode())
    unique_hash = hash_object.hexdigest()
    return unique_hash

class preview_text:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True, "dynamicPrompts": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "ComfyUI/Pixtral Large"

    def run(self, text):
        if not isinstance(text, list):
            text = [text]
            
        # Type correction
        texts = []
        for t in text:
            if not isinstance(t, str):
                t = str(t)
            texts.append(t)

        return {"ui": {"text": texts}, "text": texts}
        
class MultiImagesInput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 30, "step": 1}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "combine"
    CATEGORY = "ComfyUI/Pixtral Large"

    DESCRIPTION = """
    Creates an image batch from multiple images.
    Pixtral Large can handle up to 30 high-resolution images in its 128K context window.
    Set the number of inputs with **inputcount** and click update.
    """

    def combine(self, inputcount, **kwargs):
        from nodes import ImageBatch

        image_batch_node = ImageBatch()
        images = [kwargs[f"image_{i}"] for i in range(1, inputcount + 1) if f"image_{i}" in kwargs]
        
        if len(images) < 2:
            raise ValueError(f"At least 2 images are required. Only {len(images)} provided.")
        
        if len(images) > 30:
            raise ValueError(f"Pixtral Large supports up to 30 images. {len(images)} provided.")
        
        result = images[0]
        for image in images[1:]:
            if image is not None:
                (result,) = image_batch_node.batch(result, image)
        
        return (result,)

class ComfyUIPixtralLarge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "Describe the image"}),
                "images": ("IMAGE", {"multiple": True}),
                "api_key": ("STRING", {"default": "Enter your Mistral API key here"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.5, "step": 0.1}),
                "maximum_tokens": ("INT", {"default": 4096, "min": 1, "max": 32768, "step": 1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "ComfyUI/Pixtral Large"

    def process(self, prompt, images, api_key, temperature, maximum_tokens, top_p):
        try:
            image_urls = []
            for image in images:
                logger.info(f"Processing image. Type: {type(image)}, Shape: {getattr(image, 'shape', 'No shape')}")

                if isinstance(image, torch.Tensor):
                    image = image.squeeze().cpu().numpy()

                if len(image.shape) == 3 and image.shape[0] in [1, 3, 4]:
                    image = image.transpose(1, 2, 0)

                if len(image.shape) == 2:
                    pil_image = Image.fromarray((image * 255).astype('uint8'), 'L')
                elif len(image.shape) == 3:
                    if image.shape[2] == 1:
                        pil_image = Image.fromarray((image[:, :, 0] * 255).astype('uint8'), 'L')
                    elif image.shape[2] == 3:
                        pil_image = Image.fromarray((image * 255).astype('uint8'), 'RGB')
                    elif image.shape[2] == 4:
                        pil_image = Image.fromarray((image * 255).astype('uint8'), 'RGBA').convert('RGB')
                    else:
                        raise ValueError(f"Unexpected number of channels: {image.shape[2]}")
                else:
                    raise ValueError(f"Unexpected image shape: {image.shape}")

                logger.info(f"Processed PIL Image size: {pil_image.size}")

                buffered = io.BytesIO()
                pil_image.save(buffered, format="JPEG", quality=95)
                base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                image_urls.append(f"data:image/jpeg;base64,{base64_image}")

            if len(image_urls) > 30:
                raise ValueError(f"Pixtral Large supports up to 30 images. {len(image_urls)} provided.")

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            content_list = [{"type": "text", "text": prompt}]
            for image_url in image_urls:
                content_list.append({"type": "image_url", "image_url": image_url})

            messages = [
                {
                    "role": "user",
                    "content": content_list
                }
            ]
            
            data = {
                "model": "pixtral-12b-2409",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": maximum_tokens,
                "top_p": top_p
            }

            logger.info("Sending request to Mistral API")
            response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=data)

            if response.status_code == 200:
                result = response.json()
                logger.info("Received successful response from Mistral API")
                return (result["choices"][0]["message"]["content"],)
            else:
                error_message = f"API Error: {response.status_code}, {response.text}"
                logger.error(error_message)
                return (error_message,)

        except Exception as e:
            error_message = f"Error in process method: {str(e)}"
            logger.exception(error_message)
            return (error_message,)

# Register all nodes
NODE_CLASS_MAPPINGS = {
    "ComfyUIPixtralLarge": ComfyUIPixtralLarge,
    "MultiImagesInput": MultiImagesInput,
    "preview_text": preview_text,
}

# Optional: Add descriptions for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUIPixtralLarge": "Pixtral Large",
    "MultiImagesInput": "Multi Images Input",
    "preview_text": "Preview Text"
}