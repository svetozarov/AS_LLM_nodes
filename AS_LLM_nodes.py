import os
import io
import time
import base64
import numpy as np
from PIL import Image
import google.generativeai as genai

################################################################################
# Node 1: AS_GeminiCaptioning
################################################################################

class AS_GeminiCaptioning:
    """
    Generates a descriptive text prompt from an image using the Gemini API.
    Accepts models Gemini 2.0 Flash, Gemini 2.0 Flash-Lite, Gemini 1.5 Flash, Gemini 1.5 Pro.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "IMAGE": ("IMAGE",),
                "PROMPT TYPE": (("SD1.5 – SDXL", "FLUX"), {"default": "SD1.5 – SDXL"}),
                "APY KEY PATH": ("STRING", {"default": ""}),
                "GEMINI MODEL": (
                    (
                        "Gemini 2.0 Flash",
                        "Gemini 2.0 Flash-Lite",
                        "Gemini 1.5 Flash",
                        "Gemini 1.5 Pro"
                    ),
                    {"default": "Gemini 2.0 Flash"}
                ),
            },
            "optional": {
                "PROMPT LENGTH": ("INT", {"default": 0, "defaultInput": True}),
                "PROMPT REFERENCE": ("STRING", {"default": "", "defaultInput": True}),
                "PROMPT STRUCTURE": ("STRING", {"default": "", "defaultInput": True}),
                "IGNORE": ("STRING", {"default": "", "defaultInput": True}),
                "EMPHASIS": ("STRING", {"default": "", "defaultInput": True}),
                "SAVE TO PATH": ("STRING", {"default": "", "defaultInput": True}),
                "TXT NAME": ("STRING", {"default": "", "defaultInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("RESULT PROMPT", "REQUEST TEXT", "LOG")
    FUNCTION = "gemini_caption"
    CATEGORY = "AS_LLM nodes"

    def gemini_caption(self, **kwargs):
        log = []
        image = kwargs.get("IMAGE")
        prompt_type = kwargs.get("PROMPT TYPE") or ""
        apy_key_path = kwargs.get("APY KEY PATH") or ""
        gemini_model = kwargs.get("GEMINI MODEL") or "Gemini 2.0 Flash"
        prompt_length = kwargs.get("PROMPT LENGTH")
        prompt_reference = kwargs.get("PROMPT REFERENCE") or ""
        prompt_structure = kwargs.get("PROMPT STRUCTURE") or ""
        ignore = kwargs.get("IGNORE") or ""
        emphasis = kwargs.get("EMPHASIS") or ""
        save_to_path = kwargs.get("SAVE TO PATH") or ""
        txt_name = kwargs.get("TXT NAME") or ""

        try:
            if isinstance(image, bytes):
                image_data = image
                img_obj = Image.open(io.BytesIO(image_data))
            elif hasattr(image, "read"):
                image_data = image.read()
                img_obj = Image.open(io.BytesIO(image_data))
            elif hasattr(image, "cpu") and hasattr(image, "detach"):
                tensor = image.cpu().detach().numpy()
                while tensor.ndim > 3 and tensor.shape[0] == 1:
                    tensor = tensor[0]
                if tensor.ndim == 3 and tensor.shape[0] in [1, 3]:
                    tensor = tensor.transpose(1, 2, 0)
                if tensor.max() <= 1:
                    tensor = (tensor * 255).astype("uint8")
                else:
                    tensor = tensor.astype("uint8")
                img_obj = Image.fromarray(tensor)
                buffer = io.BytesIO()
                img_obj.save(buffer, format="PNG")
                image_data = buffer.getvalue()
            elif isinstance(image, np.ndarray):
                while image.ndim > 3 and image.shape[0] == 1:
                    image = image[0]
                if image.ndim == 3 and image.shape[0] in [1, 3]:
                    image = image.transpose(1, 2, 0)
                if image.max() <= 1:
                    image = (image * 255).astype("uint8")
                else:
                    image = image.astype("uint8")
                img_obj = Image.fromarray(image)
                buffer = io.BytesIO()
                img_obj.save(buffer, format="PNG")
                image_data = buffer.getvalue()
            else:
                img_obj = image
                buffer = io.BytesIO()
                fmt = img_obj.format if img_obj.format else "PNG"
                img_obj.save(buffer, format=fmt)
                image_data = buffer.getvalue()

            detected_format = img_obj.format if img_obj.format else "PNG"
            log.append(f"Detected image format: {detected_format}")
        except Exception as e:
            log.append(f"Error processing image: {str(e)}")
            return ("", "", "\n".join(log))

        # Sample reference texts for SD1.5 – SDXL or FLUX
        SDXL_type = (
                    "It should be in CLIP-L comma-separated keywords SDXL prompt style: “Architecture, high-end modernist residential complex, minimalist design, open balconies, subtle architectural details, concrete and glass façades, elegant geometric volumes, tiered rooftop terraces, panoramic floor-to-ceiling windows, neutral-toned stone panels, tinted glass curtain walls, brushed metal railings, integrated with lush landscaping, manicured hedges, ornamental grasses, sculptural trees, wooden pathway leading to a reflective metal sphere, secluded urban oasis, tranquil environment, free from city noise, surrounded by curated greenery, creating a serene and balanced atmosphere, soft diffused lighting, overcast sky, early morning mist, gentle atmospheric glow, cinematic wide-angle perspective, symmetrical framing, high dynamic range, RAW photo, hyper-detailed, photorealistic”"
                )
        FLUX_type = (
                    "It should be in CLIP-G natural language FLUX prompt style: “Architecture, high-end modernist residential complex surrounded by lush greenery, designed with a minimalist and elegant aesthetic. The buildings feature a combination of natural stone and glass façades, with subtle architectural details and open balconies. A linear yet dynamic composition with clean geometric volumes, softened by carefully curated landscaping, including hedges, ornamental grasses, and small trees. The façade combines smooth concrete panels with floor-to-ceiling tinted glass windows, creating a refined balance of opacity and transparency. The outdoor space is defined by a wooden pathway meandering through a meticulously designed garden, leading towards a focal point—a polished metal sphere sculpture. Strategic lighting elements subtly highlight the landscape, while the gentle play of reflections on the glass surfaces enhances the depth of the environment. Set in a tranquil urban enclave, free from visual noise, framed by an overcast sky that casts a soft, diffused glow over the buildings. Early morning atmosphere with slight fog in the distance, lending an ethereal and cinematic quality to the scene. RAW photo, slightly elevated wide-angle viewpoint, long exposure, cinematic framing, balanced symmetry, moderate depth of field, high dynamic range, hyper-detailed, photorealistic rendering.”"
                )

        blocks = []
        blocks.append(
            "Give me a description of this image in English in the format of a text prompt for Stable Diffusion. "
            "It should be only the descriptive text according to the template I provided, "
            "without any additional comments from you. The text should be continuous, "
            "without headings, lists, or any other formatting."
        )
        ref_text = prompt_reference.strip() if prompt_reference.strip() else (
            SDXL_type if prompt_type == "SD1.5 – SDXL" else FLUX_type
        )
        blocks.append(
            "Use the following reference as an example of the prompt format and structure, "
            "showing how the text should look. Use it only as a reference, do not use its content "
            "for the current request unless it is present in the attached image:\n" + ref_text
        )
        default_structure = (
            "1) Type of building, 2) Shape of the building, 3) Materials, 4) Location and surroundings, "
            "5) Season, weather, time of day, lighting, 6) Camera position and angle, composition, camera parameters"
        )
        structure_text = prompt_structure.strip() if prompt_structure.strip() else default_structure
        blocks.append(
            "The structure of the prompt should be as follows (do not create headings or comments, "
            "only follow the order of information in the description):\n" + structure_text
        )
        if ignore.strip():
            blocks.append("In the prompt, ignore any mention of: " + ignore.strip())
        if emphasis.strip():
            blocks.append("In the prompt, emphasize: " + emphasis.strip())
        if isinstance(prompt_length, int) and prompt_length > 0:
            blocks.append(f"The approximate number of words in the prompt should be as close as possible to 50 {prompt_length}")

        prompt_text = "\n".join(blocks)
        log.append("Constructed prompt text successfully.")

        try:
            with open(apy_key_path, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
            log.append("Read API key from file.")
        except Exception as e:
            log.append(f"Error reading API key file: {str(e)}")
            return ("", prompt_text, "\n".join(log))

        genai.configure(api_key=api_key)

        try:
            encoded_image = base64.b64encode(image_data).decode("utf-8")
            log.append("Image successfully encoded to base64.")
        except Exception as e:
            log.append(f"Error encoding image: {str(e)}")
            return ("", prompt_text, "\n".join(log))

        fmt_upper = detected_format.upper()
        if fmt_upper == "PNG":
            mime_type = "image/png"
        elif fmt_upper in ("JPG", "JPEG"):
            mime_type = "image/jpeg"
        elif fmt_upper == "WEBP":
            mime_type = "image/webp"
        elif fmt_upper in ("HEIC", "HEIF"):
            mime_type = "image/heic"
        else:
            mime_type = "image/png"
        log.append(f"MIME type: {mime_type}")

        payload = [
            {"mime_type": mime_type, "data": encoded_image},
            prompt_text,
        ]
        log.append("Prepared payload for Gemini request.")

        model_mapping = {
            "Gemini 2.0 Flash": "gemini-2.0-flash",
            "Gemini 2.0 Flash-Lite": "gemini-2.0-flash-lite-preview",
            "Gemini 1.5 Flash": "gemini-1.5-flash",
            "Gemini 1.5 Pro": "gemini-1.5-pro",
        }
        model_name = model_mapping.get(gemini_model, "gemini-2.0-flash")

        try:
            model = genai.GenerativeModel(model_name=model_name)
            log.append(f"Sending request to Gemini model: {gemini_model}")
            response = model.generate_content(payload, request_options={"timeout": 600})
            result_text = response.text if response.text else ""
            log.append("Gemini request completed successfully.")
        except Exception as e:
            log.append(f"Error during Gemini request: {str(e)}")
            result_text = f"Error: {str(e)}"

        if save_to_path.strip():
            try:
                filename = txt_name.strip() if txt_name.strip() else "result.txt"
                if not filename.lower().endswith(".txt"):
                    filename += ".txt"
                full_path = os.path.join(save_to_path.strip(), filename)
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(result_text)
                log.append(f"Result saved to: {full_path}")
            except Exception as e:
                log.append(f"Error saving result: {str(e)}")

        return (result_text, prompt_text, "\n".join(log))


################################################################################
# Node 2: AS_MultimodalGemini
################################################################################

class AS_MultimodalGemini:
    """
    Sends text input and up to three images to Gemini API.
    Accepts models Gemini 2.0 Flash, Gemini 2.0 Flash-Lite, Gemini 1.5 Flash, Gemini 1.5 Pro.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "TEXT_INPUT": ("STRING", {"default": "", "defaultInput": True}),
                "API_KEY_PATH": ("STRING", {"default": ""}),
                "GEMINI MODEL": (
                    (
                        "Gemini 2.0 Flash",
                        "Gemini 2.0 Flash-Lite",
                        "Gemini 1.5 Flash",
                        "Gemini 1.5 Pro"
                    ),
                    {"default": "Gemini 2.0 Flash"}
                ),
            },
            "optional": {
                "IMAGE_1": ("IMAGE",),
                "IMAGE_2": ("IMAGE",),
                "IMAGE_3": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("RESULT", "LOG")
    FUNCTION = "process_inputs"
    CATEGORY = "AS_LLM nodes"

    def process_inputs(self, **kwargs):
        log = []
        text_input = kwargs.get("TEXT_INPUT", "")
        api_key_path = kwargs.get("API_KEY_PATH", "")
        gemini_model = kwargs.get("GEMINI MODEL", "Gemini 2.0 Flash")
        images = [kwargs.get("IMAGE_1"), kwargs.get("IMAGE_2"), kwargs.get("IMAGE_3")]

        encoded_images = []
        for idx, image in enumerate(images):
            if image is not None:
                try:
                    if isinstance(image, bytes):
                        image_data = image
                        img_obj = Image.open(io.BytesIO(image_data))
                    elif hasattr(image, "read"):
                        image_data = image.read()
                        img_obj = Image.open(io.BytesIO(image_data))
                    elif hasattr(image, "cpu") and hasattr(image, "detach"):
                        tensor = image.cpu().detach().numpy()
                        while tensor.ndim > 3 and tensor.shape[0] == 1:
                            tensor = tensor[0]
                        if tensor.ndim == 3 and tensor.shape[0] in [1, 3]:
                            tensor = tensor.transpose(1, 2, 0)
                        if tensor.max() <= 1:
                            tensor = (tensor * 255).astype("uint8")
                        else:
                            tensor = tensor.astype("uint8")
                        img_obj = Image.fromarray(tensor)
                        buffer = io.BytesIO()
                        img_obj.save(buffer, format="PNG")
                        image_data = buffer.getvalue()
                    elif isinstance(image, np.ndarray):
                        while image.ndim > 3 and image.shape[0] == 1:
                            image = image[0]
                        if image.ndim == 3 and image.shape[0] in [1, 3]:
                            image = image.transpose(1, 2, 0)
                        if image.max() <= 1:
                            image = (image * 255).astype("uint8")
                        else:
                            image = image.astype("uint8")
                        img_obj = Image.fromarray(image)
                        buffer = io.BytesIO()
                        img_obj.save(buffer, format="PNG")
                        image_data = buffer.getvalue()
                    else:
                        img_obj = image
                        buffer = io.BytesIO()
                        fmt = img_obj.format if img_obj.format else "PNG"
                        img_obj.save(buffer, format=fmt)
                        image_data = buffer.getvalue()

                    encoded_image = base64.b64encode(image_data).decode("utf-8")
                    encoded_images.append({"mime_type": "image/png", "data": encoded_image})
                    log.append(f"Image {idx + 1} processed.")
                except Exception as e:
                    log.append(f"Error processing image {idx + 1}: {str(e)}")

        try:
            with open(api_key_path, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
            log.append("Read API key from file.")
        except Exception as e:
            log.append(f"Error reading API key file: {str(e)}")
            return ("", "\n".join(log))

        genai.configure(api_key=api_key)

        payload = encoded_images + [text_input]
        log.append("Payload prepared for Gemini request.")

        model_mapping = {
            "Gemini 2.0 Flash": "gemini-2.0-flash",
            "Gemini 2.0 Flash-Lite": "gemini-2.0-flash-lite-preview",
            "Gemini 1.5 Flash": "gemini-1.5-flash",
            "Gemini 1.5 Pro": "gemini-1.5-pro",
        }
        model_name = model_mapping.get(gemini_model, "gemini-2.0-flash")

        try:
            model = genai.GenerativeModel(model_name=model_name)
            log.append(f"Sending request to Gemini model: {gemini_model}")
            response = model.generate_content(payload, request_options={"timeout": 600})
            result_text = response.text if response.text else ""
            log.append("Gemini request completed.")
        except Exception as e:
            log.append(f"Error during Gemini request: {str(e)}")
            result_text = f"Error: {str(e)}"

        return (result_text, "\n".join(log))


################################################################################
# Node 3: AS_ComfyGPT 
################################################################################

from openai import OpenAI

class AS_ComfyGPT:
    """
    Uses the OpenAI client to call chat completions.
    """
    HIDDEN = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key_file": ("STRING", {"default": ""}),
                "model": (("gpt-4", "gpt-4-32k", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"), {"default": "gpt-4"}),
                "prompt": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "chat_completion"
    CATEGORY = "AS_LLM nodes"

    def chat_completion(self, api_key_file, model, prompt):
        try:
            with open(api_key_file, "r") as f:
                key = f.read().strip()
        except Exception as e:
            return (f"Error reading API key file: {str(e)}",)

        print("AS_ComfyGPT: Starting API call...")
        time.sleep(2)  # Artificial delay for demonstration

        client = OpenAI(api_key=key)
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            answer = completion.choices[0].message.content
            print("AS_ComfyGPT: API call completed, returning answer.")
            return (answer,)
        except Exception as e:
            print("AS_ComfyGPT: Error encountered:", str(e))
            return (f"Error: {str(e)}",)
