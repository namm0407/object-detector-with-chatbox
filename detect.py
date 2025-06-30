# test to see if it can correctly identify the picture
# the accuracy is not quite high bacause of the model. But the code works

import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
import requests
from io import BytesIO

# Set device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BLIP-2 model (OPT-2.7B for captioning + VQA)
model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip2_opt",
    model_type="pretrain_opt2.7b",
    is_eval=True,
    device=device
)

# Load an image (replace with your image path or URL)
image_path = "dogs.jpg"  # Local file
# image_path = "https://example.com/image.jpg"  # Or fetch from URL

# Load and preprocess the image
if image_path.startswith("http"):
    response = requests.get(image_path)
    image = Image.open(BytesIO(response.content)).convert("RGB")
else:
    image = Image.open(image_path).convert("RGB")

image_processed = vis_processors["eval"](image).unsqueeze(0).to(device)

# --- 1. Generate a Caption ---
caption = model.generate({"image": image_processed})
print("\n[CAPTION]:", caption[0])

# --- 2. Ask Questions About the Image ---
questions = [
    "What is the main object in this image?",
    "What colors are dominant?",
    "Is there any text visible?",
    "How would you describe the scene in one sentence?"
]

print("\n[QUESTION ANSWERING]:")
for question in questions:
    # Preprocess the question
    question_processed = txt_processors["eval"](question)
    
    # Generate answer
    answer = model.generate({
        "image": image_processed,
        "prompt": f"Question: {question} Answer:"
    })
    print(f"Q: {question}\nA: {answer[0]}\n")
