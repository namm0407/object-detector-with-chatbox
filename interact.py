# the user can upload an image and ask questions about the images

# For example
'''
[CAPTION]: A golden retriever sitting in a grassy field.

Ask questions about the image (type 'quit' to exit):
You: What color is the dog?
BLIP-2: Golden.

You: Is it outdoors?
BLIP-2: Yes, in a field.

You: quit
'''

import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
import requests
from io import BytesIO

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip2_opt",
    model_type="pretrain_opt2.7b",
    is_eval=True,
    device=device
)

# Load image
'''
#get image from user
image_path = input("Enter the path to your image (e.g., 'dogs.jpg' or 'https://example.com/image.jpg'): ")
'''
image_path = "dogs.jpg"  

# Load and preprocess the image
if image_path.startswith("http"):
    response = requests.get(image_path)
    image = Image.open(BytesIO(response.content)).convert("RGB")
else:
    image = Image.open(image_path).convert("RGB")
image_processed = vis_processors["eval"](image).unsqueeze(0).to(device)

# Generate initial caption (optional)
caption = model.generate({"image": image_processed})
print("\n[CAPTION]:", caption[0])

# Interactive Q&A loop
print("\nAsk questions about the image (type 'quit' to exit):")
while True:
    question = input("\nYou: ")
    if question.lower() in ["quit", "exit", "q"]:
        break
    
    # Generate answer
    answer = model.generate({
        "image": image_processed,
        "prompt": f"Question: {question} Answer:"
    })
    print(f"BLIP-2: {answer[0]}")
