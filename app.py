import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Realistic-Gender-Classification"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# ID to label mapping
id2label = {
    "0": "female portrait",
    "1": "male portrait"
}

def classify_gender(image):
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    prediction = {id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    return prediction

# Gradio Interface
iface = gr.Interface(
    fn=classify_gender,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=2, label="Gender Classification"),
    title="Realistic-Gender-Classification",
    description="Upload a realistic portrait image to classify it as 'female portrait' or 'male portrait'."
)

if __name__ == "__main__":
    iface.launch()
