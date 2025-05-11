
![WrD.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/6TixjrntJJtmfFIFeoIef.png)

# **Realistic-Gender-Classification**

> **Realistic-Gender-Classification** is a binary image classification model based on `google/siglip2-base-patch16-224`, designed to classify **gender** from realistic human portrait images. It can be used in **demographic analysis**, **personalization systems**, and **automated tagging** in large-scale image datasets.

> [!note]
*SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features* https://arxiv.org/pdf/2502.14786

```py
Classification Report:
                 precision    recall  f1-score   support

female portrait     0.9754    0.9656    0.9705      1600
  male portrait     0.9660    0.9756    0.9708      1600

       accuracy                         0.9706      3200
      macro avg     0.9707    0.9706    0.9706      3200
   weighted avg     0.9707    0.9706    0.9706      3200
```

![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/Hl1qDGrIIZyiSOzOX8K8t.png)

---

## **Label Classes**

The model distinguishes between the following portrait gender categories:

```
0: female portrait  
1: male portrait
```

---

## **Installation**

```bash
pip install transformers torch pillow gradio
```

---

## **Example Inference Code**

```python
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
```

---

## Demo Inference

> [!note]
female portrait

![Screenshot 2025-05-10 at 17-09-35 Realistic-Gender-Classification.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/EC02LG1gUHsEkLCxCtBBH.png)
![Screenshot 2025-05-10 at 17-10-09 Realistic-Gender-Classification.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/ttk_eJYsSLTZIaao7u10f.png)

> [!note]
male portrait

![Screenshot 2025-05-10 at 17-10-48 Realistic-Gender-Classification.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/qCeP_BcpV5gWHtozZkhpE.png)
![Screenshot 2025-05-10 at 17-11-39 Realistic-Gender-Classification.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/mVT0ogVrQOckIHET6Vq4H.png)

## **Applications**

* **Demographic Insights in Visual Data**
* **Dataset Curation & Tagging**
* **Media Analytics**
* **Audience Profiling for Marketing** 
