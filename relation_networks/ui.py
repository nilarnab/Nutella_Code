import torch
import gradio as gr
from torchvision import transforms
from main import EmbeddingModule, RelationModule, compute_relation_scores, Config

DEVICE = Config.DEVICE

def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    dataset = ckpt.get("config", {}).get("DATASET", "omniglot")
    in_ch, input_size = (1, 64) if dataset == "omniglot" else (3, 576)
    embed  = EmbeddingModule(in_ch).to(DEVICE)
    relate = RelationModule(input_size).to(DEVICE)
    embed.load_state_dict(ckpt["embed"])
    relate.load_state_dict(ckpt["relate"])
    embed.eval(); relate.eval()
    return embed, relate, dataset

def get_tf(dataset):
    if dataset == "omniglot":
        return transforms.Compose([transforms.Grayscale(), transforms.Resize((28, 28)), transforms.ToTensor()])
    return transforms.Compose([transforms.Resize(84), transforms.CenterCrop(84), transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def classify(ckpt_path, s1, s2, s3, s4, s5, n1, n2, n3, n4, n5, query):
    support_imgs = [s1, s2, s3, s4, s5]
    names = [n or f"Class {i+1}" for i, n in enumerate([n1, n2, n3, n4, n5])]

    if any(img is None for img in support_imgs) or query is None:
        return "Please upload all 5 support images and a query image."

    print("checkpiont path", ckpt_path)
    embed, relate, dataset = load_model(ckpt_path)
    tf = get_tf(dataset)

    to_t = lambda img: tf(img.convert("RGB"))
    support  = torch.stack([to_t(img) for img in support_imgs]).to(DEVICE)
    s_labels = torch.arange(5, dtype=torch.long).to(DEVICE)
    query_t  = to_t(query).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        scores = compute_relation_scores(embed, relate, support, s_labels, query_t, 5, 1).squeeze(0)

    pred = scores.argmax().item()
    result = "\n".join(f"{'→' if i==pred else '  '} {names[i]}: {scores[i]:.4f}" for i in range(5))
    return f"Predicted: {names[pred]}\n\n{result}"

with gr.Blocks(title="Relation Network Demo") as demo:
    gr.Markdown("## Relation Network — 5-way 1-shot")
    ckpt = gr.Textbox(value="relation_net_best_working.pth", label="Checkpoint path")

    gr.Markdown("**Support set** — one image per class")
    with gr.Row():
        supports = [gr.Image(label=f"Class {i+1}", type="pil", height=120) for i in range(5)]
    with gr.Row():
        names = [gr.Textbox(placeholder=f"Class {i+1} name", show_label=False) for i in range(5)]

    gr.Markdown("**Query image**")
    with gr.Row():
        query = gr.Image(label="Query", type="pil", height=150)
        output = gr.Textbox(label="Result", lines=8)

    gr.Button("Classify").click(classify, inputs=[ckpt] + supports + names + [query], outputs=output)

demo.launch(inbrowser=True)