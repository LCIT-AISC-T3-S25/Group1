import torch
import numpy as np
import cv2

def get_gradcam_image(model, image_tensor, target_layer="features.29"):
    model.eval()
    gradients = []
    activations = []

    def save_gradients_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def save_activations_hook(module, input, output):
        activations.append(output)

    layer = dict([*model.named_modules()])[target_layer]
    handle_activations = layer.register_forward_hook(save_activations_hook)
    handle_gradients = layer.register_backward_hook(save_gradients_hook)

    output = model(image_tensor)
    class_idx = torch.argmax(output)
    score = output[0, class_idx]
    model.zero_grad()
    score.backward()

    grads = gradients[0].squeeze(0)
    acts = activations[0].squeeze(0)

    weights = grads.mean(dim=(1, 2))
    cam = torch.zeros(acts.shape[1:], dtype=torch.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam.detach().numpy(), 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    cam /= cam.max()
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img = image_tensor.squeeze().permute(1, 2, 0).detach().numpy()
    img = np.uint8((img - img.min()) / (img.max() - img.min()) * 255)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    handle_activations.remove()
    handle_gradients.remove()

    return overlay