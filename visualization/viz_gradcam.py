import torch
import sys
sys.path.append('../')
import cv2
import os.path
from torchvision import transforms
from PIL import Image
from utils import augmentation
import matplotlib.pyplot as plt
import numpy as np


def main(trainer):

    print('\n==== Visualizing gradcam ====\n')
    trainer.model.eval()

    input_seq, labels, index = iter(trainer.loaders['test']).next()
    input_seq = input_seq.to(trainer.args.device)

    trainer.get_base_model().backbone.layer4[1].conv2.register_forward_hook(hook)

    # with torch.no_grad():
    input_seq.requires_grad = True
    output_model = trainer.model(input_seq)
    pred, feature_dist, sizes_pred = output_model

    activation_conv2 = trainer.get_base_model().backbone.layer4[1].conv2._value_hook

    loss_uncertainty = pred.pow(2).sum(-1).mean()*1000
    # gradient = torch.autograd.grad(outputs=loss_uncertainty, inputs=input_seq, retain_graph=True)[0]
    # gradient = gradient.permute(0, 1, 3, 2, 4, 5).reshape(gradient.shape[0], -1, 3, gradient.shape[-1], gradient.shape[-1]).cpu()

    gradient = torch.autograd.grad(outputs=loss_uncertainty, inputs=activation_conv2, retain_graph=True)[0]
    alpha_k = gradient.mean(dim=[-1, -2, -3])
    l_gradcam = torch.relu((alpha_k[:, :, None, None, None] * activation_conv2).sum(1))
    l_gradcam = l_gradcam.view(input_seq.shape[0], input_seq.shape[1], 2, 4, 4).detach().cpu()

    transform = transforms.Compose([
        augmentation.CenterCrop(size=input_seq.shape[-1], consistent=True),
        augmentation.ToTensor(),
    ])

    for i, idx in enumerate(index):
        idx_block, vpath = trainer.loaders['test'].dataset.get_info(idx.cpu().numpy())

        os.makedirs(f'/proj/vondrick/didac/results/heatmaps/{idx}', exist_ok=True)
        # video = cv2.VideoWriter(f'/proj/vondrick/didac/results/video_heatmap_{idx}.avi', 0, 1, (gradient.shape[-1], gradient.shape[-1]))

        # grad = gradient[i].sum(1)  # sum three channels
        grad = l_gradcam[i]

        grad = grad.permute(2, 3, 0, 1).reshape(4, 4, 16)
        grad = grad.numpy()

        grad = cv2.resize(grad, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)

        grad = grad.transpose(2, 0, 1)

        idx_block = idx_block[:, [0, -1]]
        for j, image_id in enumerate(idx_block.reshape(-1)):
            grad_j = grad[j]

            grad_j = grad_j - grad_j.min()
            grad_j = grad_j / grad_j.max()

            path_img = os.path.join(vpath, f'image_{image_id+1:05d}.jpg')
            img = Image.open(path_img)
            img = transform([img])[0].permute(1, 2, 0).numpy()

            cmap = plt.get_cmap('jet')
            rgba_img = cmap(grad_j)
            rgb_img = np.delete(rgba_img, 3, 2)

            alpha = np.maximum(grad_j - 0.3, 0)[:, :, None]/0.7
            final_img = rgb_img * alpha + img * (1 - alpha)

            # video.write(np.uint8(255*final_img))

            fig, ax = plt.subplots(1, 1)
            ax.imshow(final_img)
            plt.savefig(f'/proj/vondrick/didac/results/heatmaps/{idx}/{image_id}.jpg')

        print('hey')

        # cv2.destroyAllWindows()
        # video.release()


def hook(module, input, output):
    setattr(module, "_value_hook", output)
