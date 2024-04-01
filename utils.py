import os
import torch
from dataset import *
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torch import nn
from torchvision.utils import make_grid


def plot_cco(dataset, label=None):
    index = [i for i, l in enumerate(dataset.labels) if l == label][0]
    image = Image.open(dataset.images[index])
    return image

def cco_indices(dataset, indices):
    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()
    index_to_label = {v: k for k, v in dataset.label_to_index.items()}
    labels = [index_to_label[i] for i in indices]
    images = [plot_cco(dataset, label=l) for l in labels]
    return images

def plot_cco_indices(dataset, indices, shape=None):
    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()
    index_to_label = {v: k for k, v in dataset.label_to_index.items()}
    labels = [index_to_label[i] for i in indices]
    images = [plot_cco(dataset, label=l) for l in labels]

    N = len(images)
    if shape is not None:
        nrows, ncols = shape
    elif N > 4:
        ncols = 4
        nrows = math.ceil(len(images) / ncols)
    else:
        ncols = N
        nrows = 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    axs = axs.flatten()
    for i, img in enumerate(images):
        axs[i].axis("off")
        axs[i].set_title(labels[i])
        axs[i].imshow(img)

    plt.show()

def plot_cco_indices_grid(dataset, indices, show_labels=True, shape=(2,3), size=(224, 224)):
    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()

    index_to_label = {v: k for k, v in dataset.label_to_index.items()}
    labels = [index_to_label[i] for i in indices]

    # Transform to resize images and convert them to PyTorch tensors
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    images = [transform(plot_cco(dataset, label=l)) for l in labels]

    # Convert list of images to a single tensor
    image_grid = torch.stack(images)

    # Use make_grid to create a grid layout
    grid = make_grid(image_grid, nrow=shape[1])

    # Plot the grid
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    
    # Optionally display labels
    if show_labels:
        title = ' | '.join(labels)
        plt.title(title)

    plt.show()

def plot_indices_grid(images, shape=(2,3), size=(224, 224)):

    # Transform to resize images and convert them to PyTorch tensors
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    images = [transform(image) for image in images]

    # Convert list of images to a single tensor
    image_grid = torch.stack(images)

    # Use make_grid to create a grid layout
    grid = make_grid(image_grid, nrow=shape[1])

    # Plot the grid
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')

    plt.show()

@torch.no_grad()
def ce_extracting(imgs, vlmodel, proj, preprocess, device='cuda'):
    # imgs: list of images
    clip_dtype = torch.float16
    ce_dtype = torch.float
    print(f'clip_dtype is {clip_dtype} and ce_dtype is {ce_dtype}')
    image_features = torch.tensor([])
    concept_embeddings = torch.tensor([])
    for image in imgs:
        image = preprocess(image).to(dtype=clip_dtype, device=device).unsqueeze(0)
        image_feature = vlmodel.encode_image(image)
        image_features = torch.cat((image_features, image_feature.cpu()), dim=0)
        concept_embedding = proj(image_feature.to(dtype=ce_dtype, device=device))
        concept_embeddings = torch.cat((concept_embeddings, concept_embedding.cpu()), dim=0)
    return image_features, concept_embeddings

def plot_bar(names, values, num_display=7, figsize=(1, 2)):
    # names: list of names
    # values: tensor of values
    # num_display: number of names to display
    # select top num_display values
    values, indices = torch.topk(values, num_display)
    names = [names[i] for i in indices]
    # plot bar using seaborn
    plt.figure(figsize=figsize)

    # exchange x and y, color changed with values
    sns.barplot(
        x=values.cpu().numpy(), y=names, hue=names,
        palette="Blues_d", legend=False,
    )
    # remove x axis and ticks
    plt.xlabel('')
    # remove ticks but keep labels in x axis
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
    
    # remove y axis
    plt.ylabel('')
    # set yticks by names
    plt.yticks(range(len(names)), names)
    # remove ticks but keep labels in y axis
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=True)

    # remove border
    sns.despine(left=True, bottom=True)

    quantile_90 = 0.3632
    plt.axvline(x=quantile_90, ymax=1, color='black', linestyle='--', linewidth=1)
    plt.show()