##Training model##
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torchvision
import torch


from tqdm import tqdm, trange
#For graphs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE  # For dimensionality reduction (optional)

#Import Transformer
from FusionVisionFaceTransfomer import FusionVisionFaceTransfomer
#from  FusionVisionFaceTransfomer import FusionVisionFaceTransfomer
from Load_Dataset import load_IR_RGB_dataset
from Load_Dataset import pair_show_images





def train_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    #embed()
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res[0]


##Graph method
def show_graph(arcface_embeddings, num_samples, embedding_dim, num_classes):
    arcface_embeddings = np.random.rand(num_samples, embedding_dim)
    arcface_embeddings /= np.linalg.norm(arcface_embeddings, axis=1, keepdims=True)  # Normalize embeddings
    class_labels = np.random.randint(0, num_classes, num_samples)  # Assign random class labels

    # Optionally, perform dimensionality reduction for visualization (e.g., using t-SNE)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    reduced_embeddings = tsne.fit_transform(arcface_embeddings)

    # Get unique class labels
    unique_labels = np.unique(class_labels)

    # Create a color map for classes
    num_unique_labels = len(unique_labels)
    color_map = plt.get_cmap('tab20', num_unique_labels)

    # Plot the scatter plot
    plt.figure(figsize=(8, 8))
    for i, label in enumerate(unique_labels):
        indices = np.where(class_labels == label)
        #plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], color=color_map(i), label=f'Class {label}')
        plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], color=color_map(i))

    plt.title("ArcFace Embeddings Scatter Plot")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.show()



def train(train_loader, test_loader=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    #device = "cpu"
    custom_config = {"img_size" :  128,
                 "in_channels" :  3,
                 "patch_size" : 16,
                 "embed_dim" : 768,
                 "depth" : 12,
                 "n_classes" : 114,
                 "n_heads" : 12,
                 "qkv_bias" :  True,
                 "mlp_ratio" :  4
                 }
    print('Setting us custom config...', custom_config)

    model = FusionVisionFaceTransfomer(**custom_config).to(device)

    m = 0.4
    s = 64.0
    n_classes = 114

    N_EPOCHS = 200
    LR = 0.01

    #Plotting controller
    epoch_count=0
    epoch_show_limit = 10
    epoch_out = ''

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        epoch_count += 1

        for rgb_images, ir_images, labels in train_loader:

            rgb_images, ir_images, labels = rgb_images.to(device), ir_images.to(device), labels.to(device)
            
            outputs, embedding = model(rgb_images, ir_images, labels)
            loss = criterion(outputs, labels)

            prec1= train_accuracy(outputs.data, labels, topk = (1,))

            epoch_out = embedding
            
            train_loss += loss.detach().cpu().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch_count == epoch_show_limit):
            epoch_count = 0
            show_graph(epoch_out, 32, 768, 113)
        print('train_loss after epoch : ', train_loss)
        train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.4f}")
        print('Epoch Completed : ', epoch+1)

    ##Will add verification code to avoid overfitting if required

    torch.save(model, "Face_Fusion_ViT_tune_model.pth")


##Load data
print('Initialising Data loader...')
train_loader, test_loader = load_IR_RGB_dataset()
print('Data loaded... train size :', len(train_loader), 'test_loader : ', len(test_loader))

#Show sample loaded data
for rgb_images, ir_images, labels in train_loader:
    print('rgb : ', rgb_images.shape)
    print('ir : ', ir_images.shape)
    pair_show_images(rgb_images, ir_images, labels)
    break

##Initialising Model Training
print('**********Initialising Model Training...')
train(train_loader, test_loader)

