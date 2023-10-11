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
from Load_Dataset import load_IR_RGB_dataset, load_IR_RGB_dataset_kfold
from Utils import show_images
import Utils
from ROC_curve_calculation import filter_train_test
from Eval_Model import evaluate_model
from Eval_Model_loss import evaluate_model_loss





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

# Define a function to update the graph
def update_graph(epoch, train_losses, test_accuracies):
    print('Update Loss accuracy graph')
    print('Train losses till now : ', train_losses)
    print('Test accuracy till now : ', test_accuracies)
    plt.figure(figsize=(10, 5))
    
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_accuracies, label='Test Accuracy')
    
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    
    plt.legend()
    
    plt.tight_layout()
    #plt.savefig(f'epoch_{epoch}.png')
    plt.show()
    plt.close()




def train(train_loader, test_loader=None, kFold = 1):
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

    N_EPOCHS = 15
    LR = 0.001

    #Plotting controller
    epoch_count=0
    epoch_show_limit = 10
    epoch_out = ''

    
    train_losses = []
    test_accuracies = []
    best_eval_score = 200.0
    stable_best_score_epoch_limit = 9
    stable_best_score_epoch = 0

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    for epoch in trange(N_EPOCHS, desc="Training"):
        model.train()
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
            show_graph(epoch_out, 1350, 768, 113)
        print('train_loss after epoch : ', train_loss)
        train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.4f}")

        #Evaluate the model till here
        model.eval()
        with torch.no_grad():
            #test_accuracy = evaluate_model_loss(train_loader, test_loader, model, device, log=False )
            #print('test_accuracy for epoch : ',epoch+1, ' - ', test_accuracy)
            #test_accuracies.append(val_loss)
            val_loss = evaluate_model_loss(train_loader, test_loader, model, device, log=False )
            print('Validation loss for epoch : ',epoch+1, ' - ', val_loss)
            train_losses.append(train_loss)
            test_accuracies.append(val_loss)
            update_graph(epoch, train_losses, test_accuracies)

            if val_loss < best_eval_score:
                #best_eval_score = test_accuracy
                best_eval_score = val_loss
                model_name = 'Fusion_Transfomer_5XK_{}_best.pth'.format(kFold)

                stable_best_score_epoch = 0

                #save the model
                print('saving model with accuracy : ',best_eval_score)
                torch.save(model, model_name)
        '''else:
            if stable_best_score_epoch < stable_best_score_epoch_limit:
                stable_best_score_epoch += 1
            else:
                if (epoch > 22):
                    print('Reached local minima in epoch : ', epoch+1, ' with accuracy : ',best_eval_score)
                    print('Stopping Training and saving the model - ',model_name)
                    ##If reaches local minima, save the model and terminate
                    print('saving model with accuracy : ',best_eval_score)
                    torch.save(model, model_name)
                    break
                else:
                    stable_best_score_epoch = 0'''

        print('End of epoch saving model with accuracy : ',best_eval_score)
        model_name_epoch = 'Fusion_Transfomer_5XK_{}_epoch.pth'.format(kFold)
        torch.save(model, model_name_epoch)
        print('Epoch Completed : ', epoch+1)

    ##Will add verification code to avoid overfitting if required
   
    update_graph(N_EPOCHS, train_losses, test_accuracies)
    #torch.save(model, "Face_Fusion_ViT_tune_model_2X.pth")


##Load data

rgb_source = 'data/full_train/RGB'
ir_source = 'data/full_train/IR'
data_source_file = 'ir_rgb_label_mapping.csv'
batch_size = 16


kfolds = 3
train_loaders, test_loaders = load_IR_RGB_dataset_kfold(rgb_source, ir_source, data_source_file, 5, batch_size, Utils.rgb_transform, Utils.ir_transform)
                              #load_IR_RGB_dataset_kfold(rgb_source, ir_source, data_source_file, k_folds=5, batch_size=32, rgb_transform=None, ir_transform=None):

print('train_loader size : ', len(train_loaders))

for i in range((kfolds)):
    filter_train_test(train_loaders[i], test_loaders[i], log=True)

    #Show sample loaded data
    for rgb_images, ir_images, labels in train_loaders[i]:
        print('rgb : ', rgb_images.shape)
        print('ir : ', ir_images.shape)
        show_images(rgb_images, labels, batch_size)
        break
    ##Initialising Model Training
    print('**********Initialising Model Training for K - ', i)
    train(train_loaders[i], test_loaders[i], i)
