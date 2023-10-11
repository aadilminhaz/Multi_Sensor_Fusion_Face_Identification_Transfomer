import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch
from torch import tensor



import torch

#Importing project Modules

import Utils
from prepare_ds import prepare_ds_source
from Load_Dataset import load_IR_RGB_dataset
from Utils import unique_elements_from_lists
#from Test_Model import evaluate_test_data
from Test_Model import evaluate_TN_on_test_data
from Calculate_Distanace import calculate_POS_distance, calculate_NEG_distance


def calculate_ROC(ground_truth, predicted_distance):

    # Assuming you have ground truth labels and predicted distances/similarities
    #ground_truth = np.array([0, 0, 1, 1])  # 0 for non-match, 1 for match
    #predicted_distance = np.array([0.1, 0.3, 0.8, 0.6])  # Hypothetical distances

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(ground_truth, predicted_distance)  # Negative distances because ROC is based on ranks
    #roc_auc = auc(fpr, tpr)
    roc_auc = auc(fpr, tpr)

    print('Thresholds : ', thresholds)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    mid_index = len(thresholds)//2
    #mid_index = 2
    print('Mid : ', mid_index)
    return thresholds[mid_index]
    
    









#rgb_source = 'data/full_train/RGB'
#ir_source = 'data/full_train/IR'
#data_source_file = 'ir_rgb_label_mapping.csv'

#batch_size = 32

#Needs to be calculated by ROC - pending - next in line

#Generate Data_Source_File - Run it only one time to initialise
#prepare_ds_source(data_source_file, rgb_source, ir_source)


#train_loader, test_loader = load_IR_RGB_dataset(rgb_source=rgb_source, ir_source=ir_source, 
#                                                data_source_file=data_source_file, batch_size=batch_size, rgb_transform=Utils.rgb_transform, ir_transform=Utils.ir_transform)


def filter_train_test(train_loader, test_loader, log=False):
    train_labels_list = []
    test_labels_list = []
    for rgb_images, ir_images, labels in train_loader:
        train_labels_list.append(labels)

    for rgb_images, ir_images, labels in test_loader:
        test_labels_list.append(labels)

    train_labels = torch.unique(torch.cat((train_labels_list)))
    test_labels = torch.unique(torch.cat((test_labels_list)))
    if log == True:
        print('Train Labels-------- : ', train_labels)
        print('Test Labels---------- : ', test_labels)


    set1 = set(train_labels.tolist())
    set2 = set(test_labels.tolist())

    # Find common elements
    common_elements = torch.tensor(list(set1.intersection(set2)))
    if log == True:
        print('Overlapping Data :', common_elements)

        print('Remove overllaping element from train labels and remove bad data')
    ##Remove the overlapping element 
    filtered_test_labels = test_labels[~torch.isin(test_labels, common_elements)]
    ##Filter out bad data in out case - 96
    bad_labels = torch.tensor([96])
    filtered_test_labels = filtered_test_labels[~torch.isin(filtered_test_labels, bad_labels)]
    if log == True:
        print('Filtered test labels :', filtered_test_labels)
    return filtered_test_labels, train_labels






# Load model 
#Load the model and set is to EVAL mode
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

#model = torch.load('Face_Fusion_ViT_tune_model.pth').to(device)
#model.eval()


def find_best_threshold(filtered_train_labels, model, device, batch_size=2):
    result_POS_distance = calculate_POS_distance(filtered_train_labels, model, device=device, batch_size=2)
    result_NEG_distance = calculate_NEG_distance(filtered_train_labels, model, device=device, batch_size=2)

    #print('Result pos distances : ', result_POS_distance)
    ##print('Result neg distances : ', result_NEG_distance)


    ground_truth_POS = torch.ones(len(filtered_train_labels))
    ground_truth_NEG = torch.zeros(len(filtered_train_labels))
    ground_truth = torch.cat((ground_truth_POS, ground_truth_NEG))


    #print('POS ground truth : ', ground_truth_POS)
    #print('NEG ground truth : ', ground_truth_NEG)
    print('Ground Truth :', ground_truth)

    pred_distance = torch.cat((torch.tensor(result_POS_distance), torch.tensor(result_NEG_distance)))
    print('Pred  :', pred_distance)

    best_thresold = calculate_ROC(ground_truth, pred_distance)
    print('Best Thresold : ', best_thresold)
    return best_thresold




