
import torch

#Importing project Modules
import Test_Model
import Utils
from prepare_ds import prepare_ds_source
from Load_Dataset import load_IR_RGB_dataset_kfold
from Utils import unique_elements_from_lists
#from Test_Model import evaluate_test_data
from Test_Model import evaluate_TP_on_test_data, evaluate_TN_on_test_data
from ROC_curve_calculation import filter_train_test
from ROC_curve_calculation import find_best_threshold





rgb_source = 'data/full_train/RGB'
ir_source = 'data/full_train/IR'
data_source_file = 'ir_rgb_label_mapping.csv'

batch_size = 32
k_folds=5
#threshold - Needs to be calculated by ROC - pending - next in line

#Generate Data_Source_File - Run it only one time to initialise
#prepare_ds_source(data_source_file, rgb_source, ir_source)

train_loaders, test_loaders = load_IR_RGB_dataset_kfold(rgb_source, ir_source, data_source_file, 5, batch_size, Utils.rgb_transform, Utils.ir_transform)
                              #load_IR_RGB_dataset_kfold(rgb_source, ir_source, data_source_file, k_folds=5, batch_size=32, rgb_transform=None, ir_transform=None):
print('train_loader size : ', len(train_loaders))

for i in range((k_folds)):
    # Load model 
    #Load the model and set is to EVAL mode
    i =2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    #model = torch.load('Face_Fusion_ViT_tune_model.pth').to(device)
    #model = torch.load('MSXXFusion_Transfomer_1.pth').to(device)#
    model_name = 'Fusion_Transfomer_ADAMW_{}_best.pth'.format(i)
    model = torch.load(model_name).to(device)
    model.eval()

    test_labels, train_labels =  filter_train_test(train_loaders[i], test_loaders[i])

    threshold = find_best_threshold(test_labels, model, device, batch_size)



    #Test The model for test labels
    result_TP = evaluate_TP_on_test_data(test_labels, model, threshold, device=device, batch_size=2, log = True)
    print('TP - TRUE POSITIVE FOR TEST DATASET : ', len(test_labels), ' - ', result_TP)

    result_TN = evaluate_TN_on_test_data(test_labels, model, threshold, device=device, batch_size=2, log=True)
    print('TN - TRUE NEGATIVE FOR TEST DATASET : ', len(test_labels), ' - ', result_TN)

    print('\n\n***************************\n')
    print('Trained on ',len(train_labels),' People')
    print(train_labels)
    print('Evaluated on ',len(test_labels),' different People (not in train set)')
    print(test_labels)

    TP = result_TP
    TN = result_TN 
    FP = 100 - TP
    FN = 100 - TN

    print('TP - ', result_TP)
    print('TN - ', result_TN)
    print('FP - ', FP)
    print('FN - ', FN)

    accuracy = (TP + TN)/(TP+TN+FN+FP)
    print('Accuracy : ', accuracy)

    if (TP+FP) > 0 and (TP+FN):
        precision = TP/(TP+FP)
        print('Precision : ',precision)

        recall = TP/(TP+FN)
        print('Recall  : ', recall)


        F1 = (2*precision*recall)/(precision+recall)
        print('F1 score : ', F1 )
    break


'''
train_loader, test_loader = load_IR_RGB_dataset(rgb_source=rgb_source, ir_source=ir_source, 
                                                data_source_file=data_source_file, batch_size=batch_size, rgb_transform=Utils.rgb_transform, ir_transform=Utils.ir_transform)


test_labels, train_labels =  filter_train_test(train_loader, test_loader)

'''


'''
# Load model 
#Load the model and set is to EVAL mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

#model = torch.load('Face_Fusion_ViT_tune_model.pth').to(device)
#model = torch.load('MSXXFusion_Transfomer_1.pth').to(device)
model = torch.load('MKXFolds_Fusion_Transfomer_1.pth').to(device)
model.eval()

threshold = find_best_threshold(test_labels, model, device, batch_size)



#Test The model for test labels
result_TP = evaluate_TP_on_test_data(test_labels, model, threshold, device=device, batch_size=2, log = True)
print('TP - TRUE POSITIVE FOR TEST DATASET : ', len(test_labels), ' - ', result_TP)

result_TN = evaluate_TN_on_test_data(test_labels, model, threshold, device=device, batch_size=2, log=True)
print('TN - TRUE NEGATIVE FOR TEST DATASET : ', len(test_labels), ' - ', result_TN)

print('\n\n***************************\n')
print('Trained on ',len(train_labels),' People')
print(train_labels)
print('Evaluated on ',len(test_labels),' different People (not in train set)')
print(test_labels)

TP = result_TP
TN = result_TN 
FP = 100 - TP
FN = 100 - TN

print('TP - ', result_TP)
print('TN - ', result_TN)
print('FP - ', FP)
print('FN - ', FN)

accuracy = (TP + TN)/(TP+TN+FN+FP)
print('Accuracy : ', accuracy)

precision = TP/(TP+FP)
print('Precision : ',precision)

recall = TP/(TP+FN)
print('Recall  : ', recall)


F1 = (2*precision*recall)/(precision+recall)
print('F1 score : ', F1 )
'''