import torch
import torch.nn as nn
from PIL import Image
import numpy as np

import Utils
from Utils import pair_show_images






##Method to calculate COSINE-Similarity
def is_similar(input_embedding, original_embedding, threshold, decimal_points = 5):
        
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        similarity_result = cos(input_embedding, original_embedding)
        threshold = torch.tensor(threshold)

        similarity_result = similarity_result+ 10**(-decimal_points)
        threshold = threshold+ 10**(-decimal_points)
        #print('Best Selected Thresold  : ', threshold)
        
        print('Similarity : ', similarity_result)
        if similarity_result >= threshold:
                return True
        else:
                return False

#Method to authenticate and Input-Image(combo or RGB-IR) with Original Image        
def test_identity(input_rgb_image, input_ir_image, original_rgb_image, original_ir_image, model, threshold = 0.8, device = 'cpu', input_label=None, original_label=None, batch_size=32, log=False):
    #with torch.no_grad():
    correct, total = 0, 0
    test_loss = 0.0
    x_rgb, x_ir, xo_rgb, xo_ir = input_rgb_image[None,:,:,:], input_ir_image[None,:,:,:], original_rgb_image[None,:,:,:], original_ir_image[None,:,:,:]
    x_rgb, x_ir, xo_rgb, xo_ir = x_rgb.to(device), x_ir.to(device), xo_rgb.to(device), xo_ir.to(device)

    if log == True:
        print('shapes : x_rgb : ',x_rgb.shape, ' x_ir :', x_ir.shape, ' xo_rgb : ', xo_rgb.shape, ' xo_ir :', xo_ir.shape)

    input_embedding = model(input_image_rgb=x_rgb, input_image_ir=x_ir)
    original_embedding = model(input_image_rgb=xo_rgb, input_image_ir=xo_ir)

    
    if log == True:
        print('Face Embedding after transformer : input : ', input_embedding.shape)
    #To print images, place it in one array
    input_images = []
    input_images.append(input_rgb_image)
    input_images.append(input_ir_image)
    if  log == True:
        pair_show_images(input_images, input_label, batch_size)
    
    if log == True:
        print('Face Embedding after transformer : original : ', original_embedding.shape)
    original_images = []
    original_images.append(original_rgb_image)
    original_images.append(original_ir_image)
    if log == True:
        pair_show_images(original_images, original_label, batch_size)

    authenticated = is_similar(input_embedding, original_embedding, threshold)
    if log == True:
        print('Authenticated : ', authenticated)
    return authenticated



def evaluate_TP_on_test_data(test_labels, model, threshold=0.8, device='cpu', batch_size=32, log=False):
     
    total = 0.0
    TRUE_POSITIVE = 0.0
    for test_label in test_labels:
        test_label = test_label.item()
        if log == True:
            print('**********************\nEvaluating for person : ',test_label)
        total += 1.0
        input_rgb_image_path = 'data/full_train/RGB/'+str(test_label)+'-TD-A-4.jpg'
        input_ir_image_path = 'data/full_train/IR/'+str(test_label)+'-TD-A-4.jpg'
            
        original_image_rgb_path = 'data/full_train/RGB/'+str(test_label)+'-TD-E-1.jpg'
        original_image_ir_path = 'data/full_train/IR/'+str(test_label)+'-TD-E-1.jpg'
                

        input_rgb_image = Image.open(input_rgb_image_path)
        input_rgb_image = Utils.rgb_transform(input_rgb_image)
        input_ir_image = Image.open(input_ir_image_path)
        input_ir_image = Utils.ir_transform(input_ir_image)


        original_rgb_image = Image.open(original_image_rgb_path)
        original_rgb_image = Utils.rgb_transform(original_rgb_image)
        original_ir_image = Image.open(original_image_ir_path)
        original_ir_image = Utils.ir_transform(original_ir_image)

            
        result = test_identity(input_rgb_image, input_ir_image, original_rgb_image, original_ir_image, model, threshold, device, test_label, test_label, batch_size, log )
        
        if result is True:
            TRUE_POSITIVE +=1
    
    if total < 1:
        return ''
    accuracy = TRUE_POSITIVE/total *100
    #result = 'Face Authentication accuracy for person : '+str(target_label)+' ||||  '+ str(accuracy) 
    #result = 'Face Authentication accuracy for test set |||| :::  '+ str(accuracy) 
    #results.append(result)
    return accuracy



def evaluate_TN_on_test_data(test_labels, model, threshold=0.8, device='cpu', batch_size=32, log=False):
     
    total = 0.0
    TRUE_NEGATIVE = 0.0
    #last_label_index = len(test_labels)
    for i in range(len(test_labels)):
        last = i+1
        #print('fist label : ', test_labels[i], ' last label :', test_labels[-last])
        
             
        fist_label = test_labels[i].item()
        second_label = test_labels[-last].item()
        if log == True:
            print('**********************\nEvaluating for person : ',fist_label, ' with person : ', second_label)
        total += 1.0
        
        input_rgb_image_path = 'data/full_train/RGB/'+str(fist_label)+'-TD-A-4.jpg'
        input_ir_image_path = 'data/full_train/IR/'+str(fist_label)+'-TD-A-4.jpg'
            
        original_image_rgb_path = 'data/full_train/RGB/'+str(second_label)+'-TD-E-1.jpg'
        original_image_ir_path = 'data/full_train/IR/'+str(second_label)+'-TD-E-1.jpg'
                

        input_rgb_image = Image.open(input_rgb_image_path)
        input_rgb_image = Utils.rgb_transform(input_rgb_image)
        input_ir_image = Image.open(input_ir_image_path)
        input_ir_image = Utils.ir_transform(input_ir_image)


        original_rgb_image = Image.open(original_image_rgb_path)
        original_rgb_image = Utils.rgb_transform(original_rgb_image)
        original_ir_image = Image.open(original_image_ir_path)
        original_ir_image = Utils.ir_transform(original_ir_image)

                

        result = test_identity(input_rgb_image, input_ir_image, original_rgb_image, original_ir_image, model, threshold, device, fist_label, second_label, batch_size, log )
        
        if result is False:
            TRUE_NEGATIVE +=1
       
    if total < 1:
        return ''
    accuracy = TRUE_NEGATIVE/total *100
    #result = 'Face Authentication accuracy for person : '+str(target_label)+' ||||  '+ str(accuracy) 
    #result = 'Face Authentication accuracy for test set |||| :::  '+ str(accuracy) 
    #results.append(result)
    return accuracy
'''   
def evaluate_test_data(target_label, test_loader, model, threshold=0.8, device='cpu', batch_size=32):
    print('**********************\nEvaluating for person : ',target_label )
     ##Test for all the test images
    target_images = []
    #target_label = t
    target_labels = []
    
    
    
    total = 0.0
    correct = 0.0
    for rgb_images, ir_images, labels in test_loader:
        target_index = np.where(labels == target_label)[0]
        if (len(target_index) != 0):
            for i in target_index:
                total +=1
                print('\n###########################\n')
                #print('Testing for image : ',i)
                #input_rgb_image = rgb_images[i]
                #input_ir_image = ir_images[i]

                #original_trained_image_rgb = 'data/IR_RGB_ds/RGB-faces-128x128/'+str(target_label)+'-TD-E-1.jpg'
                #original_trained_image_ir = 'data/IR_RGB_ds/thermal-face-128x128/'+str(target_label)+'-TD-E-1.jpg'
                input_rgb_image_path = 'data/full_train/RGB/'+str(i)+'-TD-A-4.jpg'
                input_ir_image_path = 'data/full_train/IR/'+str(i)+'-TD-A-4.jpg'
            
                original_image_rgb_path = 'data/full_train/RGB/'+str(i)+'-TD-E-1.jpg'
                original_image_ir_path = 'data/full_train/IR/'+str(i)+'-TD-E-1.jpg'
                

                input_rgb_image = Image.open(input_rgb_image_path)
                input_rgb_image = Utils.rgb_transform(input_rgb_image)
                input_ir_image = Image.open(input_ir_image_path)
                input_ir_image = Utils.ir_transform(input_ir_image)


                original_rgb_image = Image.open(original_image_rgb_path)
                original_rgb_image = Utils.rgb_transform(original_rgb_image)
                original_ir_image = Image.open(original_image_ir_path)
                original_ir_image = Utils.ir_transform(original_ir_image)

                

                result = test_identity(input_rgb_image, input_ir_image, original_rgb_image, original_ir_image, model, threshold, device, target_index, target_index, batch_size )
                if result is True:
                    correct +=1
    if total < 1:
         return ''
    accuracy = correct/total *100
    #result = 'Face Authentication accuracy for person : '+str(target_label)+' ||||  '+ str(accuracy) 
    result = 'Face Authentication accuracy  |||| :::  '+ str(accuracy) 
    #results.append(result)
    print(result)
    return result
'''