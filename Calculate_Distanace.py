import torch
import torch.nn as nn
from PIL import Image
import numpy as np

import Utils
from Utils import pair_show_images


##Method to calculate COSINE-Similarity
def calculate_distance(input_embedding, original_embedding):
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    distance = cos(input_embedding, original_embedding)
    return distance

#Method to authenticate and Input-Image(combo or RGB-IR) with Original Image        
def calculate_distances(input_rgb_image, input_ir_image, original_rgb_image, original_ir_image, model, device = 'cpu', input_label=None, original_label=None, batch_size=32):
    #with torch.no_grad():
    correct, total = 0, 0
    test_loss = 0.0
    x_rgb, x_ir, xo_rgb, xo_ir = input_rgb_image[None,:,:,:], input_ir_image[None,:,:,:], original_rgb_image[None,:,:,:], original_ir_image[None,:,:,:]
    x_rgb, x_ir, xo_rgb, xo_ir = x_rgb.to(device), x_ir.to(device), xo_rgb.to(device), xo_ir.to(device)

    #print('shapes : x_rgb : ',x_rgb.shape, ' x_ir :', x_ir.shape, ' xo_rgb : ', xo_rgb.shape, ' xo_ir :', xo_ir.shape)

    input_embedding = model(input_image_rgb=x_rgb, input_image_ir=x_ir)
    original_embedding = model(input_image_rgb=xo_rgb, input_image_ir=xo_ir)

    

    #print('Face Embedding after transformer : input : ', input_embedding.shape)
    #To print images, place it in one array
    input_images = []
    input_images.append(input_rgb_image)
    input_images.append(input_ir_image)
    #pair_show_images(input_images, input_label, batch_size)
    

    #print('Face Embedding after transformer : original : ', original_embedding.shape)
    original_images = []
    original_images.append(original_rgb_image)
    original_images.append(original_ir_image)
    #pair_show_images(original_images, original_label, batch_size)

    distance = calculate_distance(input_embedding, original_embedding)
    return distance



def calculate_POS_distance(test_labels, model, device='cpu', batch_size=32):
     
    total = 0.0
    TRUE_POSITIVE = 0.0
    positive_distances = []
    for test_label in test_labels:
        test_label = test_label.item()
        #print('**********************\nEvaluating for person : ',test_label)
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

                

        result = calculate_distances(input_rgb_image, input_ir_image, original_rgb_image, original_ir_image, model, device, test_label, test_label, batch_size )
        #print('Calculated positive distance for person : ',test_label, ' - ', result)
        positive_distances.append(result.item())
            
    if total < 1:
        return ''
    return positive_distances



def calculate_NEG_distance(test_labels, model, device='cpu', batch_size=32):
     
    total = 0.0
    negative_distances = []
    for i in range(len(test_labels)):
        last = i+1
        #print('fist label : ', test_labels[i], ' last label :', test_labels[-last])
        
             
        fist_label = test_labels[i].item()
        second_label = test_labels[-last].item()
        #print('**********************\nEvaluating for person : ',fist_label, ' with person : ', second_label)
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

                

        result = calculate_distances(input_rgb_image, input_ir_image, original_rgb_image, original_ir_image, model, device, fist_label, second_label, batch_size )
        #print('Calculated Negative distance for person : ',fist_label, ' - ', result)
        negative_distances.append(result.item())
       
    if total < 1:
        return ''
    return negative_distances

