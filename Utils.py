
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch



rgb_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

ir_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def show_images(images, labels, image_at_once):
    fig = plt.figure(figsize=(12, 10))  # Adjust the figure size as needed
    for i in range(image_at_once):  # Display 32 images
        plt.subplot(4, 8, i + 1)  # Create a 4x8 grid of subplots#
        plt.imshow(images[i][0], cmap='gray', interpolation='none')
        plt.title("Name: {}".format(labels[i]))
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()

def pair_show_images(image_pair, labels=None, image_at_once=2):
    fig = plt.figure(figsize=(12, 10))  # Adjust the figure size as needed
    pair_label_counter = 0
    print('labels :',labels)
    for i in range(image_at_once):  # Display 32 images
        plt.subplot(4, 8, i + 1)  # Create a 4x8 grid of subplots#
        image_to_show = image_pair[i].permute(2,1,0)
        plt.imshow(image_to_show)
        #plt.imshow(image_pair[i][0], cmap='gray', interpolation='none')
        if labels is not None:
            #if (i%2 == 0):
            #    pair_label_counter += 1
            #plt.title("Name: {}".format(labels[pair_label_counter]))
            plt.title("Name: {}".format(labels))
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()

def unique_elements_from_lists(*lists):
    
    unique_elements = torch.unique(torch.cat((lists)))
    #print('Input :', lists.count)
    #for list in lists:
    #    unique_set = set().union(list)
    #    unique_list = list(unique_set)
    #final_unique_set = set().union(list)
    #final_unique_list = list(final_unique_set)
    #return final_unique_list


def find_common_elements(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    common_elements = set1.intersection(set2)
    return list(common_elements)
