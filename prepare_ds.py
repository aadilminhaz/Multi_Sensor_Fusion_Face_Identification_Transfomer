##Just one time to create the data_label mapping
#The code loads Rgb and IR images in a csv and puts labels against the combo
import os
import csv

from torchvision import io

# Path to the output CSV file
#output_csv = 'ir_rgb_label_train.csv'
# Path to the directory containing images

def prepare_ds_source(output_csv, rgb_source, ir_source):
    print('Preparing Data_Source_File to create custom dataset...')
    # Open the CSV file for writing
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
    
        # Write the header row
        csvwriter.writerow(['rgb_image', 'ir_image', 'label'])

        ir_image_names = []
        rgb_image_names = []

    
        for filename in os.listdir(rgb_source):
            if filename.endswith('.jpg') or filename.endswith('.jpeg'):
                rgb_image_names.append(os.path.splitext(filename)[0])
        
        for filename in os.listdir(ir_source):
            if filename.endswith('.jpg') or filename.endswith('.jpeg'):
                ir_image_names.append(os.path.splitext(filename)[0])
   
        print('Length of rgb_images : ', len(rgb_image_names))
        print('Length of ir_images : ', len(ir_image_names))

        for i in range(len(rgb_image_names)):
            rgb_image_name = rgb_image_names[i]+'.jpg'
            print('processing : i-',i,' image :',rgb_image_name)
            rgb_image_label = rgb_image_name.split('-')[0]

            ir_image_name = ir_image_names[i]+'.jpg'
            ir_image_label = ir_image_name.split('-')[0]
        
            if rgb_image_label == ir_image_label:
                csvwriter.writerow([rgb_image_name, ir_image_name, ir_image_label])


        print("CSV file created successfully.")





