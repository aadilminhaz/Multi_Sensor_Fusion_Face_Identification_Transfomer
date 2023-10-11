#IR_Fusion based Face Recognition Transfomer Deep Network
##########

'''
Research Reference:
- ECAPA-TDNN - https://github.com/TaoRuijie/ECAPA-TDNN
- face transformer for recognition -  https://arxiv.org/pdf/2103.14803.pdf
- Paper face transformer for recognition - https://github.com/zhongyy/Face-Transformer

Datasets in use
--------------------------
IR_RGB Combo data set used - http://tdface.ece.tufts.edu/downloads/TD_IR_RGB_CROPPED/
in other test --
IR Facial Dataset source:
http://tdface.ece.tufts.edu/downloads/TD_IR_A/
http://tdface.ece.tufts.edu/downloads/TD_IR_E/

'''

#Load the dependencies
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image

'''
---------------Vision Image Transformer----------------
All ViT components
- PatchModule
- Embedding
- Multi-Head Attention Module
- MLP
- Normalisation, Residual connection, Dropout'''


##Load sub-components
import Encoder
import AAMSoftmax
import Fusion_PatchEmbed
#from model_components.Encoder import Encoder
#from model_components.AAMSoftmax import AAMSoftmax
#from model_components.Fusion_PatchEmbed import Fusion_PatchEmbed


class FusionVisionFaceTransfomer(nn.Module):
    def __init__(self, img_size = 250, patch_size = 25, in_channels=3, n_classes=6000, embed_dim=768, depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True, p=0., attn_p=0., GPU_ID=''):
        super(FusionVisionFaceTransfomer, self).__init__()

        #Embedding to perform fusion on two images from two different sensors and convert into one patched layer
        self.patch_embed = Fusion_PatchEmbed.Fusion_PatchEmbed(image_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1,1, embed_dim)) #setting up classification token in the patched input
        self.pos_embed = nn.Parameter(torch.zeros(1, 1+self.patch_embed.num_patches*2, embed_dim))  #positional embedding of the input
        self.pos_drop = nn.Dropout(p=p)
        self.encoder_blocks = nn.ModuleList([
            Encoder.Encoder(dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: ", self.device, f"({torch.cuda.get_device_name(self.device)})" if torch.cuda.is_available() else "")

        self.s=64.0
        self.m=0.50

        m = 0.4
        s = 64.0
        self.loss = AAMSoftmax.AAMSoftmax(n_class = n_classes, m = m, s =s).to(self.device)


    def forward(self, input_image_rgb, input_image_ir,  label=None):

        n_samples_inp = input_image_rgb.shape[0]
        x = self.patch_embed(input_image_rgb, input_image_ir).to(self.device)
        cls_token_inp = self.cls_token.expand(n_samples_inp, -1, -1)
        x = torch.cat((cls_token_inp, x), dim=1)
        x = x + self.pos_embed.to(self.device)
        x = self.pos_drop(x).to(self.device)

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)

        x = self.norm(x)

        cls_token_final = x[:, 0]
        x = self.to_latent(cls_token_final)

        face_embedding_input = self.mlp_head(x)



        if label is not None:
            x = self.loss(face_embedding_input, label)
            return x, face_embedding_input
        else:
            return face_embedding_input
