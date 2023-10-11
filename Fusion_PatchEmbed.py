import torch
import torch.nn as nn

class Fusion_PatchEmbed(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super(Fusion_PatchEmbed, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x_rgb, x_ir):
        #Inputs - rgb and ir
        batch_size_rgb, in_channels_rgb, _, _ = x_rgb.shape
        batch_size_ir, in_channels_ir, _, _ = x_ir.shape

        # Reshape the rgb and ir inputs tensor to form patches
        patches_rgb = x_rgb.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches_rgb = patches_rgb.contiguous().view(batch_size_rgb, in_channels_rgb, -1, self.patch_size * self.patch_size)

        patches_ir = x_ir.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches_ir = patches_ir.contiguous().view(batch_size_ir, in_channels_ir, -1, self.patch_size * self.patch_size)

        # Reshape the rgb and ir patches and apply projection
        patches_rgb = patches_rgb.transpose(2, 3).contiguous().view(batch_size_rgb, -1, in_channels_rgb * self.patch_size * self.patch_size)
        x_rgb = self.proj(patches_rgb)

        patches_ir = patches_ir.transpose(2, 3).contiguous().view(batch_size_ir, -1, in_channels_ir * self.patch_size * self.patch_size)
        x_ir = self.proj(patches_ir)

        patch_embed = torch.cat((x_rgb, x_ir), dim=1)

        return patch_embed