from modules.ViT_modules import *
from einops.layers.torch import Rearrange
import modules.UNet_parts as up


class ViT_encoder(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.,
                 learned_pos=True, use_token=False):
        super().__init__()
        image_height, image_width, image_depth = pair(image_size)
        patch_height, patch_width, patch_depth = pair(patch_size)
        self.use_token = use_token

        assert image_height % patch_height == 0 and image_width % patch_width == 0 and image_depth % patch_depth == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (image_depth // patch_depth)
        patch_dim = channels * patch_height * patch_width * patch_depth
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1=patch_height, p2=patch_width, p3=patch_depth),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) if learned_pos else nn.Parameter(
            positional_encoding(num_patches+1, dim), requires_grad=False)
        if self.use_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_regression = nn.Sequential(
            nn.Linear(dim, 1),
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        if self.use_token:
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding[:, :n + self.use_token]
        x = self.dropout(x)

        x = self.transformer(x)

        if self.use_token:
            cls_tokens = x[:, 0, :]
            x = x[:, 1:, :]
            cls_tokens = self.to_regression(cls_tokens)

        return {"embedding": x, "token": cls_tokens}


class UNet_decoder(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim):
        super().__init__()
        image_height, image_width, image_depth = pair(image_size)
        patch_height, patch_width, patch_depth = pair(patch_size)
        self.patch_height = patch_height
        self.deconv_1 = nn.ConvTranspose3d(dim, 512, kernel_size=2, stride=2)
        self.right_conv_1 = up.double_conv(512, 512)
        self.deconv_2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.right_conv_2 = up.double_conv(256, 256)
        self.deconv_3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.right_conv_3 = up.double_conv(128, 128)
        self.deconv_4 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.right_conv_4 = up.double_conv(64, 64)
        self.deconv_5 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.right_conv_5 = up.double_conv(32, 32)
        self.last_conv_16 = nn.Conv3d(64, 1, (3, 3), padding=1)
        self.last_conv_32 = nn.Conv3d(32, 1, (3, 3), padding=1)
        self.rearrange = Rearrange('b (h w d) (p1 p2 p3 c) -> b c (h p1) (w p2) (d p3)', h=(image_height // patch_height),
                  w=(image_width // patch_width), d=(image_depth // patch_depth), p1=patch_height,
                  p2=patch_width, p3=patch_depth, c=1),
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        embedding = input['embedding']
        cls_tokens = input['token']
        x = self.deconv_1(embedding)
        x = self.right_conv_1(x)
        x = self.deconv_2(x)
        x = self.right_conv_2(x)
        x = self.deconv_3(x)
        x = self.right_conv_3(x)
        x = self.deconv_4(x)
        x = self.right_conv_4(x)
        if self.patch_height == 16:
            x = self.last_conv_16(x)
        elif self.patch_height == 32:
            x = self.deconv_5(x)
            x = self.right_conv_5(x)
            x = self.last_conv_32(x)
        else:
            raise ZeroDivisionError("ViT_patch_size暂时只能为16或32")
        x = self.rearrange(x)
        return {"out": x, "token": cls_tokens}

if __name__ == "__main__":
    model = UNet_decoder(image_size=128, patch_size=16, num_classes=3, dim=768)
    x = torch.randn([5,1,128,128,128])
    print(x.shape)
    y = model(x)
    print(y['out'].shape)



