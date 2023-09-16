from modules.ViT_bolck import *

class ViTSeg(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.,
                 learned_pos=True, use_token=False):
        super().__init__()
        self.dim = dim
        self.image_size = image_size
        self.encoder = ViT_encoder(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, channels=channels,
                   pool=pool, dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout, learned_pos=learned_pos, use_token=use_token)
        self.decoder = UNet_decoder(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim)

    def forward(self, img):
        batch_size = img.shape[0]
        x = self.encoder(img)
        x['embedding'] = x['embedding'].permute(0, 2, 1).view(batch_size, self.dim, self.image_size[0]//16,self.image_size[1]//16,self.image_size[2]//16)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    model = ViTSeg(image_size=[128,128,32], patch_size=16, num_classes=3, dim=768, depth=6, heads=12, mlp_dim=2048, channels=1,
                   learned_pos=False, use_token=True)
    x = torch.randn([5,1,128,128,32])
    print(x.shape)
    y = model(x)
    print(y['out'].shape)
