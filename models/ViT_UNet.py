from modules.ViT_bolck import *

class ViTSeg(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.,
                 learned_pos=True, use_token=False):
        super().__init__()
        self.encoder = ViT_encoder(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, channels=channels,
                   pool=pool, dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout, learned_pos=learned_pos, use_token=use_token)
        self.decoder = UNet_decoder(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim)

    def forward(self, img):
        x = self.encoder(img)
        print(x['embedding'].shape)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    model = ViTSeg(image_size=128, patch_size=16, num_classes=3, dim=768, depth=6, heads=12, mlp_dim=2048, channels=1,
                   learned_pos=False, use_token=True)
    x = torch.randn([5,1,128,128,128])
    print(x.shape)
    y = model(x)
    print(y['out'].shape)
