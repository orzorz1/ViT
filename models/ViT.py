from modules.ViT_modules import *
from einops.layers.torch import Rearrange

class ViTSeg(nn.Module):
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

        self.pool = pool
        self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes),
        #     nn.Sigmoid()
        # )

        self.to_reconstructed = nn.Sequential(
            nn.Linear(dim, patch_height * patch_width * patch_depth),
            Rearrange('b (h w d) (p1 p2 p3 c) -> b c (h p1) (w p2) (d p3)', h=(image_height // patch_height),
                      w=(image_width // patch_width), d=(image_depth // patch_depth), p1=patch_height,
                      p2=patch_width, p3=patch_depth, c=1),
        )

        self.to_regression = nn.Sequential(
            nn.Linear(dim, 1),
        )

        self.sigmoid = nn.Sigmoid()

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

        x = self.to_reconstructed(x)
        # x = self.sigmoid(x)

        return {"out": x, "token": cls_tokens}


if __name__ == "__main__":
    model = ViTSeg(image_size=112, patch_size=7, num_classes=3, dim=768, depth=6, heads=12, mlp_dim=2048, channels=1,
                   learned_pos=False, use_token=True)
    x = torch.randn([5,1,112,112,112])
    print(x.shape)
    y = model(x)
    print(y['out'].shape)
