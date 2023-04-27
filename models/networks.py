from .unet_parts import *
import numbers
from einops import rearrange

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # 512*
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # 512*3
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        # 512
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def shuffle_data(self, x, block):
        B, C, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        if h % block != 0 or w % block != 0:
            raise ValueError(f'Feature map size {(h, w)} not divisible by block ({block})')
        x = x.reshape(-1, block, int(h // block),
                      block,  int(w // block), C)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, h*w, C)
        x = rearrange(x, "b (h w) c -> b c h w", h=h)
        return x

    def shuffle_back(self, x, block):
        B, C, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        x = x.reshape(-1,  int(h // block), int(w // block),
                      block, block,  C)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(B, h * w, C)
        x = rearrange(x, "b (h w) c -> b c h w", h=h)
        return x

    def forward(self, x, shuffle):
        b, c, h, w = x.shape
        block_size = 4
        if shuffle:
            x = self.shuffle_data(x, 4)

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (c head) (h p1) (w p2) -> (b h w) head c (p1 p2)', head=self.num_heads, p1=block_size,
                      p2=block_size)

        k = rearrange(k, 'b (c head) (h p1) (w p2) -> (b h w) head c (p1 p2)', head=self.num_heads, p1=block_size,
                      p2=block_size)
        v = rearrange(v, 'b (c head) (h p1) (w p2) -> (b h w) head c (p1 p2)', head=self.num_heads, p1=block_size,
                      p2=block_size)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, '(b h w) head c (p1 p2) -> b (c head) (h p1) (w p2)', b=b, h=h // block_size,
                        w=w // block_size, head=self.num_heads, p1=block_size, p2=block_size)


        out = self.project_out(out)
        if shuffle:
            out = self.shuffle_back(out, 4)
        return out

class AttentionBlock(nn.Module):
    def __init__(self,
                 dim,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 num_heads=4,
                 LayerNorm_type='WithBias',
                 shuffle = True
                 ):
        super(AttentionBlock, self).__init__()
        self.shuffle = shuffle
        self.norm1 = LayerNorm(dim=int(dim), LayerNorm_type=LayerNorm_type)
        self.ffn = FeedForward(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor, bias=bias)
        self.norm2 = LayerNorm(dim=int(dim), LayerNorm_type=LayerNorm_type)
        self.att = Attention(dim=int(dim), num_heads=num_heads, bias=bias)

    def forward(self, input):
        att_op = input
        att_op = att_op + self.att(self.norm2(att_op), self.shuffle)
        att_op = att_op + self.ffn(self.norm1(att_op))

        return att_op

class attblock(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False, ws=2):
        super(attblock, self).__init__()
        self.att = Attention(dim, num_heads, bias, ws)

    def forward(self, x):
        x = self.att(x, True)
        return x

class UNetEncoder(nn.Module):
    def __init__(self, n_channels=3, shuffle = True):
        super(UNetEncoder, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = nn.Sequential(
            down(64, 128),
            AttentionBlock(128, shuffle)
            # attblock(128)
        )
        self.down2 = nn.Sequential(
            down(128, 256),
            AttentionBlock(256, shuffle)
        )
        self.down3 = nn.Sequential(
            down(256, 512),
            AttentionBlock(512, shuffle)
        )
        self.down4 = nn.Sequential(
            down(512, 512),
            AttentionBlock(512, shuffle)
        )


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x5, (x1, x2, x3, x4)

class UNetDecoder(nn.Module):
    def __init__(self, n_channels=3, shuffle=True, bilinear=True):
        super(UNetDecoder, self).__init__()
        self.up1 = up(1024, 256)
        self.up12 = AttentionBlock(256, shuffle)
        self.up2 = up(512, 128)
        self.up22 = AttentionBlock(128, shuffle)
        self.up3 = up(256, 64)
        self.up32 = AttentionBlock(64, shuffle)
        self.up4 = up(128, 64)
        self.up42 = AttentionBlock(64, shuffle)
        self.outc = outconv(64, n_channels)
        self.sigmoid = nn.Sigmoid()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 上采样
        else:
            self.up = nn.ConvTranspose2d(256, 512, 2, stride=2)
        self.conv = nn.Conv2d(512, 512, 1, 1)

    def forward(self, x, enc_outs, x_c, domain_drive):
        x = self.sigmoid(x)
        if domain_drive:
            x_c = self.up(x_c)
            x = torch.cat(x, x_c)
            x = self.conv(x)
        x = self.up1(x, enc_outs[3])
        x = self.up12(x)
        x = self.up2(x, enc_outs[2])
        x = self.up22(x)
        x = self.up3(x, enc_outs[1])
        x = self.up32(x)
        x = self.up4(x, enc_outs[0])
        x = self.up42(x)
        x = self.outc(x)
        return nn.Tanh()(x)

class Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(Classifier, self).__init__()
        self.num = num_classes
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=3, stride=1, padding=1)
        )
        self.encoder = UNetEncoder()

    def forward(self, x):
        for i in range(len(self.classifier)):
            x = self.classifier[i](x)
            if i == 3:
                x_c = x
        b, _, h, _ = x.shape
        x = nn.MaxPool2d(kernel_size=h, stride=h)(x)
        return x.view(b, self.num), x_c
