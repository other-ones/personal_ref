import sys
sys.path.insert(0, './packages')

from diffusers.models.vae import Encoder
import torch 
from torch import nn
import pdb
# from datasets.ml_scripts import scripts,script2idx,idx2script,char2idx

class TextRecNetML(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_dim=1000,
        max_char=25,
    ):
        super().__init__()
    
        self.net = Encoder(
            in_channels=in_channels,
            out_channels=128,
            down_block_types=("DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D"),
            block_out_channels=(32,64,64,128),
            layers_per_block=2,
            norm_num_groups=16,
            act_fn="silu",
            double_z=False,
        )

        self.final = nn.Linear(128, out_dim)

        self.read_tkns = nn.Parameter(torch.empty(1, max_char, 128))
        self.read_tkns.data.normal_(0.0, 0.02)

        self.ca_layers = nn.ModuleList()
        for _ in range(4):
            self.ca_layers.append(nn.MultiheadAttention(128, num_heads=4, batch_first=True))

        self.pos_emb = nn.Parameter(torch.zeros(1, 64, 128))

    def forward(self, x):
        # pdb.set_trace()
        x = self.net(x).flatten(2).permute(0,2,1)
        x = x + self.pos_emb
        
        read_tkns = self.read_tkns.repeat(x.shape[0], 1, 1)

        for layer in self.ca_layers:
            read_tkns, _ = layer(read_tkns, x, x)

        x = self.final(read_tkns)
        return x

if __name__ == '__main__':
    net = TextRecNetML(4, 96).cuda()
    data = torch.randn(10, 4, 32, 128).cuda()
    print(net(data).shape)