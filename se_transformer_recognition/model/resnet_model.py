import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):

        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        # Shortcut Path (Identity)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # RESIDUAL CONNECTION: Output + Input
        out += self.shortcut(x)

        out = self.relu(out)
        return out

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):

        super(ResNetFeatureExtractor, self).__init__()

        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.layer1 = self._make_layer(BasicBlock, 128, num_blocks=1, stride=1)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.layer2 = self._make_layer(BasicBlock, 256, num_blocks=2, stride=1)

        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.layer3 = self._make_layer(BasicBlock, 512, num_blocks=2, stride=1)

        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.layer4 = self._make_layer(BasicBlock, 512, num_blocks=1, stride=1)

        self.final_pool = nn.AdaptiveAvgPool2d((2, 32))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.layer1(x)
        x = self.pool2(x)

        x = self.layer2(x)
        x = self.pool3(x) 

        x = self.layer3(x)
        x = self.pool4(x)

        x = self.layer4(x)

        x = self.final_pool(x)
        return x

class PatchEncoder(nn.Module):
    def __init__(self, in_channels, emb_dim, k1=2, k2=1, max_patches=256):
        """
        Implements PATCHENC(k1, k2).
        """
        super().__init__()

        self.k1 = k1
        self.k2 = k2

        # Linear projection via Conv2d (ViT-style)
        self.proj = nn.Conv2d(
            in_channels,
            emb_dim,
            kernel_size=(k1, k2),
            stride=(k1, k2)
        )

        self.pos_emb = nn.Parameter(
            torch.zeros(max_patches, emb_dim)
        )
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, F):
        """
        F: (B, C, H', W')
        """
        x = self.proj(F)  

        B, D, Hp, Wp = x.shape
        N = Hp * Wp

        x = x.flatten(2).transpose(1, 2) 

        x = x + self.pos_emb[:N].unsqueeze(0)

        return x, N

def make_encoder(emb_dim=384, nhead=8, num_layers=3, dim_feedforward=1024, dropout=0.1):

    enc_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead,
                                           dim_feedforward=dim_feedforward,
                                           dropout=dropout, activation='relu')
    
    encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    return encoder

class Merger(nn.Module):
    def __init__(self, emb_dim, max_total_len=4096):

        super().__init__()

        self.global_pos = nn.Parameter(torch.zeros(max_total_len, emb_dim))
        nn.init.trunc_normal_(self.global_pos, std=0.02)

    def merge_and_pad(self, enc_list):
        """
        enc_list: list of tensors (B,), where each tensor is (L_i, D)
        Returns:
          enc_batch: (B, Lmax, D)
          mask: (B, Lmax) Boolean (True=Padding)
        """

        device = enc_list[0].device
        B = len(enc_list)
        D = enc_list[0].size(1)

        lengths = [t.size(0) for t in enc_list]
        Lmax = max(lengths)

        enc_batch = torch.zeros(B, Lmax, D, device=device)
        mask = torch.ones(B, Lmax, dtype=torch.bool, device=device)

        for i, t in enumerate(enc_list):
            L_i = t.size(0)
            enc_batch[i, :L_i, :] = t
            mask[i, :L_i] = False

        enc_batch = enc_batch + self.global_pos[:Lmax, :].unsqueeze(0)

        return enc_batch, mask
    
class TransformerDecoderWrapper(nn.Module):
    def __init__(self, vocab_size, emb_dim, nhead=8, num_layers=3, pad_idx=0, max_len=256):

        super().__init__()

        self.tok_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        dec_layer = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=emb_dim*4, dropout=0.1)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.pos_emb = nn.Parameter(torch.zeros(max_len, emb_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.1)
        self.out_proj = nn.Linear(emb_dim, vocab_size)
        self.pad_idx = pad_idx

    def generate_square_subsequent_mask(self, sz):

        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tgt_tokens, memory, memory_key_padding_mask):
        B, T = tgt_tokens.size()
        device = tgt_tokens.device

        tok = self.tok_emb(tgt_tokens)
        pos = self.pos_emb[:T,:].unsqueeze(0).expand(B,-1,-1)
        tgt = (tok + pos).transpose(0,1)

        tgt_key_padding_mask = (tgt_tokens == self.pad_idx)

        if memory_key_padding_mask is not None:
             memory_key_padding_mask = memory_key_padding_mask.bool()

        tgt_mask = self.generate_square_subsequent_mask(T).to(device)

        mem = memory.transpose(0,1)

        dec_out = self.decoder(
            tgt,
            mem,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask 
        )

        logits = self.out_proj(dec_out.transpose(0,1))
        return logits

class KhmerOCR(nn.Module):
    def __init__(self, vocab_size, pad_idx=0, emb_dim=256, max_global_len=4096):

        super().__init__()

        self.cnn = ResNetFeatureExtractor()

        self.patch = PatchEncoder(512, emb_dim=emb_dim, k1=2, k2=1)

        self.enc = make_encoder(emb_dim=emb_dim, nhead=8, num_layers=2)

        self.global_pos = nn.Parameter(torch.zeros(max_global_len, emb_dim))
        nn.init.trunc_normal_(self.global_pos, std=0.02)

        self.dec = TransformerDecoderWrapper(vocab_size, emb_dim=emb_dim, nhead=8,
                                             num_layers=2, pad_idx=pad_idx)
        self.pad_idx = pad_idx

    def forward(self, chunk_lists, tgt_tokens):
        """
        chunk_lists: List of Lists of Tensors.
                     Img1: [C1, C2]
                     Img2: [C1, C2, C3]
        tgt_tokens: (B, L)
        """
        chunk_sizes = [len(c) for c in chunk_lists]

        flat_input_list = [chunk for img_chunks in chunk_lists for chunk in img_chunks]

        flat_input = torch.stack(flat_input_list)

        f = self.cnn(flat_input)

        p, _ = self.patch(f)

        p = p.transpose(0, 1).contiguous()
        enc_out = self.enc(p)

        enc_out = enc_out.transpose(0, 1)

        batch_encoded_list = []
        cursor = 0
        feature_dim = enc_out.size(-1)

        for size in chunk_sizes:

            img_chunks = enc_out[cursor : cursor + size]

            merged_seq = img_chunks.reshape(-1, feature_dim)

            batch_encoded_list.append(merged_seq)
            cursor += size

        memory = pad_sequence(batch_encoded_list, batch_first=True, padding_value=0.0)

        B, T, _ = memory.shape

        limit = min(T, self.global_pos.size(0))
        pos_emb = self.global_pos[:limit, :].unsqueeze(0)

        if T > self.global_pos.size(0):
             memory = memory[:, :limit, :] + pos_emb

             T = limit
        else:
             memory = memory + pos_emb

        memory_key_padding_mask = torch.ones((B, T), dtype=torch.bool, device=memory.device)

        for i, seq in enumerate(batch_encoded_list):
            valid_len = min(seq.shape[0], T) 
            memory_key_padding_mask[i, :valid_len] = False

        # Decoder
        logits = self.dec(tgt_tokens, memory, memory_key_padding_mask)

        return logits