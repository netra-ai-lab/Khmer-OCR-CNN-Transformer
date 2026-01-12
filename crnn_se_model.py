import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# ==========================================
# 1. NEW: Sequence-Aware SE Block (1D-SE)
# ==========================================
class SequenceSE(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SequenceSE, self).__init__()
        # We process each column (W) independently
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.size()

        # 1. Pool Height Only (Average vertical pixels) -> [B, C, 1, W]
        y = torch.mean(x, dim=2).view(b, c, w)

        # 2. Excitation (1D Conv) -> [B, C, W]
        y = self.fc(y)

        # 3. Reshape for broadcasting -> [B, C, 1, W]
        y = y.view(b, c, 1, w)

        # 4. Scale
        return x * y

# ==========================================
# 2. VGG Backbone (Renamed & Upgraded)
# ==========================================
class ImprovedFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        # Block 1 (Edges)
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True))
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2 (Textures)
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True))
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3 (Shapes)
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        # Add 1D-SE here to refine shapes before pooling
        self.se3 = SequenceSE(256)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        # Block 4 (Complex Parts)
        self.conv5 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True))
        self.conv6 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True))
        self.se4 = SequenceSE(512)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        # Block 5 (Semantic)
        self.conv7 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(512)
        self.relu7 = nn.ReLU(True)
        self.se5 = SequenceSE(512)

        self.final_pool = nn.AdaptiveAvgPool2d((2, 32))

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))

        # Block 3
        x = self.conv4(self.conv3(x))
        x = self.se3(x) # Refine
        x = self.pool3(x)

        # Block 4
        x = self.conv6(self.conv5(x))
        x = self.se4(x) # Refine
        x = self.pool4(x)

        # Block 5
        x = self.relu7(self.bn7(self.conv7(x)))
        x = self.se5(x) # Refine

        x = self.final_pool(x)
        return x

class PatchEncoder(nn.Module):
    def __init__(self, in_channels, emb_dim, k1=2, k2=1, max_patches=256):
        """
        Implements PATCHENC(k1, k2) from the paper.
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

        # Local positional embedding (maximum length only)
        self.pos_emb = nn.Parameter(
            torch.zeros(max_patches, emb_dim)
        )
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, F):
        """
        F: (B, C, H', W')
        """
        x = self.proj(F)  # (B, D, H'/k1, W'/k2)

        B, D, Hp, Wp = x.shape
        N = Hp * Wp

        # Flatten spatial → sequence
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        # Add LOCAL positional embedding
        x = x + self.pos_emb[:N].unsqueeze(0)

        return x, N

def make_encoder(emb_dim=384, nhead=8, num_layers=3, dim_feedforward=1024, dropout=0.1):
    enc_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead,
                                           dim_feedforward=dim_feedforward,
                                           dropout=dropout, activation='relu')
    encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
    return encoder

# 1. Inherit from nn.Module
class Merger(nn.Module):
    def __init__(self, emb_dim, max_total_len=4096):
        # 2. Initialize the parent class (CRITICAL)
        super().__init__()

        # learned global pos emb for merged sequences up to max_total_len
        self.global_pos = nn.Parameter(torch.zeros(max_total_len, emb_dim))
        nn.init.trunc_normal_(self.global_pos, std=0.02)

    def merge_and_pad(self, enc_list):
        """
        enc_list: list of tensors (B,), where each tensor is (L_i, D)
        Returns:
          enc_batch: (B, Lmax, D)
          mask: (B, Lmax) Boolean (True=Padding)
        """
        # This will now work because enc_list is on GPU
        device = enc_list[0].device
        B = len(enc_list)
        D = enc_list[0].size(1)

        # 1. Calculate max length
        lengths = [t.size(0) for t in enc_list]
        Lmax = max(lengths)

        # 2. Prepare tensors
        enc_batch = torch.zeros(B, Lmax, D, device=device)
        mask = torch.ones(B, Lmax, dtype=torch.bool, device=device)

        # 3. Fill data
        for i, t in enumerate(enc_list):
            L_i = t.size(0)
            enc_batch[i, :L_i, :] = t
            mask[i, :L_i] = False

        # 4. Add global positional embeddings
        # Since this class is now an nn.Module, self.global_pos will
        # automatically move to GPU when you call model.cuda()
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
        # Generates causal mask (Float: -inf for future, 0 for past)
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tgt_tokens, memory, memory_key_padding_mask):
        B, T = tgt_tokens.size()
        device = tgt_tokens.device

        # 1. Embeddings
        tok = self.tok_emb(tgt_tokens)
        pos = self.pos_emb[:T,:].unsqueeze(0).expand(B,-1,-1)
        tgt = (tok + pos).transpose(0,1)

        # 2. Masks
        # [FIXED] Create Boolean Mask for Padding (True = Pad)
        tgt_key_padding_mask = (tgt_tokens == self.pad_idx)

        # [FIXED] Ensure memory mask is Boolean
        if memory_key_padding_mask is not None:
             # Ensure it's bool. If it was float/int 0/1, convert to bool
             memory_key_padding_mask = memory_key_padding_mask.bool()

        # Causal Mask (Float)
        tgt_mask = self.generate_square_subsequent_mask(T).to(device)

        mem = memory.transpose(0,1)

        # 3. Decoder Pass
        dec_out = self.decoder(
            tgt,
            mem,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,     # Pass BOOL here
            memory_key_padding_mask=memory_key_padding_mask # Pass BOOL here
        )

        logits = self.out_proj(dec_out.transpose(0,1))
        return logits

class KhmerOCR(nn.Module):
    def __init__(self, vocab_size, pad_idx=0, emb_dim=256, max_global_len=4096):
        super().__init__()

        # 1. Feature Extractor
        self.cnn = ImprovedFeatureExtractor()

        # 2. Patch Embedding
        self.patch = PatchEncoder(512, emb_dim=emb_dim, k1=2, k2=1)

        # 3. Local Encoder (Runs on chunks)
        self.enc = make_encoder(emb_dim=emb_dim, nhead=8, num_layers=2)

        # 4. Merger
        self.global_pos = nn.Parameter(torch.zeros(max_global_len, emb_dim))
        nn.init.trunc_normal_(self.global_pos, std=0.02)

        # ==========================================
        # 5. NEW: BiLSTM Smoothing Layer
        # ==========================================
        # Helps stitch chunks together smoothly
        self.context_bilstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=emb_dim // 2, # Output dim will be (emb_dim//2) * 2 = emb_dim
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # 6. Decoder
        self.dec = TransformerDecoderWrapper(vocab_size, emb_dim=emb_dim, nhead=8,
                                             num_layers=2, pad_idx=pad_idx)
        self.pad_idx = pad_idx

    def forward(self, chunk_lists, tgt_tokens):
        # 1. Flatten
        chunk_sizes = [len(c) for c in chunk_lists]
        flat_input_list = [chunk for img_chunks in chunk_lists for chunk in img_chunks]
        flat_input = torch.stack(flat_input_list)

        # 2. CNN (With 1D-SE)
        f = self.cnn(flat_input)

        # 3. Patch + Local Encoder
        p, _ = self.patch(f)
        p = p.transpose(0, 1).contiguous()
        enc_out = self.enc(p)
        enc_out = enc_out.transpose(0, 1)

        # 4. Merging
        batch_encoded_list = []
        cursor = 0
        feature_dim = enc_out.size(-1)

        for size in chunk_sizes:
            img_chunks = enc_out[cursor : cursor + size]
            merged_seq = img_chunks.reshape(-1, feature_dim)
            batch_encoded_list.append(merged_seq)
            cursor += size

        # 5. Pad Sequence
        memory = pad_sequence(batch_encoded_list, batch_first=True, padding_value=0.0)

        # 6. Global Positional Embedding
        B, T, _ = memory.shape
        limit = min(T, self.global_pos.size(0))
        pos_emb = self.global_pos[:limit, :].unsqueeze(0)

        if T > self.global_pos.size(0):
             memory = memory[:, :limit, :] + pos_emb
             T = limit
        else:
             memory = memory + pos_emb

        # ==========================================
        # 7. NEW: Apply BiLSTM Smoothing
        # ==========================================
        # This helps the model realize where Chunk 1 ends and Chunk 2 begins
        # memory shape: [Batch, SeqLen, Dim]
        self.context_bilstm.flatten_parameters()
        memory, _ = self.context_bilstm(memory)

        # 8. Create Mask & Decode
        memory_key_padding_mask = torch.ones((B, T), dtype=torch.bool, device=memory.device)
        for i, seq in enumerate(batch_encoded_list):
            valid_len = min(seq.shape[0], T)
            memory_key_padding_mask[i, :valid_len] = False

        logits = self.dec(tgt_tokens, memory, memory_key_padding_mask)

        return logits