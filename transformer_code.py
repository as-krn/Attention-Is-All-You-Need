import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# -------------------------
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mekanizması"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Query, Key, Value için linear katmanlar
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Scaled Dot-Product Attention"""
        # Attention skorlarını hesapla
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Maskeyi uygula (decoder için)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Softmax uygula
        attention_weights = F.softmax(scores, dim=-1)
        
        # Value ile çarp
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformasyonları uygula ve head'lere böl
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention hesapla
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Head'leri concat et
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.W_o(attention_output)
        return output


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Katmanı"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Katmanı"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        # Masked self-attention
        attn_output = self.self_attention(x, x, x, self_attn_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention with encoder output
        attn_output = self.cross_attention(x, encoder_output, encoder_output, cross_attn_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder"""
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, 
                 n_layers: int, d_ff: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Embedding + positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Encoder katmanları
        for layer in self.layers:
            x = layer(x, mask)
            
        return x


class TransformerDecoder(nn.Module):
    """Transformer Decoder"""
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, 
                 n_layers: int, d_ff: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        # Embedding + positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Decoder katmanları
        for layer in self.layers:
            x = layer(x, encoder_output, self_attn_mask, cross_attn_mask)
            
        return x


class Transformer(nn.Module):
    """Complete Transformer Model"""
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, 
                 d_model: int = 512, n_heads: int = 8, n_layers: int = 6,
                 d_ff: int = 2048, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        
        self.encoder = TransformerEncoder(src_vocab_size, d_model, n_heads, 
                                        n_layers, d_ff, max_len, dropout)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, n_heads, 
                                        n_layers, d_ff, max_len, dropout)
        
        self.linear = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encoder
        encoder_output = self.encoder(src, src_mask)
        
        # Decoder
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        
        # Final linear layer
        output = self.linear(decoder_output)
        
        return output
    
    def generate_square_subsequent_mask(self, sz):
        """Decoder için causal mask oluştur"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask




# -------------------------
# Vocabulary
# -------------------------
class Vocab:
    def __init__(self, texts, min_freq=1):
        counter = Counter(token for text in texts for token in text.split())
        self.token2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        for token, freq in counter.items():
            if freq >= min_freq:
                self.token2idx[token] = len(self.token2idx)
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
    
    def encode(self, text):
        return [self.token2idx.get(token, self.token2idx['<unk>']) for token in text.split()]
    
    def decode(self, ids):
        return ' '.join(self.idx2token[i] for i in ids if i not in [0,1,2])

# -------------------------
# Dataset
# -------------------------
class TextDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab, max_len=20):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src_ids = [self.src_vocab.token2idx['<sos>']] + self.src_vocab.encode(self.src_texts[idx]) + [self.src_vocab.token2idx['<eos>']]
        tgt_ids = [self.tgt_vocab.token2idx['<sos>']] + self.tgt_vocab.encode(self.tgt_texts[idx]) + [self.tgt_vocab.token2idx['<eos>']]
        # Padding
        src_ids = src_ids[:self.max_len] + [0]*(self.max_len - len(src_ids))
        tgt_ids = tgt_ids[:self.max_len] + [0]*(self.max_len - len(tgt_ids))
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

# -------------------------
# Mask Fonksiyonları
# -------------------------
def create_padding_mask(seq, pad_idx=0):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)

def create_causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask  # (seq_len, seq_len)

# -------------------------
# Transformer modelin burada olmalı
# (Senin mevcut Transformer kodun, MultiHeadAttention, Encoder/Decoder vs)
# -------------------------
# Buraya mevcut Transformer sınıfını ve alt sınıfları ekle

# -------------------------
# Train Step (metinler için)
# -------------------------
def train_step(model, src, tgt, optimizer, criterion, device='cpu'):
    model.train()
    optimizer.zero_grad()
    
    src, tgt = src.to(device), tgt.to(device)
    
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]
    
    # Masks
    src_mask = create_padding_mask(src)
    tgt_padding_mask = create_padding_mask(tgt_input)
    tgt_seq_len = tgt_input.size(1)
    causal_mask = create_causal_mask(tgt_seq_len).to(device)
    # combine masks: padding + causal
    tgt_mask = causal_mask.unsqueeze(0) | ~tgt_padding_mask.bool()
    
    # Forward pass
    output = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
    
    # Loss
    loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

# -------------------------
# Örnek Kullanım
# -------------------------
if __name__ == "__main__":
    # Örnek metinler
    src_texts = ["merhaba dünya", "nasılsın bugün", "benim adım ahmet selçuk"]
    tgt_texts = ["hello world", "how are you today", "my name is ahmet selçuk"]

    # Vocabulary oluştur
    src_vocab = Vocab(src_texts)
    tgt_vocab = Vocab(tgt_texts)

    dataset = TextDataset(src_texts, tgt_texts, src_vocab, tgt_vocab, max_len=10)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Model parametreleri
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Transformer(len(src_vocab.token2idx), len(tgt_vocab.token2idx),
                        d_model=128, n_heads=4, n_layers=2, d_ff=512).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Eğitim döngüsü
    for epoch in range(5):
        total_loss = 0
        for src, tgt in dataloader:
            loss = train_step(model, src, tgt, optimizer, criterion, device)
            total_loss += loss
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(dataloader):.4f}")

    # Test inference
    model.eval()
    with torch.no_grad():
        test_src = torch.tensor([[src_vocab.token2idx['<sos>'], src_vocab.token2idx.get('merhaba', 3), src_vocab.token2idx['<eos>']]]).to(device)
        test_tgt = torch.tensor([[tgt_vocab.token2idx['<sos>']]]).to(device)

        output = model(test_src, test_tgt)
        predicted_tokens = torch.argmax(output, dim=-1)
        print("Predicted:", [tgt_vocab.idx2token[t] for t in predicted_tokens[0].cpu().tolist()])
