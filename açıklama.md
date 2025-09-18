# Transformer Mimarisi: "Attention Is All You Need" Akademik İncelemesi

Bu belge, Vaswani ve arkadaşları (2017) tarafından yayımlanan seminal çalışma "Attention Is All You Need" makalesinde önerilen Transformer mimarisinin kapsamlı bir akademik analizini sunmaktadır.

## İçindekiler
1. [Giriş ve Literatür Bağlamı](#giriş-ve-literatür-bağlamı)
2. [Metodolojik Temel](#metodolojik-temel)
3. [Attention Mekanizmasının Matematiksel Formalizasyonu](#attention-mekanizmasının-matematiksel-formalizasyonu)
4. [Çok Başlıklı Attention Yapısı](#çok-başlıklı-attention-yapısı)
5. [Pozisyonel Kodlama](#pozisyonel-kodlama)
6. [Encoder Mimarisi](#encoder-mimarisi)
7. [Decoder Mimarisi](#decoder-mimarisi)
8. [Model Mimarisi ve Parametrizasyon](#model-mimarisi-ve-parametrizasyon)
9. [Eğitim Metodolojisi](#eğitim-metodolojisi)
10. [Implementasyon Detayları](#implementasyon-detayları)
11. [Deneysel Sonuçlar ve Analiz](#deneysel-sonuçlar-ve-analiz)

---

## Giriş ve Literatür Bağlamı

### Mevcut Yaklaşımların Sınırlılıkları

Sekans modelleme alanında yaygın olarak kullanılan mevcut yaklaşımlar belirli sınırlılıklara sahiptir:

**Recurrent Neural Networks (RNN):** 
- Sıralı hesaplama gerekliliği nedeniyle paralelleştirme imkanının sınırlı olması
- Uzun sekanslar için gradyan kaybolması problemi
- Temporal dependencies'in yakalanmasında başlangıç pozisyonlardan uzaklaşıldıkça performans düşüşü

**Convolutional Neural Networks (CNN):**
- Uzun menzilli bağımlılıkların yakalanması için çok sayıda katman gerekliliği
- Receptive field genişliğinin sınırlı olması
- Sekans uzunluğu ile doğrusal olmayan karmaşıklık artışı

### Araştırma Hipotezi

Bu çalışma, tamamen attention mekanizmasına dayalı bir mimarinin, recurrent ve convolutional katmanlar olmaksızın etkili sekans modelleme gerçekleştirebileceği hipotezini test etmektedir.

---

## Metodolojik Temel

### Sequence-to-Sequence Framework

Transformer modeli, encoder-decoder mimarisini benimser:

```
Input: (x₁, x₂, ..., xₙ) ∈ ℝⁿˣᵈ
Output: (y₁, y₂, ..., yₘ) ∈ ℝᵐˣᵛ
```

Burada d girdi boyutunu, v çıktı sözlük boyutunu ifade eder.

### Temel Model Parametreleri

Base model konfigürasyonu:
- **d_model**: 512 (model boyutu)
- **h**: 8 (attention head sayısı)
- **N**: 6 (encoder/decoder katman sayısı)
- **d_ff**: 2048 (feed-forward ağ boyutu)
- **d_k = d_v**: 64 (key/value projeksiyon boyutu)

---

## Attention Mekanizmasının Matematiksel Formalizasyonu

### Scaled Dot-Product Attention

Temel attention fonksiyonu aşağıdaki formülle tanımlanır:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Burada:
- **Q ∈ ℝⁿˣᵈᵏ**: Query matrisi
- **K ∈ ℝᵐˣᵈᵏ**: Key matrisi  
- **V ∈ ℝᵐˣᵈᵛ**: Value matrisi

### Scaling Factor Analizi

√d_k ile normalizasyon işleminin teorik gerekçesi:

d_k boyutu arttıkça, QK^T çarpımının varyansı d_k ile orantılı olarak artar. Bu durum softmax fonksiyonunun aşırı küçük gradyanlar üretmesine neden olur. Scaling factor bu problemi çözerek gradyan akışının kararlılığını sağlar.

### Algoritma İmplementasyonu

```python
def scaled_dot_product_attention(self, Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights
```

---

## Çok Başlıklı Attention Yapısı

### Teorik Motivasyon

Tek bir attention hesaplaması yerine h farklı öğrenilen doğrusal projeksiyonla paralel attention hesaplaması gerçekleştirilerek, modelin farklı representation alt uzaylarındaki bilgileri eş zamanlı işlemesi sağlanır.

### Matematiksel Formalizasyon

```
MultiHead(Q, K, V) = Concat(head₁, ..., head_h)W^O

head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

Proyeksiyon matrisleri:
- **W_i^Q ∈ ℝᵈᵐᵒᵈᵉˡˣᵈᵏ**: i. head için query projeksiyon matrisi
- **W_i^K ∈ ℝᵈᵐᵒᵈᵉˡˣᵈᵏ**: i. head için key projeksiyon matrisi
- **W_i^V ∈ ℝᵈᵐᵒᵈᵉˡˣᵈᵛ**: i. head için value projeksiyon matrisi
- **W^O ∈ ℝʰᵈᵛˣᵈᵐᵒᵈᵉˡ**: Çıktı projeksiyon matrisi

### Computational Efficiency

Toplam hesaplama karmaşıklığı tek başlıklı attention ile aynı kalır çünkü d_k = d_model/h seçimi yapılır.

---

## Pozisyonel Kodlama

### Problemin Tanımlanması

Transformer mimarisi inherent olarak permutation-invariant olduğundan, sekans sırası bilgisinin açık şekilde modele enjekte edilmesi gereklidir.

### Sinusoidal Pozisyonel Encoding

Deterministik pozisyonel encoding fonksiyonu:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Matematiksel Özellikler

Bu encoding şemasının seçilmesinin nedenleri:

1. **Extrapolation Capability**: Eğitim sırasında görülmeyen sekans uzunluklarına genelleme
2. **Relative Position Information**: PE_pos+k, PE_pos'un doğrusal kombinasyonu olarak ifade edilebilir
3. **Unique Patterns**: Her pozisyon için benzersiz encoding

### Implementasyon

```python
def __init__(self, d_model: int, max_len: int = 5000):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * 
        (-math.log(10000.0) / d_model)
    )
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))
```

---

## Encoder Mimarisi

### Katman Yapısı

Her encoder katmanı iki alt katman içerir:

1. **Multi-head self-attention mekanizması**
2. **Position-wise fully connected feed-forward ağ**

Her alt katman etrafında residual connection ve layer normalization uygulanır.

### Feed-Forward Network Spesifikasyonu

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

Burada W₁ ∈ ℝᵈᵐᵒᵈᵉˡˣᵈff ve W₂ ∈ ℝᵈffˣᵈᵐᵒᵈᵉˡ

### Self-Attention Mekanizması

Encoder'da self-attention için Q, K, V hepsi önceki katmanın çıktısından türetilir:

```
Q = K = V = previous_layer_output
```

---

## Decoder Mimarisi

### Katman Yapısı

Her decoder katmanı üç alt katman içerir:

1. **Masked multi-head self-attention**
2. **Multi-head attention** (encoder çıktısı üzerinde)
3. **Position-wise feed-forward network**

### Causal Masking

Auto-regressive özelliğin korunması için look-ahead masking uygulanır:

```python
def generate_square_subsequent_mask(self, sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask
```

Bu mask, pozisyon i'deki çıktının yalnızca i'den küçük pozisyonlardaki bilgilere bağımlı olmasını sağlar.

### Cross-Attention Mekanizması

İkinci attention katmanında:
- **Q**: Önceki decoder katmanından
- **K, V**: Encoder'ın final çıktısından

---

## Model Mimarisi ve Parametrizasyon

### Genel Mimari

```
Input Embeddings → Positional Encoding → Encoder Stack → 
Context Representations → Decoder Stack → Linear → Softmax → Output
```

### Embedding Katmanları

Girdi ve çıktı token'ları d_model boyutundaki vektörlere dönüştürülür. Embedding ağırlıkları √d_model ile çarpılır.

### Weight Sharing

Aşağıdaki parametreler paylaşılır:
- Encoder ve decoder embedding katmanları
- Pre-softmax linear transformation

---

## Eğitim Metodolojisi

### Optimizer Konfigürasyonu

Adam optimizer kullanılır:

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0,  # Dinamik learning rate
    betas=(0.9, 0.98),
    eps=1e-9
)
```

### Learning Rate Scheduling

```
lrate = d_model^(-0.5) × min(step_num^(-0.5), step_num × warmup_steps^(-1.5))
```

Bu şema:
- İlk warmup_steps (4000) için doğrusal artış
- Sonrasında step sayısının ters karekökü ile azalış

### Regularization Teknikleri

**Dropout (0.1):**
- Alt-katman çıktılarına
- Embedding ve positional encoding toplamına

**Label Smoothing (0.1):**
- Cross-entropy loss'una uygulanır
- Model calibration'ını iyileştirir

---

## Implementasyon Detayları

### Multi-Head Attention Sınıfı

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        attention_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.W_o(attention_output)
        return output
```

### Training Loop

```python
def train_step(model, src, tgt, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]
    
    tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1))
    
    output = model(src, tgt_input, tgt_mask=tgt_mask)
    loss = criterion(output.reshape(-1, output.size(-1)), 
                     tgt_output.reshape(-1))
    
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

---

## Deneysel Sonuçlar ve Analiz

### Computational Complexity Analizi

| Katman Tipi | Katman Başına Karmaşıklık | Sıralı İşlemler | Maksimum Path Length |
|------------|---------------------------|----------------|---------------------|
| Self-Attention | O(n² · d) | O(1) | O(1) |
| Recurrent | O(n · d²) | O(n) | O(n) |
| Convolutional | O(k · n · d²) | O(1) | O(log_k(n)) |

### Machine Translation Sonuçları

**WMT 2014 English-German:**
- Transformer (big): 28.4 BLEU
- Önceki en iyi sonuç: 26.3 BLEU
- Eğitim maliyeti: Rakiplerin 1/4'ü

**WMT 2014 English-French:**
- Transformer (big): 41.8 BLEU  
- Önceki en iyi sonuç: 41.16 BLEU

### Model Boyutları ve Performans

**Base Model:**
- Parametre sayısı: ~65M
- Eğitim süresi: 12 saat (8 P100 GPU)

**Big Model:**
- Parametre sayısı: ~213M
- Eğitim süresi: 3.5 gün (8 P100 GPU)

### Ablation Studies

Makale çeşitli komponenlerin etkisini analiz eder:
- Attention head sayısının etkisi
- Key ve value boyutlarının etkisi
- Model boyutunun performansa etkisi
- Dropout oranlarının optimizasyonu

---

## Sonuç

Transformer mimarisi, sequence modeling paradigmasında köklü bir değişiklik getirmiş ve şu avantajları sağlamıştır:

**Computational Efficiency:** O(1) sıralı işlem gerekliliği ile mükemmel paralelleştirme

**Long-range Dependencies:** Sabit sayıda işlemle uzun mesafe bağımlılıklarının yakalanması

**Training Speed:** Geleneksel RNN tabanlı modellere kıyasla önemli ölçüde hızlı eğitim

**Performance:** State-of-the-art sonuçlar ve düşük eğitim maliyeti

Bu çalışma, attention mekanizmasının sequence modeling için yeterli olduğunu empirik olarak kanıtlamış ve BERT, GPT serisi gibi sonraki breakthrough'ların temelini oluşturmuştur.

---

## Referanslar

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).