## deep voice 3 论文学习(基于deep voice 源码)

**1.模型结构**

![deepvoice3](https://github.com/sysu16340234/deep_voice_3_learning/blob/master/deepvoice3.PNG?raw=true)
deep voice 3的模型主要有三个组成部分,**编码器**,**解码器**和**转换器**:

**编码器**:全卷积编码器,可以将文本特征转换为经过学习的内部表示;

**解码器**:全卷积因果解码器,使用多跳的卷积注意力机制将经过学习的文本表示解码为低维音频表示(mel频谱图);

**转换器**:全卷积的后处理网络,从解码器的隐状态中预测声码器参数,是非因果的;

**1.1. 结构细节**

**编码器**

![encoder](https://github.com/sysu16340234/deep_voice_3_learning/blob/master/encoder.PNG?raw=true)


编码器的开始部分是一个嵌入层,用于产生可训练的嵌入向量he,这个嵌入向量是由文本嵌入和说话者嵌入组成的,说话者嵌入经过一个全连接层和softsign函数再与文本嵌入相加:
```python
        # embed text_sequences
        x = self.embed_tokens(text_sequences.long())
        x = F.dropout(x, p=self.dropout, training=self.training)

        # expand speaker embedding for all time steps
        speaker_embed_btc = expand_speaker_embed(x, speaker_embed)
        if speaker_embed_btc is not None:
            speaker_embed_btc = F.dropout(speaker_embed_btc, p=self.dropout, training=self.training)
            x = x + F.softsign(self.speaker_fc1(speaker_embed_btc))

        input_embedding = x
```

然后,这些嵌入经过一个全连接层和一连串的卷积块得到时间依赖的文本信息,在输出时增加一个全连接层进行维度转换,得到的结果就是key,即hk,代码中将前后两个全连接层与所有的卷积块放入一个ModuleList中:
```python
        # １D conv blocks
        for f in self.convolutions:
            x = f(x, speaker_embed_btc) if isinstance(f, Conv1dGLU) else f(x)
            
        keys = x.transpose(1, 2)

```
这部分卷积块的结构如下,这也是文章中使用的所有卷积块的结构:

![conv](https://github.com/sysu16340234/deep_voice_3_learning/blob/master/conv_block.PNG?raw=true)

卷积块的卷积使用的是有padding的卷积来保持序列长度,在因果卷积中(解码器),padding大小为(k - 1)* 时间步长,在非因果卷积中,padding大小为(k - 1)除以两倍时间步长:

```python
        if padding is None:
            # no future time stamps available
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
```

在卷积前使用dropout来进行正则化,卷积后输出分裂成两块,一块与经过线性投影的说话者嵌入相加,另一块经过sigmoid激活,然后将这两块相乘,然后与输入进行残差连接,再乘以一个√0.5的比例因子得到输出:
```python
    def _forward(self, x, speaker_embed, is_incremental):
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        if is_incremental:
            splitdim = -1
            x = self.conv.incremental_forward(x)
        else:
            splitdim = 1
            x = self.conv(x)
            # remove future time steps
            x = x[:, :, :residual.size(-1)] if self.causal else x

        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)
        if self.speaker_proj is not None:
            softsign = F.softsign(self.speaker_proj(speaker_embed))
            # Since conv layer assumes BCT, we need to transpose
            softsign = softsign if is_incremental else softsign.transpose(1, 2)
            a = a + softsign
        x = a * torch.sigmoid(b)
        return (x + residual) * math.sqrt(0.5) if self.residual else x
```

最后,key值经过残差连接再乘上一个√0.5的比例因子就得到了value值,即hv;

**解码器**

![decoder](https://github.com/sysu16340234/deep_voice_3_learning/blob/master/decoder.PNG?raw=true)

decoder通过预测的方式来自回归地生成音频表示(mel频谱图),在解码mel频谱时使用多帧处理,对每一帧进行dropout正则化后与说话者嵌入的线性投影相加,然后经过一层全连接层和ReLU层输出,其中r即为帧数:
```python
        # Prenet: causal convolution blocks
        self.preattention = nn.ModuleList()
        in_channels = in_dim * r
        std_mul = 1.0
        for out_channels, kernel_size, dilation in preattention:
            if in_channels != out_channels:
                # Conv1d + ReLU
                self.preattention.append(
                    Conv1d(in_channels, out_channels, kernel_size=1, padding=0,
                           dilation=1, std_mul=std_mul))
                self.preattention.append(nn.ReLU(inplace=True))
                in_channels = out_channels
                std_mul = 2.0
            self.preattention.append(
                Conv1dGLU(n_speakers, speaker_embed_dim,
                          in_channels, out_channels, kernel_size, causal=True,
                          dilation=dilation, dropout=dropout, std_mul=std_mul,
                          residual=True))
            in_channels = out_channels
            std_mul = 4.0

```
得到prenet输出后再经过一层因果卷积层和一层注意力层,注意力层的结构如下；

![attention](https://github.com/sysu16340234/deep_voice_3_learning/blob/master/attention.PNG?raw=true)

注意力模块使用query向量(解码器隐状态)和来自编码器的key向量来计算注意力权重,然后对解码器的value向量进行加权平均作为输出的上下文向量

在计算权重时时,引入了位置编码,即embed_keys_positions和embed_query_positions(代码中位置编码实际是在decoder的forward中实现的,最后将位置编码的结果输入attention块中作为key和query的输入)：
```python
        # position encodings
        if text_positions is not None:
            w = self.key_position_rate
            # TODO: may be useful to have projection per attention layer
            if self.speaker_proj1 is not None:
                w = w * torch.sigmoid(self.speaker_proj1(speaker_embed)).view(-1)
            text_pos_embed = self.embed_keys_positions(text_positions, w)
            keys = keys + text_pos_embed
        if frame_positions is not None:
            w = self.query_position_rate
            if self.speaker_proj2 is not None:
                w = w * torch.sigmoid(self.speaker_proj2(speaker_embed)).view(-1)
            frame_pos_embed = self.embed_query_positions(frame_positions, w)

```
位置编码的方式是:hp(i) = sin (ωsi/10000k/d) (当i为奇数) or cos (ωsi/10000k/d)(当i为偶数),其中i是时间步数索引,d是总通道数,wsi是位置速率,这两部分的位置编码采用同一种编码方式:

```python
def position_encoding_init(n_position, d_pos_vec, position_rate=1.0,
                           sinusoidal=True):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [position_rate * pos / np.power(10000, 2 * (i // 2) / d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc = torch.from_numpy(position_enc).float()
    if sinusoidal:
        position_enc[1:, 0::2] = torch.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = torch.cos(position_enc[1:, 1::2])  # dim 2i+1

    return position_enc


def sinusoidal_encode(x, w):
    y = w * x
    y[1:, 0::2] = torch.sin(y[1:, 0::2].clone())
    y[1:, 1::2] = torch.cos(y[1:, 1::2].clone())
    return y


class SinusoidalEncoding(nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim,
                 *args, **kwargs):
        super(SinusoidalEncoding, self).__init__(num_embeddings, embedding_dim,
                                                 padding_idx=0,
                                                 *args, **kwargs)
        self.weight.data = position_encoding_init(num_embeddings, embedding_dim,
                                                  position_rate=1.0,
                                                  sinusoidal=False)

    def forward(self, x, w=1.0):
        isscaler = np.isscalar(w)
        assert self.padding_idx is not None

        if isscaler or w.size(0) == 1:
            weight = sinusoidal_encode(self.weight, w)
            return F.embedding(
                x, weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)
        else:
            # TODO: cannot simply apply for batch
            # better to implement efficient function
            pe = []
            for batch_idx, we in enumerate(w):
                weight = sinusoidal_encode(self.weight, we)
                pe.append(F.embedding(
                    x[batch_idx], weight, self.padding_idx, self.max_norm,
                    self.norm_type, self.scale_grad_by_freq, self.sparse))
            pe = torch.stack(pe)
            return pe

```
经过位置编码后的key向量和value向量经过线性投影后进行矩阵乘法,然后经过带有dropout的softmax单调attention模块得到注意力权重,然后与经过线性投影的value值相乘得到上下文向量,然后除以value在时间维度上的平方根,再经过线性投影得到attention块的输出,最后让输出与卷积块的输出进行残差连接并乘以比例因子得到最后的attention上下文:
```python
# attention
        x = self.query_projection(query)
        x = torch.bmm(x, keys)

        mask_value = -float("inf")
        if mask is not None:
            mask = mask.view(query.size(0), 1, -1)
            x.data.masked_fill_(mask, mask_value)

        if last_attended is not None:
            backward = last_attended - self.window_backward
            if backward > 0:
                x[:, :, :backward] = mask_value
            ahead = last_attended + self.window_ahead
            if ahead < x.size(-1):
                x[:, :, ahead:] = mask_value

        # softmax over last dim
        # (B, tgt_len, src_len)
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=1)
        x = x.view(sz)
        attn_scores = x

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = torch.bmm(x, values)

        # scale attention output
        s = values.size(1)
        x = x * (s * math.sqrt(1.0 / s))

        # project back
        x = self.out_projection(x)
        x = (x + residual) * math.sqrt(0.5)
        return x, attn_scores

```
最后,输出的上下文向量经过线性投影和sigmoid激活得到mel频谱和完成标志:
```python
        decoder_states = x.transpose(1, 2).contiguous()
        x = self.last_conv(x)

        # Back to B x T x C
        x = x.transpose(1, 2)

        # project to mel-spectorgram
        outputs = torch.sigmoid(x)

        # Done flag
        done = torch.sigmoid(self.fc(x))

```

**转换器**

音频波形的生成有三种方式:Griffin-Lim算法,WORLD,和WaveNet,其中WaveNet的输入可以直接由解码器的输出mel频谱提供,而转换器的目的是为前两种方法提供输入参数;

首先,转换器的输入是解码器的上下文向量,即decode模块的隐状态,输入经过一系列非因果的卷积块和一个线性投影层,得到转换到对应维度的向量;

对于Griffin-Lim算法,将这个向量进行一层线性投影得到线性频谱,然后将频谱功率经由锐化因子提升,就可以得到Griffin-Lim算法的输入参数;

对于WORLD,预测四个参数:是否有声,基频值,频谱包络和非周期性参数;


