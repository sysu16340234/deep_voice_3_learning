## deep voice 3 论文学习

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

在卷积前使用dropout来进行正则化,卷积后输出分裂成两块,一块与经过线性投影的说话者嵌入相加,另一块经过sigmoid激活,然后将这两块相乘(点乘),然后与输入进行残差连接,再乘以一个√0.5的比例因子得到输出:
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

**2.decoder**

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

