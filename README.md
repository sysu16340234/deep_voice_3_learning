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

然后,这些嵌入经过一个全连接层和一连串的卷积块得到时间依赖的文本信息:
```python
        # １D conv blocks
        for f in self.convolutions:
            x = f(x, speaker_embed_btc) if isinstance(f, Conv1dGLU) else f(x)

```
