# Myselfie
目前分为两个部分:
- REPE生成所有中间的hidden_states
- Myselfie直接进行翻译
    - 原本的模型微调最后一层lm_head(original)
    - shortcut把第十三层的结果短接过来同时进行翻译(shortcut)
    - combine 把两个tensor拼接到一起进行翻译(combine)