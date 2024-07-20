# Myselfie
目前来看是无法进行翻译的,因为本质上来可能就是原本的tensor上有一个特别小的偏移,所以无法实现.
目前分为两个部分:
- REPE生成所有中间的hidden_states
- Myselfie直接进行翻译
    - 原本的模型微调最后一层lm_head(original)
    - shortcut把第十三层的结果短接过来同时进行翻译(shortcut)
    - combine 把两个tensor拼接到一起进行翻译(combine)

# Adversarial Training
- 目前来看比较现实的做法就是直接通过加random noise来实现更robust的model
- 如果不把最后一层lm_head进行冻结的话会出现各种各样的问题,比如说无论如何训练都是失败的
目前已经进行的尝试
- Advertrain_model_layer 16 
    - /root/autodl-tmp/finetune/adver_train_model_alpaca_layer16/checkpoint-100000_real16_0.75
    - 单独训练 layer 16 上面添加noise 0.75
    - mt_bench 6.4875
- Advertrain_model_layer 16 20 
    - /root/autodl-tmp/finetune/adver_train_model_alpaca_layer1620
    - 训练 layer 16 20 上面添加noise 0.60
    - mt_bench 6.500
- Advertrain_model_layer 26 28 0.6
    - /root/autodl-tmp/finetune/adver_train_model_alpaca_layer16/checkpoint-100000_2628
    - 训练 layer 26 28 添加noise 0.60
    - mt_bench 6.55
    - 没有什么卵用

- Advertrain_model_layer 26 28 1.4
    - /root/autodl-tmp/finetune/adver_train_model_alpaca_layer2628_rand1.4
    - 训练 layer 26 28 添加noise 1.4
    - mt_bench 未知

- Advertrain_model_layer 26 28 2.0
    - /root/autodl-tmp/finetune/adver_train_model_alpaca_layer2628_rand2.0/checkpoint-100000
    - 训练 layer 26 28 添加noise 2.0
    - mt_bench 未知

- Advertrain_model_multi_layer 16 18 20 22 24 26 28 
    - /root/autodl-tmp/finetune/adver_train_model_alpaca_more_layers/checkpoint-50000
    - layer 16 18 20 22 24 26 28  不同的noise level
    