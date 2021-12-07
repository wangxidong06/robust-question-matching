


```python
!pip install --upgrade paddlenlp
```

```python
!unzip -o data/data104940/train.zip -d data
```


```python
# 将LCQMC、BQ、OPPO三个数据集的训练集和验证集合并
!cat ./data/train/LCQMC/train ./data/train/BQ/train ./data/train/OPPO/train > train.txt
!cat ./data/train/LCQMC/dev ./data/train/BQ/dev ./data/train/OPPO/dev > dev.txt
```


```python
# 加载环境依赖
!pip install pypinyin 
!pip install pypinyin
!pip install ddparser
!pip install LAC
```

```python
# 验证预加载模型效果
from pypinyin import pinyin, lazy_pinyin, Style
lazy_pinyin("词语")
```


```python
# 在最优模型上继续训练
!$unset CUDA_VISIBLE_DEVICES
!python -u  work/train.py \
       --train_set train.txt \
       --dev_set dev.txt \
       --device gpu \
       --eval_step 100 \
       --save_dir ./check \
       --train_batch_size 32 \
       --learning_rate 2E-5 \
       --rdrop_coef 0.0 \
       --init_from_ckpt "./check/model_31200/model_state.pdparams"

```


```python
# 利用最优模型进行预测
!$ unset CUDA_VISIBLE_DEVICES
!python -u \
    work/predict.py \
    --device gpu \
    --params_path "./check/model_33500/model_state.pdparams" \
    --batch_size 128 \
    --input_file "test_B_1118.tsv" \
    --result_file "ccf_result_2021_11_22_10_38.csv"
```









