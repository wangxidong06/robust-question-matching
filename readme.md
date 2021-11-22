请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 


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
!pip install pypinyin 
```


```python
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
!$ unset CUDA_VISIBLE_DEVICES
!python -u \
    work/predict.py \
    --device gpu \
    --params_path "./check/model_33500/model_state.pdparams" \
    --batch_size 128 \
    --input_file "test_B_1118.tsv" \
    --result_file "ccf_result_2021_11_22_10_38.csv"
```


```python
from ddparser import DDParser
ddp = DDParser()
ddp = DDParser()
words1= ddp.parse("被蜜蜂蛰了怎么处理")[0]['word']
words2= ddp.parse("密蜂蛰了怎么处理")[0]['word']

print(words1)
print(words2)

```


```python
import synonyms
L1=synonyms.nearby('功效', 16)[0] 
print(L1)
L1=[char for str1 in L1 for char in str1 ]
print(L1)
```


```python
from pypinyin import pinyin, lazy_pinyin, Style
lazy_pinyin("词语")
```


```python
! pip install pypinyin
! pip install ddparser
! pip install LAC

```
