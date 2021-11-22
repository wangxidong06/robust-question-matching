import paddle
import numpy as np
from pypinyin import pinyin, lazy_pinyin, Style



from paddlenlp.datasets import MapDataset
from paddlenlp import Taskflow
text_correction = Taskflow("text_correction")

def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)


def read_text_pair(data_path, is_test=False):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if(len(data[0])==len(data[1])):
                dissame=0
                idx=0
                for i in range (len(data[0])):
                    if(data[0][i]!=data[1][i]):
                        idx=i
                        dissame=dissame+1
                if(dissame==1):
                    char1=data[0][idx]
                    char2=data[1][idx]
                    py1=lazy_pinyin(char1)[0]
                    py2=lazy_pinyin(char2)[0]
                    if py1==py2 and py1!='ta':
                        print("data[0]:{} data[1]:{}".format(data[0],data[1]))
                        data[1]=data[0]
                        print("data[0]:{} data[1]:{}".format(data[0],data[1]))

            if is_test == False:
                if len(data) != 3:
                    continue
                yield {'query1': data[0], 'query2': data[1], 'label': data[2]}
            else:
                if len(data) != 2:
                    continue
                yield {'query1': data[0], 'query2': data[1]}



def convert_example(example, tokenizer, max_seq_length=512, is_test=False):

    query, title = example["query1"], example["query2"]

    encoded_inputs = tokenizer(
        text=query, text_pair=title, max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids