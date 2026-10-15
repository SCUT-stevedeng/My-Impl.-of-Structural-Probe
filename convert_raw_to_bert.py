import json
from transformers import BertTokenizerFast, BertModel
import torch
from tqdm import tqdm
import numpy as np
import h5py
import argparse

def clean_word(word):
    return word[2:] if word.startswith("##") else word

def append_hidden(ds, tensor):
    t = tensor.cpu().contiguous().half().numpy()
    n = ds.shape[0]
    ds.resize(n+1, axis=0)
    ds[n] = t

def token_word_alignment(word_ids, og_word_ids):
    """
    For a sentence, align its BERT tokens and original words.
    确保CLS和sep已经删除
    return alignment:
    list of (word_id_in_tree, list of token ids)
    """
    if len(set(word_ids))-2 != len(og_word_ids): #减2是不考虑特殊token如[CLS]
        return None #无法对齐
    alignment = [(og_word_ids[i], []) for i in range(len(og_word_ids))]
    for i, word_id in enumerate(word_ids):
        #i是word_ids的索引，word_id+1即对齐上og_word_ids的索引(考虑了word_id 0是[CLS]的)
        if i == 0 or i == len(word_ids)-1:
            continue
        alignment[word_id-1][1].append(i)
    return alignment




if __name__ == "__main__":
    '''
    h5 structure:
    groups of sentence:
        dataset of id_in_tree:(word_num)
        dataset of hidden_states:(word_num, layer_num, hidden_state_num)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="输入json文件路径")
    parser.add_argument("output_path", help="输出BERT隐状态路径") #例如 "./data/bert_last_6_mean_hidden_states.h5"
    args = parser.parse_args()

    output_path = args.output_path
    raw_data = json.load(open(args.input_path))
    sentences = []
    og_sentence_word_ids = [] #list of list word_id in the parse tree [sent1(word_id2......), sent2(word_id1......),...]
    for sent_id, dic in enumerate(raw_data):
        sentences.append((sent_id,"[CLS] " + dic['text'].strip() + " [SEP]"))
        og_sentence_word_ids.append([word_dict['id'] for word_dict in dic['tokens']])
    tokenizer=BertTokenizerFast.from_pretrained("bert-base-cased", do_lower_case=False)
    model=BertModel.from_pretrained("bert-base-cased", output_hidden_states=True)
    model.eval()


    with h5py.File(output_path, "w") as f:
        for index, sent in tqdm(sentences, total=len(sentences), desc="Encoding"):
            encodings = tokenizer(sent, return_tensors="pt", add_special_tokens=False)
            word_ids = encodings.word_ids(batch_index=0)
            alignment = token_word_alignment(word_ids, og_sentence_word_ids[index])
            if alignment is None:
                continue
            sent_group = f.create_group(f"sentence_{index:06d}")
            with torch.no_grad():
                hidden_states = torch.concat(model(**encodings).hidden_states[-6:], dim=0) #形状为(num_layers=6, seq_len, hidden_size)


            all_ids = []
            all_hiddens = []
            for word_id_in_tree, token_ids in alignment:
                word_hidden_states = hidden_states[:, token_ids, :]
                mean_states = torch.mean(word_hidden_states, dim=1)
                all_ids.append(word_id_in_tree)
                all_hiddens.append(mean_states)

            sent_group.create_dataset("id_in_tree", data=np.array(all_ids, dtype=object), dtype=h5py.string_dtype())
            sent_group.create_dataset("hidden_states", data=np.stack(all_hiddens))
            