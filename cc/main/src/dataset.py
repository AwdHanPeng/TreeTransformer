#!/usr/bin/env python3
# the code is partly borrowed from Code Prediction By Feeding Trees To Transformers 
# https://arxiv.org/abs/2003.13848

import logging
import warnings
import os
import pickle
import torch
import json
import numpy as np
from src.base_dataset import BaseDataset, BaseSetup
from src.utils.tree_utils import generate_positions
from src.utils import utils
from src.utils.constants import UNK, PAD, EMPTY
import random
from collections import Counter

logging.basicConfig(level=logging.INFO)


class Setup(BaseSetup):
    def _add_extra_filepaths(self, base_dir):
        if self.use_tree_att:
            self.filepaths["rel_vocab"] = os.path.join(base_dir, "rel_vocab.pkl")  # tree relative attention vocab
            self.filepaths["tree_rel"] = os.path.join(base_dir, "tree_rel.txt")  # tree relative attention
            self.filepaths["tree_rel_conv"] = os.path.join(base_dir,
                                                           f"tree_rel_conv.vocab={self.tree_rel_max_vocab}.txt")
        if self.root_paths:
            self.filepaths["root_paths"] = os.path.join(base_dir, "paths.txt")  # tree positional encodings

        # HERE
        if self.use_td_tree_pos_enc:
            self.filepaths["td_tree"] = os.path.join(base_dir, "td_paths.txt")
        if self.local_relation:
            self.filepaths["local_relation"] = os.path.join(base_dir, "lr.txt")

    def _create_vocab(self):
        vocabs = {
            "dp": TypesValuesVocab(self.filepaths["vocab_types"], self.filepaths["vocab_values"], self.use_anonymized,
                                   self.anon_vocab, self.max_values_vocab),
            "tree_rel": TreeRelVocab(self.filepaths["rel_vocab"],
                                     self.tree_rel_max_vocab) if self.use_tree_att else None
        }
        return vocabs

    def _create_dataset(self, fp):
        file_paths = {"dps": fp}
        if self.root_paths:
            file_paths.update({"root_paths": self.filepaths["root_paths"]})
        if self.use_tree_att:
            file_paths.update({"rel_mask": self.filepaths["tree_rel_conv"]})

        # HERE
        if self.use_td_tree_pos_enc:
            file_paths.update({"td_tree": self.filepaths["td_tree"]})
        if self.local_relation:
            file_paths.update({"local_relation": self.filepaths["local_relation"]})

        return Dataset(file_paths)


def get_elem2id(unique_items, anon_vocab_size, randomize=False):
    """
        creates anonymization mapping
        randomize: if do permutation of inputs
    """
    assert len(unique_items) <= anon_vocab_size, \
        "anon vocab size should be larger than maximum possible number of unique tokens"
    mapping = list(range(anon_vocab_size))
    if randomize:
        random.shuffle(mapping)
    elem2id = {elem: i for elem, i in zip(unique_items, mapping)}
    return elem2id


class DPVocab(object):
    def __init__(self, vocab_fp, use_anonymized=False, anon_vocab=0, max_values_vocab=100003):
        super().__init__()
        self.unk_token = UNK
        self.pad_token = PAD
        self.empty_token = EMPTY
        self.pad_idx = None
        self.unk_idx = None
        self.empty_idx = None
        self.use_anonymized = use_anonymized
        self.anon_vocab = anon_vocab
        if not use_anonymized:
            self.anon_vocab = 0

        if not os.path.exists(vocab_fp):
            raise Exception("Get the vocab from generate_vocab.py")

        # regular vocab
        with open(vocab_fp, "rb") as fin:
            self.idx2vocab = pickle.load(fin)
        if max_values_vocab >= 0:
            self.idx2vocab = self.idx2vocab[:min(max_values_vocab, len(self.idx2vocab))]
        logging.info("Loaded vocab from: {}".format(vocab_fp))
        self.vocab2idx = {token: i for i, token in enumerate(self.idx2vocab)}
        self.unk_idx = self.vocab2idx[self.unk_token]
        self.pad_idx = self.vocab2idx[self.pad_token]
        self.empty_idx = self.vocab2idx[self.empty_token]
        logging.info("Vocab size: {}".format(len(self.idx2vocab)))

    def convert(self, dp):
        # converts dataset to idx
        if self.use_anonymized:
            unique = []
            was = set()
            for token in dp:
                if not token in self.vocab2idx:
                    if not token in was:
                        unique.append(token)
                        was.add(token)
            elem2id = get_elem2id(unique, self.anon_vocab)

        dp_converted = []
        for token in dp:
            if token in self.vocab2idx:
                dp_converted.append(self.vocab2idx[token])  # in vocab
            else:
                if self.use_anonymized:
                    # we do not have OOV anymore
                    dp_converted.append(len(self.idx2vocab) + elem2id[token])
                else:
                    dp_converted.append(self.unk_idx)
        return dp_converted

    def __len__(self):
        return len(self.idx2vocab) + self.anon_vocab


class TypesValuesVocab(object):
    def __init__(self, types_fp=None, values_fp=None, use_anonymized=False, anon_vocab=0, max_values_vocab=100003):
        self.values_vocab = DPVocab(values_fp,
                                    use_anonymized=use_anonymized, anon_vocab=anon_vocab,
                                    max_values_vocab=max_values_vocab)
        if types_fp is not None:
            self.types_vocab = DPVocab(types_fp)
            assert self.values_vocab.unk_idx == self.types_vocab.unk_idx
            assert self.values_vocab.pad_idx == self.types_vocab.pad_idx
            assert self.values_vocab.empty_idx == self.types_vocab.empty_idx
        else:
            self.types_vocab = None

    @property
    def unk_idx(self):
        return self.values_vocab.unk_idx

    @property
    def pad_idx(self):
        return self.values_vocab.pad_idx

    @property
    def empty_idx(self):
        return self.values_vocab.empty_idx

    def convert(self, dp):
        (types, values), ext = dp
        if self.types_vocab is not None:
            types_converted = self.types_vocab.convert(types)
        else:
            types_converted = []
        values_converted = self.values_vocab.convert(values)
        dp_converted = (types_converted, values_converted)
        return dp_converted, ext

    def __len__(self):
        return len(self.values_vocab)


def mask_first_anon(dp, unk, pad, empty, anon_starts_token=0):
    """
        anon_starts_token: 0 in case full anon, >0 if full + anon for UNK
    """
    # mask first encounter of each anonimized variable since we do not know what it is
    was = set()
    new_dp = []
    for it in dp:
        if not it in was and not it in {unk, pad, empty} and it >= anon_starts_token:
            new_dp.append(unk)
        else:
            new_dp.append(it)
        was.add(it)
    return new_dp


class TreeRelVocab(object):
    def __init__(self, rel_vocab_fp=None, tree_rel_max_vocab=10405):
        # open rel vocab
        with open(rel_vocab_fp, "rb") as fin:
            self.idx2rel = pickle.load(fin)
            if tree_rel_max_vocab < len(self.idx2rel):
                self.idx2rel = self.idx2rel[:tree_rel_max_vocab]  # crop vocab
        logging.info("Loaded rel vocab from: {}".format(rel_vocab_fp))
        self.rel2idx = {token: i for i, token in enumerate(self.idx2rel)}
        self.rel_unk_idx = self.rel2idx[UNK]
        logging.info("Rel vocab sizes: {}".format(len(self.idx2rel)))

    def convert(self, rel_info):
        rel_converted = [
            [
                self.rel2idx[token] if token in self.rel2idx else self.rel_unk_idx
                for token in rel.split()
            ]
            for rel in rel_info
        ]
        return rel_converted


class Dataset(BaseDataset):
    @staticmethod
    def collate(seqs, values_vocab, args):
        pad_idx = values_vocab.pad_idx
        max_len = max(len(dp["dps"][0][1]) for dp in seqs)
        max_len = max(max_len, 2)
        input_types, input_values = [], []
        target_types, target_values = [], []
        extended = []
        position_seqs = []
        rel_mask = torch.zeros((len(seqs), max_len - 1, max_len - 1)).long() if args.use_tree else []

        # here
        code_td_paths_rep = torch.zeros(len(seqs), max_len - 1, args.td_max_depth, 2, dtype=torch.int)
        code_local_relations_rep = torch.zeros(len(seqs), max_len - 1, max_len - 1, 3, dtype=torch.long)

        for i, dp in enumerate(seqs):
            ((types, values), ext) = dp["dps"]
            if len(values) < 2:
                warnings.warn("got len(values) < 2. skip.")
                continue
            assert len(types) == len(values) or len(types) == 0, (types, values)
            # ids.append(dp["ids"])
            padding = [pad_idx] * (max_len - len(values))
            if not args.only_values:
                assert len(types) == len(values)
                input_types.append(types[:-1] + padding)
                target_types.append(types[1:] + padding)
            input_values.append(values[:-1] + padding)
            if args.base_dir.endswith("struct") or values_vocab.use_anonymized:
                # anon setup
                # for anonimization of UNKs:
                anon_starts_token = 0 if not values_vocab.use_anonymized else len(values_vocab.idx2vocab)
                values = mask_first_anon(values, values_vocab.unk_idx, values_vocab.pad_idx, values_vocab.empty_idx,
                                         anon_starts_token)
            target_values.append(values[1:] + padding)
            extended.append(ext)
            if "rel_mask" in dp:
                mask = dp["rel_mask"]
                assert (len(mask) == len(values) - 1), (len(mask), len(values) - 1)
                # tree relative attention
                for j, rel in enumerate(mask):
                    rel_mask[i][j][: len(rel)] = torch.tensor(rel)
            if "root_paths" in dp:
                # tree positional encodings
                root_paths = dp["root_paths"]
                assert len(root_paths) == len(values)
                root_paths = root_paths[:-1] + [[] for _ in range(max_len - len(values))]
                positions = generate_positions(root_paths, max_width=args.max_width, max_depth=args.max_depth)
                position_seqs.append(positions.unsqueeze(0))

            # HERE

            if "td_tree" in dp:
                td_tree = dp["td_tree"]
                assert len(td_tree) == 2

                # json.dump([child_ast, father_ast],
                #          fp=fout)  # each line is the json of a list [[[1],[1,2,3,4]],[[1],[1,2,3,4]]]
                # # list[0] = childs = [[],[1],[1,2],[1,2,3]]
                # # list[1] = fathers = [[],[1],[1,2],[1,2,3]]
                # => [[],[[1,1]],[[1,1],[2,2]],[[1,1],[2,2],[3,3]]]
                # 1 pre 2pad
                child_ast, father_ast = td_tree[0], td_tree[1]
                assert len(child_ast) == len(father_ast) == len(values)
                tds = []
                for one_elem, two_elem in zip(child_ast, father_ast):
                    assert len(one_elem) == len(two_elem)
                    if len(one_elem) == 0:
                        tds.append([])
                    else:
                        temp = []
                        for one, two in zip(one_elem, two_elem):
                            temp.append([one, two])
                        tds.append(temp)
                tds = generate_TD_position(tds, args.td_max_ary, args.td_max_depth)[:-1]
                # [len(root_paths), max_depth, 2]
                code_td_paths_rep[i, :tds.shape[0]].copy_(tds)

            if "local_relation" in dp:
                lr = dp['local_relation']
                lr = generate_local_relations(lr, args.td_max_ary)[:-1, :-1, ]
                # [len(root_paths) x len(root_paths) x 3]
                code_local_relations_rep[i, :lr.shape[0], :lr.shape[0]].copy_(lr)
                # each line is the json of a list [dict,dict,...]

        positions = torch.cat(position_seqs, dim=0) if len(position_seqs) > 0 else []
        return {
            "input_seq": {
                "types": torch.tensor(input_types),
                "values": torch.tensor(input_values)
            },
            "target_seq": {
                "types": torch.tensor(target_types),
                "values": torch.tensor(target_values)
            },
            "extended": torch.tensor(extended),
            "rel_mask": rel_mask,
            # "ids": ids,
            "positions": positions,
            "tds": code_td_paths_rep,
            "lr": code_local_relations_rep
        }


def generate_TD_position(td_paths, max_ary, max_depth):
    '''
    td_paths: len,depth,2
    the root is [], need converted first
    first dim range 0-max_width
    second dim range 1-max_width+1
    returns: tensor: [len(root_paths), max_depth, 2]
    # [ [], [[1,1]], [[1,1],[2,2]] , [[3,3],[2,2],[3,3]] ]

    '''

    for td_path in td_paths:
        td_path.insert(0, [0, 1])
    # add virtual node

    # [ [[1, 1]], [[1, 1],[1,1]], [[1, 1],[1,1],[2,2]] , [[1, 1],[3,3],[2,2],[3,3]] ]

    overflow_flag = max_ary

    # print(td_paths)

    def abs(length):
        return length if length >= 0 else 0

    def convert(x):
        # +1 for the overflow flag
        if x >= max_ary:
            return overflow_flag
        else:
            return x

    sample_idx = []

    for td_path in td_paths:
        tds = [[convert(x + 1), convert(y)] for x, y in td_path]
        sample_idx.append(tds[:max_depth] + [[0, 0]]
                          * abs(max_depth - len(tds)))
    return torch.tensor(sample_idx)


def move_to_device(batch, device):
    for key in batch:
        if batch[key] is not None and isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)


def generate_local_relations(relations, max_ary):
    '''
    relations:[{"0':[1,2,1]."1":[2,3,0]}]
    return tensor:[len(root_paths) x len(root_paths) x 3]
    '''
    overflow_flag = max_ary

    def convert(x):
        # +1 for the overflow flag
        if x >= max_ary:
            return overflow_flag
        else:
            return x

    totals = len(relations)
    result = torch.zeros(totals, totals, 3)
    for i, relation in enumerate(relations):
        for (key, val) in relation.items():
            new_val = [convert(val[0] + 1), convert(val[1]), val[-1]]
            result[i, int(key)] = torch.tensor(new_val)
    return result
