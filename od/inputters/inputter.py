# -*- coding: utf-8 -*-
import os
import json

import torch
from torch.utils.data import DataLoader
from transformers import cached_path

from od.inputters.dataset_wb import WBDataset, WBdistDataset

LCCC_URL = "https://coai-dataset.oss-cn-beijing.aliyuncs.com/CleanWB.zip"
SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[speaker1]", "[speaker2]"]


def strs_reshaped(strs):
  return list(map(lambda s: "".join(list(map(lambda c: c + " ", s.replace(" ", ""))))[:-1], strs))


def clean_knowledge(value):
  if "Information" in value:
    value.remove("Information")
  if len(value) > 1 and len(value[1]) < 10:
    value.insert(0, value[0] + " " + value[1])
    del value[1:3]
  return strs_reshaped(value)


from more_itertools import flatten


def reshape(dataset):
  return list(flatten(map(lambda x: list(
    list(map(clean_knowledge, x["knowledge"])) + [strs_reshaped(x["conversation"])] if "conversation" in x else []), dataset)))


def reshape2(dataset):
  return list(flatten(map(lambda x: [[x["history"], x["response"]], x["knowledge"]], dataset)))


def reshape3(dataset):
  return list(flatten(map(lambda x: list(map(clean_knowledge, x["knowledge"])), dataset)))


def get_data(tokenizer, dataset_path, dataset_cache, logger):
    """ Get tokenized dataset from COTK or cache."""
    dataset_path = dataset_path or LCCC_URL
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
        samples = None
    else:
        logger.info("Download dataset from %s", dataset_path)
        # cache_file = cached_path(dataset_path)
        kdconv_train = open("data/input/kdconv/train.txt")
        kdconv_valid = open("data/input/kdconv/dev.txt")
        kd_objs_train = [json.loads(line) for line in kdconv_train.readlines()]
        kd_objs_valid = [json.loads(line) for line in kdconv_valid.readlines()]
        duconv_train = open("data/input/duconv/train.txt")
        duconv_valid = open("data/input/duconv/dev.txt")
        du_objs_train = [json.loads(line) for line in duconv_train.readlines()]
        du_objs_valid = [json.loads(line) for line in duconv_valid.readlines()]
        tencent_train = open("data/input/tencent/train.txt", errors="ignore")
        tencent_valid = open("data/input/tencent/dev.txt", errors="ignore")
        tencent_objs_train = [json.loads(line) for line in tencent_train.readlines()]
        tencent_objs_valid = [json.loads(line) for line in tencent_valid.readlines()]

        dataset = {
            "train": reshape(kd_objs_train),
            "valid": reshape(kd_objs_valid)
        }
        samples = None

        logger.info("Tokenize and encode the dataset")

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)

        dataset = tokenize(dataset)
        torch.save(dataset, dataset_cache)
    return dataset, samples


def build_dataloaders(args, tokenizer, logger):
    logger.info("Build train and validation dataloaders")

    datasets, raw_samples = get_data(tokenizer, args.data_path, args.dataset_cache, logger)
    train_dataset, valid_dataset = WBDataset(datasets["train"], tokenizer), WBDataset(datasets["valid"], tokenizer)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset,
                              sampler=train_sampler,
                              collate_fn=train_dataset.collate,
                              num_workers=args.num_workers,
                              batch_size=args.train_batch_size,
                              shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler,
                              collate_fn=valid_dataset.collate,
                              num_workers=args.num_workers,
                              batch_size=args.valid_batch_size,
                              shuffle=False)

    return train_loader, valid_loader, train_sampler, valid_sampler


def build_dist_loaders(args, tokenizer, logger):
    logger.info("Build train and validation dataloaders")

    train_dataset = WBdistDataset(tokenizer, data_path=args.train_path)
    valid_dataset = WBdistDataset(tokenizer, data_path=args.valid_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset,
                              collate_fn=train_dataset.collate,
                              pin_memory=(args.device == "cuda"),
                              num_workers=args.num_workers,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset,
                              collate_fn=valid_dataset.collate,
                              pin_memory=(args.device == "cuda"),
                              num_workers=args.num_workers,
                              sampler=valid_sampler,
                              batch_size=args.valid_batch_size,
                              shuffle=False)
    return train_loader, valid_loader, train_sampler, valid_sampler
