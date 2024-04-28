import os
import csv
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from Bench.dataset.multi_dataset import CapDataset
# from LaMed.src.model.language_model import *
import evaluate

accuracy = evaluate.load("accuracy")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="", choices=[])
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--temperature', type=float, default=1.0)

    # data
    parser.add_argument('--data_root', type=str, default="./Data/data")
    parser.add_argument('--cap_data_path', type=str, default="./Data/data/M3D_Cap_npy/M3D_Cap.json")
    parser.add_argument('--output_dir', type=str, default="./LaMed/output/LaMed-finetune-0000/eval_caption/")

    parser.add_argument('--proj_out_num', type=int, default=256)

    return parser.parse_args(args)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels
        

def main():
    seed_everything(42)
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, model_max_length=args.max_length,
                                                   padding_side="right", use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    test_dataset = CapDataset(args, tokenizer=tokenizer, mode='test') # test1k

    test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=32,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
    )  

    # device = 'cuda' #'cpu', 'cuda'
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_path = os.path.join(args.output_dir, "eval_caption.csv")

    with open(output_path, mode='w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Question", "Ground Truth", "pred", "bleu", "rouge1", "meteor", "bert_f1"])
        with torch.no_grad():
            for sample in tqdm(test_dataloader):
                question = sample["question"]
                answer = sample['answer']

                input_id = tokenizer(question, return_tensors="pt")['input_ids'].to(device)
                image = sample["image"].to(device)

                model.model.seg_enable = False
                generation = model.generate(image, input_id, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams, do_sample=args.do_sample, temperature=args.temperature)
                generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)

                result = dict()
                decoded_preds, decoded_labels = postprocess_text(generated_texts, answer)
                bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=1)
                result["bleu"] = bleu_score['bleu']

                rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=['rouge1'])
                result["rouge1"] = rouge_score['rouge1']

                meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)
                result["meteor"] = meteor_score['meteor']

                bert_score = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
                result["bert_f1"] = sum(bert_score['f1']) / len(bert_score['f1'])

                writer.writerow([question[0], answer[0], generated_texts[0], result["bleu"], result["rouge1"], result["meteor"], result["bert_f1"]])


if __name__ == "__main__":
    main()
       