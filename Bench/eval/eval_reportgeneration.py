import os
import sys
import csv
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Bench.dataset.CTdataset import CapDataset
# If the model is not from huggingface but local, please uncomment and import the model architecture.
# from LaMed.src.model.language_model import *
from evaluatemetrics import bleu_score, rouge_score, meteor_score_fn
import evaluate
from radgraph import F1RadGraph
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# accuracy = evaluate.load("accuracy")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")
f1radgraph = F1RadGraph(reward_level="partial", reward_type="f1")

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
    parser.add_argument('--model_name_or_path', type=str, default="GoodBaiBai88/M3D-LaMed-Llama-2-7B", choices=[])
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--device', type=str, default="cuda")

    # data
    # parser.add_argument('--data_root', type=str, default="./Data/data")
    # parser.add_argument('--cap_data_path', type=str, default="./Data/data/M3D_Cap_npy/M3D_Cap.json")
    parser.add_argument('--dataset_name', type=str,default='all',required=True, help="Name of the dataset")
    parser.add_argument('--cap_data_path', type=str, default="/media/Datacenter_storage/Devam/CT/ReportGen/csvfiles/All_findings_impressions.csv")
    # parser.add_argument('--output_dir', type=str, default="/media/Datacenter_storage/Devam/CT/ReportGen/evalresults/datasetname.csv")

    parser.add_argument('--proj_out_num', type=int, default=256)
    parser.add_argument('--question', type=str, default="Please caption this medical scan with impressions.")
    parser.add_argument('--label', type=int, default=None)
    return parser.parse_args(args)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels
        

def main():
    seed_everything(42)
    args = parse_args()
    device = torch.device(args.device)

    # Construct paths based on dataset name
    args.cap_data_path = f"/media/Datacenter_storage/Devam/CT/CurateCTdatasets/{args.dataset_name}/{args.dataset_name}_merged_all_final_train_test.csv"
    args.output_dir = f"/media/Datacenter_storage/Devam/CT/ReportGen/evalresults/{args.dataset_name}_eval_results.csv"
    args.output_txt = f"/media/Datacenter_storage/Devam/CT/ReportGen/evalresults/{args.dataset_name}_eval_results.txt"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map='cpu',
        trust_remote_code=True
    )
    torch.cuda.empty_cache()
    model = model.to(device=device)

    test_dataset = CapDataset(args, tokenizer=tokenizer, mode='test') # test1k
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Test dataset: {test_dataset.df}")
    test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
    )  

    if not os.path.exists(os.path.dirname(args.output_dir)):
        os.makedirs(os.path.dirname(args.output_dir))
    output_path = args.output_dir
    output_txt_path = args.output_txt
    # output_path = os.path.join(args.output_dir, "eval_report1.csv")

    with open(output_path, mode='w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Accession","Question", "Ground Truth", "pred", "bleu", "rouge1", "meteor", "bert_f1","f1radgraph"])
        for sample in tqdm(test_dataloader):
            question = sample["question"]
            answer = sample['answer']
            prompt_question = sample['prompt_question']

            input_id = tokenizer(question, return_tensors="pt")['input_ids'].to(device=device)
            image = sample["image"].to(device=device,dtype=torch.bfloat16)

            generation = model.generate(image, input_id, max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, top_p=args.top_p, temperature=args.temperature)
            generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
            result = {
                "bleu": [],
                "rouge1": [],
                "meteor": [],
                "bert_f1": [],
                "f1radgraph_scores": []
            }

            decoded_preds, decoded_labels = postprocess_text(generated_texts, answer)
            bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=1)
            result["bleu"].append(bleu_score['bleu'])

            rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=['rouge1'])
            result["rouge1"].append(rouge_score['rouge1'])

            meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)
            result["meteor"].append(meteor_score['meteor'])

            bert_score = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
            bert_score_f1 = sum(bert_score['f1']) / len(bert_score['f1'])
            result["bert_f1"].append(bert_score_f1)

                    
            f1radgraphscore, _, _, _ = f1radgraph(hyps=[generated_texts[0]], refs=[answer[0]])
            result["f1radgraph_scores"].append(f1radgraphscore)
            
            writer.writerow([sample["acc"][0],prompt_question[0], answer[0], generated_texts[0], bleu_score["bleu"], rouge_score["rouge1"], meteor_score["meteor"], bert_score_f1, f1radgraphscore])
        writer.writerow(["Average", "", "", "", np.mean(result["bleu"]), np.mean(result["rouge1"]), np.mean(result["meteor"]), np.mean(result["bert_f1"]), np.mean(result["f1radgraph_scores"])])

    with open(output_txt_path, mode='w') as txtfile:
        txtfile.write(f"Score bleu: {np.mean(result['bleu'])}\n")
        txtfile.write(f"Score rouge1: {np.mean(result['rouge1'])}\n")
        txtfile.write(f"Score meteor: {np.mean(result['meteor'])}\n")
        txtfile.write(f"Score bert_f1: {np.mean(result['bert_f1'])}\n")
        txtfile.write(f"Score f1radgraph: {np.mean(result['f1radgraph_scores'])}\n")

if __name__ == "__main__":
    main()
