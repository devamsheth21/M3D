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
import json
from scipy import stats
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Bench.dataset.CTdataset import CapDataset
# If the model is not from huggingface but local, please uncomment and import the model architecture.
# from LaMed.src.model.language_model import *
from evaluatemetrics import bleu_score, rouge_score, meteor_score_fn
import evaluate
from radgraph import F1RadGraph
import yaml
import pandas as pd
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
    parser.add_argument('--genimpression',action='store_true', help = "Generate impressions")
    parser.add_argument('--genfindings',action='store_true', help = "Generate findings")
    parser.add_argument('--model_name_or_path', type=str, default="GoodBaiBai88/M3D-LaMed-Llama-2-7B", choices=[])
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--config_file', type=str,default="/media/Datacenter_storage/Devam/CT/CurateCTdatasets/config.yaml", help="Path to the configuration file")
    parser.add_argument('--question', type=str, default="Please caption this medical scan with impressions.")
    parser.add_argument('--label', type=int, default=None)
    parser.add_argument('--proj_out_num', type=int, default=256)
    parser.add_argument('--version', type=str, required=True, help="Version of the experiment")
    parser.add_argument('--sample',action='store_true', help = "Sample the dataset")
    return parser.parse_args(args)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def calculate_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    margin = sem * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, margin

def setup_logger(version,findings=True):
    log_dir = f"/media/Datacenter_storage/Devam/CT/ReportGen/evalresults/experiments{version}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if findings:
        logging.basicConfig(filename=os.path.join(log_dir, f'experiment{version}_findings.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(filename=os.path.join(log_dir, f'experiment{version}.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(), log_dir

def boot_format(arr):
    mu = np.mean(arr)
    std = np.std(arr)
    return f"{mu:0.3f}({mu-2*std:0.3f},{mu+2*std:0.3f})"

def get_metrics(my_df, dataset_name):
    my_df = my_df[my_df['Dataset Name'] == dataset_name]
    metric_names = my_df.columns[5:]
    metrics = {name: [] for name in metric_names}
    np.random.seed(123)

    for i in range(100):
        sub_pop = np.random.randint(26, 75) / 100
        t_df = my_df.sample(frac=sub_pop, random_state=i)
        for name in metric_names:
            metric_value = t_df[name].mean()  # Example metric calculation
            metrics[name].append(metric_value)

    out_d = {}
    for name, values in metrics.items():
        boot_ci = boot_format(values)
        mean, margin = calculate_confidence_interval(values)
        ci = f"{mean:0.3f}({mean-margin:0.3f},{mean+margin:0.3f})"
        out_d[name] = {'Bootstrap CI': boot_ci, 'Confidence Interval': ci}

    return out_d
def main():
    seed_everything(42)
    args = parse_args()
    device = torch.device(args.device)

    logger,log_dir = setup_logger(args.version,args.genfindings)
    logger.info("Starting evaluation script")

    # Load configuration file
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

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
    if args.genimpression:
        logger.info("Generating impressions")
        args.question = "Please caption this medical scan with impressions."
        output_path = os.path.join(log_dir, "eval_results_impressionall.csv")
        output_txt_path = os.path.join(log_dir, "eval_results_impressionall.txt")
    elif args.genfindings:
        logger.info("Generating findings")
        args.question = "Please caption this medical scan with findings."
        output_path = os.path.join(log_dir, "eval_results_findingsall.csv")
        output_txt_path = os.path.join(log_dir, "eval_results_findingsall.txt")

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    maindf = pd.read_csv('/media/Datacenter_storage/Devam/CT/CurateCTdatasets/All_findings_impressionv2.csv')
    dataset_mapping = {
        'ASCVD_Debiasing': 'ASCVD_Debiasing',
        'BiologicalAge': 'Bioage',
        'ABC_IHD_ramon': 'ABC_IHD_ramon',
        'chestCT_birads': 'ChestCT',
        # 'MRI_rectal_cancer': 'MRI_rectal_cancer',
        'Bladder_project': 'bladder',
        # 'Bhavika_radiogenomics': 'radiogen',
        'CTpancreas': 'CTpancreas',
        'PE_data': 'PE_data'
    }

    maindf['dataset'] = maindf.Image.apply(lambda x: x.split('/')[3])
    maindf['dataset'] = maindf['dataset'].replace(dataset_mapping)
    for dataset_name, dataset_info in config.items():
        print(f"Processing dataset: {dataset_name}")
        logger.info(f"Processing dataset: {dataset_name}")
        dataset_category = dataset_info['category']
        args.cap_data_path = None
        if dataset_name in maindf['dataset'].unique():
            tempdf = maindf[maindf['dataset'] == dataset_name]
            print(f" {dataset_name} dataset size: {len(tempdf)}")
        # args.cap_data_path = os.path.join("/media/Datacenter_storage/Devam/CT/CurateCTdatasets", dataset_name, f"{dataset_name}_merged_all_final_train_test.csv")
        args.df = tempdf
        test_dataset = CapDataset(args, tokenizer=tokenizer, mode='test', logger=logger)
        logger.info(f"Test dataset size: {len(test_dataset)}")
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=16,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        results = []

        for batch in tqdm(test_dataloader):
            questions = batch["question"]
            answers = batch['answer']
            prompt_questions = batch['prompt_question']

            input_ids = tokenizer(questions, return_tensors="pt", padding=True, truncation=True)['input_ids'].to(device=device)
            images = batch["image"].to(device=device, dtype=torch.bfloat16)

            generations = model.generate(images, input_ids, max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, top_p=args.top_p, temperature=args.temperature)
            generated_texts = tokenizer.batch_decode(generations, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(generated_texts, answers)
            bleu_scores = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=1)
            rouge_scores = rouge.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=['rouge1'])
            meteor_scores = meteor.compute(predictions=decoded_preds, references=decoded_labels)
            bert_scores = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
            bert_score_f1 = sum(bert_scores['f1']) / len(bert_scores['f1'])
            f1radgraph_scores = [f1radgraph(hyps=[pred], refs=[label])[0] for pred, label in zip(generated_texts, answers)]

            for i in range(len(questions)):
                results.append({
                    "Dataset Name": dataset_name,
                    "Dataset Category": dataset_category,
                    "Accession": batch["acc"][i],
                    "Ground Truth": answers[i],
                    "pred": generated_texts[i],
                    "bleu": bleu_scores["bleu"],
                    "rouge1": rouge_scores["rouge1"],
                    "meteor": meteor_scores["meteor"],
                    "bert_f1": bert_score_f1,
                    "f1radgraph": f1radgraph_scores[i],
                    "Sample Size": len(test_dataset)
                })

        # Convert results to DataFrame
        df = pd.DataFrame(results)

        # Calculate metrics and confidence intervals
        metrics = ["bleu", "rouge1", "meteor", "bert_f1", "f1radgraph"]
        ci_results = get_metrics(df, dataset_name)
        # Prepare CI results for DataFrame
        ci_data = {
            "Dataset Name": dataset_name,
            "Dataset Category": dataset_category,
            "Accession": "Confidence Interval",
            "Ground Truth": "",
            "pred": "",
            "Sample Size": len(test_dataset)
        }
        for metric in metrics:
            ci_data[metric] = ci_results[metric]['Bootstrap CI']
            # ci_data[metric] = ci_results[metric]['Confidence Interval']
            # ci_data[f"{metric}_bootstrap"] = ci_results[metric]['Bootstrap CI']

        ci_df = pd.DataFrame([ci_data])

        # Append CI results to the main DataFrame
        df = pd.concat([df, ci_df], ignore_index=True)

        # Save DataFrame to CSV, appending if the file exists
        if os.path.exists(output_path):
            df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            df.to_csv(output_path, index=False)

        with open(output_txt_path, mode='a+') as txtfile:
            txtfile.write(f"Results saved to {output_path}\n")
            txtfile.write(f"Dataset Name: {dataset_name}, Dataset Category: {dataset_category}\n and Sample Size: {len(test_dataset)}\n")
            for metric in metrics:
                txtfile.write(f"Parametric CI {metric}: {ci_results[metric]['Confidence Interval']}\n")
                txtfile.write(f"Bootstrap CI {metric}: {ci_results[metric]['Bootstrap CI']}\n")

        logger.info("Evaluation script finished")

if __name__ == "__main__":
    main()
