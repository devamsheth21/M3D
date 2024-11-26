import os
import torch
from transformers import Trainer
from transformers.utils import logging, SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from typing import Optional
from radgraph import F1RadGraph

logger = logging.get_logger(__name__)
TRAINING_ARGS_NAME = "training_args.bin"

def f1Radgraph_loss(logits, labels):
    # Compute the f1Radgraph loss
    f1radgraph = F1RadGraph(reward_level="partial", reward_type="f1")
    score = f1radgraph_scores = [f1radgraph(hyps=[pred], refs=[label])[0] for pred, label in zip(generated_texts, answers)]
    loss = 0
    return loss


class LaMedTrainer(Trainer):
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     outputs = model(**inputs)
    #     logits = outputs.get("logits")
    #     labels = inputs.get("labels")
    #     # Compute the f1Radgraph loss
    #     loss = f1Radgraph_loss(logits, labels)
    #     return (loss, outputs) if return_outputs else loss    

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if state_dict is None:
            state_dict = self.model.state_dict()

        logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))