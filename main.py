import argparse

from utils import load_text, get_unique_labels, create_dataset, labels_to_ids
import importlib
from main_utils import run_predict
import os
import wandb
os.environ["WANDB_PROJECT"] = "parse_benchmark"

def main(mode):
    data_filenames = {
        "train": "benchmark_dataset/train.txt",
        "test": "benchmark_dataset/dev.txt",
        "eval": "benchmark_dataset/test.txt",
    }
    eval_sentences, eval_labels = load_text(data_filenames['eval'])

    splited_data, eval_dataset, label_list = create_dataset(data_filenames)

    list_of_models = [
        "distilbert-base-uncased",
        "roberta-base",
        "bert-base-uncased",
        "chatgpt"
    ]

    for model_name in list_of_models:
        wandb.init(project="parse_benchmark", name=model_name, config={"model_name": model_name})
        # import modules dynamically
        module_name = model_name.replace('-', '_')

        if mode == "train" and module_name != "chatgpt":
            print("Start training")
            model_module = importlib.import_module(f"models.{module_name}")
            train_model = getattr(model_module, "train_model")
            train_model(splited_data, label_list, model_name = model_name, eval_dataset = eval_dataset, epoch = 25)
        elif mode == "predict":
            print("Start predicting")
            run_predict(eval_sentences, eval_labels, model_name, eval_dataset, label_list, verbose = True)
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "predict"], help="Choose between 'train' or 'predict'")
    args = parser.parse_args()
    main(args.mode)
