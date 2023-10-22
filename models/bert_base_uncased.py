from transformers import BertTokenizerFast
from utils import load_text, get_unique_labels, create_dataset, labels_to_ids, create_label_dicts, get_output_dir

from transformers import pipeline
from transformers import BertForTokenClassification, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
import evaluate
import numpy as np

from transformers import DataCollatorForTokenClassification
    

def train_model(splited_data, label_list, model_name = "bert-base-uncased", eval_dataset = None, epoch = 50):
    output_dir = get_output_dir(model_name)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_data = splited_data.map(tokenize_and_align_labels, batched=True)


    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    seqeval = evaluate.load("seqeval")


    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
        
        
    id2label, label2id = create_label_dicts(label_list)

    model = BertForTokenClassification.from_pretrained(
        model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id,
        hidden_dropout_prob=0.5,
        attention_probs_dropout_prob=0.5
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        num_train_epochs=epoch,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    if eval_dataset:
        print("evaluating on eval dataset")
        tokznized_eval = eval_dataset.map(tokenize_and_align_labels, batched=True)
        eval_result = trainer.evaluate(eval_dataset=tokznized_eval)
        print(eval_result)
        with open(f"{output_dir}/eval_result.txt", "w") as f:
            f.write(str(eval_result))
