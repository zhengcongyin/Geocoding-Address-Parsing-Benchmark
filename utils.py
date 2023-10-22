import numpy as np
from transformers import pipeline
from tqdm import tqdm
from datasets import DatasetDict


def load_text(filename):
    lines = []
    with open(filename, 'r') as file:
        lines = file.readlines()

    sentences = []
    labels = []
    current_sentence = []
    current_labels = []

    for line in lines:
        if line.strip() == "-END- -X- -X-":
            sentences.append(current_sentence)
            labels.append(current_labels)
            current_sentence = []
            current_labels = []
        else:
            parts = line.strip().split()
            if len(parts) > 1:
                current_sentence.append(parts[0])
                current_labels.append(parts[-1])

    return sentences, labels


def get_unique_labels(labels_list):
    unique_labels = set()
    for labels in labels_list:
        for label in labels:
            unique_labels.add(label)
    return list(unique_labels)


def create_dataset(data_filenames):
    train_sentences, labels = load_text(data_filenames['train'])
    label_list = get_unique_labels(labels)
    train_label_ids = [labels_to_ids(x, label_list) for x in labels]
    train_data = {"tokens": train_sentences, "ner_tags": train_label_ids}

    test_sentences, labels = load_text(data_filenames['test'])
    test_label_ids = [labels_to_ids(x, label_list) for x in labels]
    test_data = {"tokens": test_sentences, "ner_tags": test_label_ids}

    eval_sentences, labels = load_text(data_filenames['eval'])
    eval_label_ids = [labels_to_ids(x, label_list) for x in labels]
    eval_data = {"tokens": eval_sentences, "ner_tags": eval_label_ids}

    from datasets import Dataset

    train_test = DatasetDict(
        {
            "train": Dataset.from_dict(train_data),
            "test": Dataset.from_dict(test_data)
        }
    )
    eval_dataset = Dataset.from_dict(eval_data)

    return train_test, eval_dataset, label_list


def create_label_mapping(label_list):
    return {label: idx for idx, label in enumerate(label_list)}


def one_hot_encode_label(label, label_mapping):
    one_hot_vector = np.zeros(len(label_mapping), dtype=int)
    one_hot_vector[label_mapping[label]] = 1
    return one_hot_vector


def labels_to_ids(labels, label_list):
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    return [label_to_id[label] for label in labels]


def create_label_dicts(label_list):
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in id2label.items()}
    return id2label, label2id


def predict(text='1701 N 23rd St Canyon texas 79015', model_path="./locationtokenizer/checkpoint-4900",
            eval_dataset=None):
    classifier = pipeline("ner", model=model_path, device=0)
    if eval_dataset is None:
        return classifier(text)
    else:
        result = []
        for example in tqdm(eval_dataset["tokens"]):
            text = " ".join(example)
            result.append(classifier(text))
        return result


def postprocess(text):
    return text.replace("#", " ").strip()


def combine_labels(labels):
    result = []
    i = 0
    while i < len(labels):
        label = labels[i]

        if label.startswith("B-"):
            result.append(label[2:])
            i += 1
            while i < len(labels) and labels[i].startswith("I-") and label[2:] == labels[i][2:]:
                i += 1
        else:
            # If for some reason there are standalone I- labels without a preceding B-
            result.append(label[2:])
            i += 1
    return result


def combine_tokenized_text(pred, tokenized):
    combined_tags = []
    combined_text = []
    current_tag = None
    current_text = ""

    for p, t in zip(pred, tokenized):
        if p.startswith("B-"):
            if current_tag == p:
                current_text += t
            else:
                if current_tag is not None:
                    combined_tags.append(current_tag[2:])
                    combined_text.append(current_text)
                current_tag = p
                current_text = t
        elif p.startswith("I-"):
            current_text += ' '
            current_text += t
        else:
            raise ValueError(f"Invalid tag: {p}")

    if current_tag is not None:
        combined_tags.append(current_tag[2:])
        combined_text.append(current_text)

    return combined_tags, combined_text


def combine_text_from_index(p):
    combined_tags = []
    current_text = ""
    current_start = 0
    for x in p:
        word = postprocess(x['word']);
        if current_start == x['start']:
            if x['start'] == 0:
                combined_tags.append(x['entity'])
            current_text += word.upper()
        if current_start + 1 == x['start']:
            current_text += '##'
            current_text += word.upper()
            combined_tags.append(x['entity'])
        current_start = x['end'];
    return combined_tags, current_text.split('##')


def get_output_dir(s):
    return s + "-ckpt"