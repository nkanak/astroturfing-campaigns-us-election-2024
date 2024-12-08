import os
import json

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

# finetune, zero-shot and lora finetuning, peft, qlora
# roberta, distilbert, tweetbert

def read_dataset(directory_name: str):
    directory_filenames = os.listdir(directory_name)
    dataset = []
    for fname in directory_filenames:
        fname = directory_name + "/" + fname
        with open(fname, 'r') as f:
            data = json.load(f)
        dataset.append(data)

    return dataset

label2number = {
    "real": 0,
    "fake": 1,
}

train_set = read_dataset("./dataset1/train")
test_set = read_dataset("./dataset1/test")

#train_set[0]["label"]
#train_set[0]["nodes"][0]["tweet_text"]


train_data = [[sample["nodes"][0]["tweet_text"], label2number[sample["label"]]] for sample in train_set]
test_data = [[sample["nodes"][0]["tweet_text"], label2number[sample["label"]]] for sample in test_set]


# Convert the lists to Hugging Face Dataset objects
train_dataset = Dataset.from_dict({
    'text': [item[0] for item in train_data],
    'label': [item[1] for item in train_data]
})

test_dataset = Dataset.from_dict({
    'text': [item[0] for item in test_data],
    'label': [item[1] for item in test_data]
})

# 2. Load the DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# 3. Tokenize the datasets
def tokenize_function(example):
    return tokenizer(example['text'], padding='max_length', truncation=True)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# 4. Load pre-trained DistilBERT for binary classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# 5. Set the device to MPS if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model.to(device)

# 6. Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy='epoch'
)

# 7. Define the evaluation metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=1)
    probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=1)[:, 1]
    
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    roc_auc = roc_auc_score(labels, probs)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

# 8. Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics
)

# 9. Train the model
trainer.train()

# 10. Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)

# 11. Save the fine-tuned model and tokenizer
model.save_pretrained('./distilbert-finetuned-binary')
tokenizer.save_pretrained('./distilbert-finetuned-binary')