import os
import torch
import s3fs
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling
from composer.models.huggingface import HuggingFaceModel
from composer import Trainer
from composer.optim import DecoupledAdamW, LinearWithWarmupScheduler
from composer.metrics.nlp import LanguageCrossEntropy, MaskedAccuracy
from composer.utils import reproducibility


# Setup S3 filesystem client with s3fs
fs = s3fs.S3FileSystem(
    key=os.environ['AWS_ACCESS_KEY_ID'],
    secret=os.environ['AWS_SECRET_ACCESS_KEY'],
    client_kwargs={'endpoint_url': os.environ['S3_ENDPOINT_URL']},
    use_ssl=False  # Minio typically does not use SSL in local setups
)

bucket_name = 'parbucket'
train_object_name = 'parquet_data/train-00000-of-00001.parquet'
eval_object_name = 'parquet_data/test-00000-of-00001.parquet'
text_column_name = 'text'

class StreamingDataset(IterableDataset):
    def __init__(self, s3_path):
        self.fs = s3fs.S3FileSystem(
            key=os.environ['AWS_ACCESS_KEY_ID'],
            secret=os.environ['AWS_SECRET_ACCESS_KEY'],
            client_kwargs={'endpoint_url': os.environ['S3_ENDPOINT_URL']},
            use_ssl=False
        )
        self.parquet_file = pq.ParquetFile(s3_path, filesystem=self.fs)
        self.num_rows = self.parquet_file.metadata.num_rows

    def __iter__(self):
        return self.read_batches()
    
    def __len__(self):
        # Return the total number of rows in the dataset
        return self.num_rows

    def read_batches(self):
        batch_iterator = self.parquet_file.iter_batches()
        for batch in batch_iterator:
            df = batch.to_pandas()
            for i in range(len(df)):
                yield {text_column_name: df.iloc[i][text_column_name]}

# Initialize transformers models and tokenizer
reproducibility.seed_all(17)
config = AutoConfig.from_pretrained('google/electra-small-discriminator')
model = AutoModelForMaskedLM.from_config(config)
tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator')

train_dataset = StreamingDataset(f's3://{bucket_name}/{train_object_name}')
eval_dataset = StreamingDataset(f's3://{bucket_name}/{eval_object_name}')

class CustomCollator:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=mlm_probability)

    def __call__(self, batch):
        # Extract texts from the batch
        texts = [item[text_column_name] for item in batch]

        # Tokenize texts
        tokenized_inputs = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )

        # Convert BatchEncoding to a list of dictionaries suitable for the data collator
        examples = [{key: val[i].clone() for key, val in tokenized_inputs.items()} for i in range(len(texts))]
        
        # Create batch for masked language model training using the data collator
        batch_for_mlm = self.data_collator(examples)
        return batch_for_mlm


collator = CustomCollator(tokenizer=tokenizer)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=64, collate_fn=collator)
eval_dataloader = DataLoader(eval_dataset, batch_size=64, collate_fn=collator)

# for i, batch in enumerate(train_dataloader):
#     print(f"Batch {i} has {batch['input_ids'].shape[0]} samples.")
#     if batch['input_ids'].shape[0] != 64:
#         print(f"Batch size mismatch: {batch['input_ids'].shape[0]}")
#     else:
#         print("Batch size correct.")


# Setup model, optimizer, and metrics
metrics = [
    LanguageCrossEntropy(ignore_index=-100),
    MaskedAccuracy(ignore_index=-100)
]

composer_model = HuggingFaceModel(model, tokenizer=tokenizer, metrics=metrics, use_logits=True)

optimizer = DecoupledAdamW(composer_model.parameters(), lr=1.0e-4, betas=[0.9, 0.98], eps=1.0e-06, weight_decay=1.0e-5)
lr_scheduler = LinearWithWarmupScheduler(t_warmup='250ba', alpha_f=0.02)

trainer = Trainer(
    model=composer_model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_duration='1ep',
    save_folder='checkpoints/pretraining/',
    optimizers=optimizer,
    schedulers=[lr_scheduler],
    precision='fp32',
    seed=17,
)

trainer.fit()
trainer.close()

