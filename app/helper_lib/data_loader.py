import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np

def get_data_loader(data_dir, batch_size=32, train=True, dataset_name='CIFAR10', max_length=512):
    """
    Get data loader for different datasets
    
    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size
        train: Whether to load training or test set
        dataset_name: 'CIFAR10' (32x32) or 'CIFAR10_64' (64x64) or 'FashionMNIST' or 'SQUAD'
        max_length: Maximum sequence length for text datasets
    """
    if dataset_name == 'FashionMNIST':
        # Preprocessing for Fashion-MNIST (pad to 32x32)
        def preprocess(img):
            img = np.pad(img, ((2, 2), (2, 2)), constant_values=0.0)
            return img
        
        transform = transforms.Compose([
            transforms.Lambda(preprocess), 
            transforms.ToTensor()
        ])
        
        dataset = datasets.FashionMNIST(
            root=data_dir,
            train=train,
            download=True,
            transform=transform
        )
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train
        )
    
    elif dataset_name == 'CIFAR10':
        # Standard CIFAR-10 (32x32)
        transform = transforms.Compose([transforms.ToTensor()])
        
        dataset = datasets.CIFAR10(
            root=data_dir,
            train=train,
            download=True,
            transform=transform
        )
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train
        )
    
    elif dataset_name == 'CIFAR10_64':
        # CIFAR-10 resized to 64x64 for Assignment 2
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        
        dataset = datasets.CIFAR10(
            root=data_dir,
            train=train,
            download=True,
            transform=transform
        )
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train
        )
    
    elif dataset_name == 'SQUAD':
        # SQuAD dataset for question answering
        split = 'train' if train else 'validation'
        squad_dataset = SQuADQADataset(split=split, max_length=max_length)
        
        # Create collator with the tokenizer's pad token
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        collator = DataCollator(pad_token_id=tokenizer.pad_token_id)
        
        loader = DataLoader(
            squad_dataset,
            batch_size=batch_size,
            shuffle=train,
            collate_fn=collator
        )
    
    return loader


class SQuADQADataset(Dataset):
    """Dataset for fine-tuning GPT-2 on SQuAD for Question Answering."""

    def __init__(self, split='train', max_length=512, tokenizer_name="openai-community/gpt2", use_templates=True):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        self.use_templates = use_templates

        # Define 5 different response templates
        self.templates = [
            {
                'name': 'polite_formal',
                'prefix': 'That is a great question! ',
                'suffix': ' Let me know if you have any other questions.'
            },
            {
                'name': 'thoughtful',
                'prefix': 'Let me think about that. ',
                'suffix': ' Let me know if you have any other questions.'
            },
            {
                'name': 'concise',
                'prefix': 'Answer: ',
                'suffix': ''
            },
            {
                'name': 'casual_helpful',
                'prefix': '',
                'suffix': ' - hope this helps!'
            },
            {
                'name': 'enthusiastic',
                'prefix': 'Great question! ',
                'suffix': ' Feel free to ask more!'
            }
        ]

        # Load SQuAD dataset
        self.dataset = load_dataset("rajpurkar/squad", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        import random
        example = self.dataset[idx]

        # Extract data
        question = example['question']
        context = example['context']
        # SQuAD answers is a dict with 'text' list and 'answer_start' list
        answer = example['answers']['text'][0] if example['answers']['text'] else ""

        # Format as: Question: ... Context: ... Answer: ...
        prompt = f"Question: {question}\nContext: {context}\nAnswer: "

        if self.use_templates:
            # Randomly select a template
            template = random.choice(self.templates)
            formatted_answer = f"{template['prefix']}{answer}{template['suffix']}"
        else:
            # No template, just the answer
            formatted_answer = answer

        full_text = f"{prompt} {formatted_answer}{self.tokenizer.eos_token}"

        # Tokenize
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        input_ids = encodings['input_ids']

        # Create labels - compute loss only on the answer part
        # First, tokenize just the prompt to find where answer starts
        prompt_encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        prompt_length = len(prompt_encodings['input_ids'])

        # Create labels: -100 for prompt tokens (ignored in loss), actual tokens for answer
        labels = [-100] * prompt_length + input_ids[prompt_length:]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


class DataCollator:
    """Collator to pad batches of variable-length sequences."""
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch):
        # Find max length in batch
        max_length = max(len(item['input_ids']) for item in batch)

        input_ids = []
        labels = []
        attention_mask = []

        for item in batch:
            seq_len = len(item['input_ids'])
            padding_length = max_length - seq_len

            # Pad input_ids
            padded_input_ids = torch.cat([
                item['input_ids'],
                torch.full((padding_length,), self.pad_token_id, dtype=torch.long)
            ])
            input_ids.append(padded_input_ids)

            # Pad labels with -100 (ignored in loss computation)
            padded_labels = torch.cat([
                item['labels'],
                torch.full((padding_length,), -100, dtype=torch.long)
            ])
            labels.append(padded_labels)

            # Create attention mask (1 for real tokens, 0 for padding)
            mask = torch.cat([
                torch.ones(seq_len, dtype=torch.long),
                torch.zeros(padding_length, dtype=torch.long)
            ])
            attention_mask.append(mask)

        return {
            'input_ids': torch.stack(input_ids),
            'labels': torch.stack(labels),
            'attention_mask': torch.stack(attention_mask)
        }
