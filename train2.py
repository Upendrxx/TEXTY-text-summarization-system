import os
import torch
from multiprocessing.pool import ThreadPool
from transformers import BartTokenizer, BartForConditionalGeneration, Adafactor
from torch.utils.data import DataLoader, random_split
import random
from transformers import get_linear_schedule_with_warmup

# Data Preprocessing
def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        splits = content.split('@highlight')
        story = splits[0].strip()
        highlights = [h.strip() for h in splits[1:] if h.strip() != ""]
    return story, highlights


stories_dir = 'daily_mail_stories/dailymail'
stories = []
summaries = []

# Using a larger subset
NUM_STORIES = 50000

for path in random.sample(os.listdir(stories_dir), NUM_STORIES):
    if path.endswith('.story'):
        s, h = process_file(os.path.join(stories_dir, path))
        stories.append(s)
        summaries.append(h[0])

# Tokenization
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')


def tokenize_for_summarization(data):
    story, summary = data
    story_tokens = tokenizer(story, return_tensors="pt", truncation=True, padding='max_length', max_length=512)[
        'input_ids']
    summary_tokens = tokenizer(summary, return_tensors="pt", truncation=True, padding='max_length', max_length=150)[
        'input_ids']
    return story_tokens.squeeze(), summary_tokens.squeeze()


with ThreadPool() as pool:
    tokenized_data = pool.map(tokenize_for_summarization, zip(stories, summaries))

# Data Splitting and DataLoader
train_size = int(0.8 * len(tokenized_data))
val_size = len(tokenized_data) - train_size
train_data, val_data = random_split(tokenized_data, [train_size, val_size])


def collate_fn(batch):
    stories, summaries = zip(*batch)
    return {"input_ids": torch.stack(stories), "labels": torch.stack(summaries)}


BATCH_SIZE = 32
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_fn)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# Model Definition & Training
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
EPOCHS = 5

optimizer = Adafactor(model.parameters(), lr=1e-4, relative_step=False)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500,
                                            num_training_steps=len(train_dataloader) * EPOCHS)
GRADIENT_CLIP = 1.0

for epoch in range(EPOCHS):
    # Training
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        inputs = batch['input_ids'].to(device)
        targets = batch['labels'].to(device)
        outputs = model(input_ids=inputs, labels=targets)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS}:\nTraining loss: {total_loss / len(train_dataloader)}")

    # Validation
    model.eval()
    val_loss = 0
    for batch in val_dataloader:
        with torch.no_grad():
            inputs = batch['input_ids'].to(device)
            targets = batch['labels'].to(device)
            outputs = model(input_ids=inputs, labels=targets)
            loss = outputs.loss
            val_loss += loss.item()
    print(f"Validation loss: {val_loss / len(val_dataloader)}\n")

# Save model and tokenizer
model.save_pretrained('./bart_summarizer_improved')
tokenizer.save_pretrained('./bart_summarizer_improved')
