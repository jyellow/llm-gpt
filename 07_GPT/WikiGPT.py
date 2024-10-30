import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载WikiText2数据集
ds = load_dataset("wikitext", "wikitext-2-v1")

# 使用BERT的tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 定义tokenize函数
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

# 对数据集进行tokenization
tokenized_datasets = ds.map(tokenize_function, batched=True, remove_columns=["text"])

# 创建词汇表
vocab = tokenizer.get_vocab()

# 打印词汇表信息
print("词汇表大小:", len(vocab))
print("词汇示例(word to index):", 
      {word: vocab[word] for word in ["[PAD]", "[CLS]", "[SEP]", "the", "apple"]})

# 获取训练集和验证集
train_dataset = tokenized_datasets["train"]
valid_dataset = tokenized_datasets["validation"]

# 定义自定义Dataset类
class WikiDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return torch.tensor(item['input_ids']), torch.tensor(item['attention_mask'])

# 创建DataLoader
train_dataloader = DataLoader(WikiDataset(train_dataset), batch_size=32, shuffle=True)
valid_dataloader = DataLoader(WikiDataset(valid_dataset), batch_size=32, shuffle=False)

# 定义模型参数
d_model = 512
n_heads = 8
n_layers = 6
d_ff = 2048
max_seq_length = 512
vocab_size = len(vocab)

# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_length):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_mask):
        src = self.embedding(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output

# 定义位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# 初始化模型
model = TransformerModel(vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_length).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 训练函数
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids, attention_mask = batch
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        optimizer.zero_grad()
        output = model(input_ids, attention_mask)
        loss = criterion(output.view(-1, vocab_size), input_ids.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 评估函数
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            output = model(input_ids, attention_mask)
            loss = criterion(output.view(-1, vocab_size), input_ids.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, optimizer, criterion, device)
    val_loss = evaluate(model, valid_dataloader, criterion, device)
    print(f'Epoch: {epoch+1}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}')

# 保存模型
torch.save(model.state_dict(), 'transformer_model.pth')

# 生成文本函数
def generate_text(model, input_text, max_length=50):
    model.eval()
    tokens = tokenizer.encode(input_text, return_tensors='pt').to(device)
    with torch.no_grad():
        for _ in range(max_length):
            output = model(tokens, None)
            next_token = torch.argmax(output[:, -1, :])
            tokens = torch.cat([tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            if next_token == tokenizer.eos_token_id:
                break
    return tokenizer.decode(tokens[0])

# 测试生成文本
input_text = "The quick brown fox"
generated_text = generate_text(model, input_text)
print(f"Input: {input_text}")
print(f"Generated: {generated_text}")
