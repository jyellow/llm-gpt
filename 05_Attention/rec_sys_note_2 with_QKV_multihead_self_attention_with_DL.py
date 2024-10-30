### PyTorch模型还没有调试好
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# 文件路径
USER_VECTORS_FILE = 'user_vectors.pkl'
MOVIE_VECTORS_FILE = 'movie_vectors.pkl'

# 假设我们使用 MovieLens 数据集
movies_path = '/Users/johannyellow/workspace/github.com/llm-gpt/05_Attention/ml-latest-small/movies.csv'
ratings_path = '/Users/johannyellow/workspace/github.com/llm-gpt/05_Attention/ml-latest-small/ratings.csv'
movies = pd.read_csv(movies_path)
ratings = pd.read_csv(ratings_path)


def create_user_vectors():
    """
    创建用户特征向量,表示每个用户对不同类型电影的偏好。

    处理步骤:
    1. 数据合并: 将用户评分数据与电影数据基于movieId进行合并,获得每个评分对应的电影类型信息。
    2. 类型分割: 将genres列从字符串(如"Action|Adventure")分割成列表(['Action', 'Adventure'])。
    3. 数据展开: 将每部电影的多个类型展开成多行,便于单独计算每种类型的平均评分。
    4. 计算平均评分: 按用户ID和电影类型分组,计算平均评分,并转换为二维表格。
    5. 评分标准化: 使用MinMaxScaler将评分缩放到0-1之间,消除不同用户评分标准的差异。
    6. 创建DataFrame: 将标准化后的数据重新构造成DataFrame,保持原有的用户ID和电影类型信息。

    返回:
    pandas.DataFrame: 用户特征向量矩阵
        - 索引: 用户ID
        - 列: 电影类型
        - 值: 该用户对该类型电影的标准化平均评分(0-1之间)

    注意:
    - 此函数假设全局变量 'ratings' 和 'movies' 已经被定义并包含所需的数据。
    - 函数使用 pandas 和 sklearn.preprocessing.MinMaxScaler 进行数据处理。
    """
    # 合并评分数据和电影数据,基于movieId
    user_genre_ratings = ratings.merge(movies, on='movieId')
    
    # 将genres列从字符串分割成列表
    user_genre_ratings['genres'] = user_genre_ratings['genres'].str.split('|')
    
    # 将每个电影的多个类型展开成多行
    user_genre_ratings = user_genre_ratings.explode('genres')
    
    # 计算每个用户对每种类型电影的平均评分
    user_vectors = user_genre_ratings.groupby(['userId', 'genres'])['rating'].mean().unstack(fill_value=0)
    
    # 使用MinMaxScaler进行评分标准化
    scaler = MinMaxScaler()
    user_vectors = pd.DataFrame(
        scaler.fit_transform(user_vectors),  # 将评分缩放到0-1之间
        index=user_vectors.index,  # 保持原有的用户ID索引
        columns=user_vectors.columns  # 保持原有的电影类型列名
    )
    
    return user_vectors


def create_movie_vectors():
    # 获取所有电影的类型信息
    movies['genres'] = movies['genres'].str.split('|')
    
    # 获取所有唯一的电影类型
    all_genres = set(genre for genres in movies['genres'] for genre in genres)
    
    # 创建电影特征矩阵
    movie_vectors = pd.DataFrame(0, index=movies['movieId'], columns=list(all_genres))
    
    for _, row in movies.iterrows():
        movie_vectors.loc[row['movieId'], row['genres']] = 1
    
    return movie_vectors

def save_vectors(vectors, filename):
    with open(filename, 'wb') as f:
        pickle.dump(vectors, f)

def load_vectors(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_or_create_vectors():
    # 检查用户向量文件是否存在
    if os.path.exists(USER_VECTORS_FILE):
        print("正在从本地加载用户向量...")
        user_vectors = load_vectors(USER_VECTORS_FILE)
    else:
        print("正在创建用户向量...")
        user_vectors = create_user_vectors()
        save_vectors(user_vectors, USER_VECTORS_FILE)
    
    # 检查电影向量文件是否存在
    if os.path.exists(MOVIE_VECTORS_FILE):
        print("正在从本地加载电影向量...")
        movie_vectors = load_vectors(MOVIE_VECTORS_FILE)
    else:
        print("正在创建电影向量...")
        movie_vectors = create_movie_vectors()
        save_vectors(movie_vectors, MOVIE_VECTORS_FILE)
    
    return user_vectors, movie_vectors

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attention_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, V)
    
    def split_heads(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    
    def forward(self, x):
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(x.size(0), -1, self.d_model)
        
        return self.W_o(attn_output)

class RecommenderModel(nn.Module):
    def __init__(self, d_model, num_heads):
        super(RecommenderModel, self).__init__()
        self.user_attention = MultiHeadAttention(d_model, num_heads)
        self.movie_attention = MultiHeadAttention(d_model, num_heads)
        self.final_layer = nn.Linear(d_model, 1)
    
    def forward(self, user_vectors, movie_vectors):
        user_attn = self.user_attention(user_vectors)
        movie_attn = self.movie_attention(movie_vectors)
        
        scores = torch.matmul(user_attn, movie_attn.transpose(-2, -1))
        return self.final_layer(scores).squeeze(-1)

def train_model(model, user_vectors, movie_vectors, ratings, num_epochs=10, batch_size=32):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(0, len(ratings), batch_size):
            batch_ratings = ratings[i:i+batch_size]
            user_ids = batch_ratings['userId'].values
            movie_ids = batch_ratings['movieId'].values
            
            user_batch = user_vectors.loc[user_ids].values
            movie_batch = movie_vectors.loc[movie_ids].values
            
            user_batch = torch.FloatTensor(user_batch)
            movie_batch = torch.FloatTensor(movie_batch)
            
            predicted_scores = model(user_batch, movie_batch)
            actual_scores = torch.FloatTensor(batch_ratings['rating'].values)
            
            loss = criterion(predicted_scores, actual_scores)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

def calculate_interest_scores(model, user_vectors, movie_vectors):
    model.eval()
    with torch.no_grad():
        user_batch = torch.FloatTensor(user_vectors.values)
        movie_batch = torch.FloatTensor(movie_vectors.values)
        scores = model(user_batch, movie_batch)
    return scores.numpy()

# 主函数
if __name__ == "__main__":
    user_vectors, movie_vectors = get_or_create_vectors()
    
    d_model = user_vectors.shape[1]  # 假设用户向量和电影向量的特征数量相同
    num_heads = 4
    model = RecommenderModel(d_model, num_heads)
    
    # 训练模型
    train_model(model, user_vectors, movie_vectors, ratings)
    
    # 计算兴趣得分
    scores = calculate_interest_scores(model, user_vectors, movie_vectors)

    # 随机选择一个用户ID和电影ID
    random_user_id = np.random.choice(user_vectors.index)
    random_movie_id = np.random.choice(movie_vectors.index)

    # 获取该用户对该电影的兴趣得分
    user_index = user_vectors.index.get_loc(random_user_id)
    movie_index = movie_vectors.index.get_loc(random_movie_id)
    interest_score = scores[user_index, movie_index]

    # 设定阈值和判断用户是否喜欢这部电影
    threshold = np.mean(scores)
    likes_movie = interest_score > threshold

    print(f"\n随机选择的用户ID: {random_user_id}")
    print(f"随机选择的电影ID: {random_movie_id}")
    print(f"该用户对该电影的兴趣得分: {interest_score:.4f}")
    print(f"平均兴趣得分 (阈值): {threshold:.4f}")
    print(f"预测结果: 用户{'喜欢' if likes_movie else '不喜欢'}这部电影")