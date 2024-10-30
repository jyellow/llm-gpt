import pandas as pd
import numpy as np
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


def multi_head_self_attention(input_vectors, num_heads=4):
    """
    实现多头自注意力机制，使用输入向量的实际值。

    参数:
    input_vectors (np.ndarray): 形状为 (num_items, num_features) 的输入矩阵
    num_heads (int): 注意力头的数量

    返回:
    np.ndarray: 形状为 (num_items, num_features) 的多头自注意力加权结果
    """
    d_model = input_vectors.shape[1]
    d_k = d_model // num_heads
    
    def attention(Q, K, V):
        scores = np.dot(Q, K.T) / np.sqrt(d_k)
        attention_weights = np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), 1, scores)
        return np.dot(attention_weights, V)
    
    heads = []
    for i in range(num_heads):
        # 使用输入向量的不同部分作为Q、K、V
        start_idx = i * d_k
        end_idx = (i + 1) * d_k
        Q = input_vectors[:, start_idx:end_idx]
        K = input_vectors[:, start_idx:end_idx]
        V = input_vectors[:, start_idx:end_idx]
        
        head = attention(Q, K, V)
        heads.append(head)
    """
    - heads 是一个列表，包含了每个注意力头的输出
    - np.concatenate(..., axis=1) 将这些输出在列方向上连接
    - 结果 multi_head 的形状为 (num_items, d_model * num_heads)
    - 这一步将所有注意力头的结果合并成一个大矩阵
    """
    multi_head = np.concatenate(heads, axis=1)
    
    # 使用输入向量的平均值作为最后的线性变换
    W_o = np.mean(input_vectors, axis=0).reshape(d_model, 1)
    return np.dot(multi_head, W_o)


def calculate_interest_scores(user_vectors, movie_vectors, num_heads=4):
    """
    计算所有用户对所有电影的兴趣得分，使用多头自注意力机制。

    参数:
    user_vectors (pd.DataFrame): 用户特征向量
    movie_vectors (pd.DataFrame): 电影特征向量
    num_heads (int): 注意力头的数量

    返回:
    np.ndarray: 形状为 (num_users, num_movies) 的兴趣得分矩阵
    """
    # 应用多头自注意力机制到用户向量
    user_attention = multi_head_self_attention(user_vectors.values, num_heads)
    
    # 应用多头自注意力机制到电影向量
    movie_attention = multi_head_self_attention(movie_vectors.values, num_heads)
    
    # 计算最终的兴趣得分
    scores = np.dot(user_attention, movie_attention.T)
    
    return scores

# 主函数
if __name__ == "__main__":
    user_vectors, movie_vectors = get_or_create_vectors()
    scores = calculate_interest_scores(user_vectors, movie_vectors)

    # 随机选择一个用户ID
    random_user_id = np.random.choice(user_vectors.index)

    # 随机选择一部电影ID
    random_movie_id = np.random.choice(movie_vectors.index)

    # 获取该用户对该电影的兴趣得分
    user_index = user_vectors.index.get_loc(random_user_id)
    movie_index = movie_vectors.index.get_loc(random_movie_id)
    interest_score = scores[user_index, movie_index]

    # 设定一个阈值来判断用户是否喜欢这部电影
    threshold = np.mean(scores)  # 使用平均分作为阈值

    # 判断用户是否喜欢这部电影
    likes_movie = interest_score > threshold

    print(f"\n随机选择的用户ID: {random_user_id}")
    print(f"随机选择的电影ID: {random_movie_id}")
    print(f"该用户对该电影的兴趣得分: {interest_score:.4f}")
    print(f"平均兴趣得分 (阈值): {threshold:.4f}")
    print(f"预测结果: 用户{'喜欢' if likes_movie else '不喜欢'}这部电影")