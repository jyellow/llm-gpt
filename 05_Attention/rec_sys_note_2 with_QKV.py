import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# 文件路径
USER_VECTORS_FILE = 'user_vectors.pkl'
MOVIE_KEY_VECTORS_FILE = 'movie_key_vectors.pkl'
MOVIE_VALUE_VECTORS_FILE = 'movie_value_vectors.pkl'

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


def create_movie_key_vectors():
    # 获取所有电影的类型信息
    movies['genres'] = movies['genres'].str.split('|')
    
    # 获取所有唯一的电影类型
    all_genres = set(genre for genres in movies['genres'] for genre in genres)
    
    # 创建电影特征矩阵
    movie_vectors = pd.DataFrame(0, index=movies['movieId'], columns=list(all_genres))
    
    for _, row in movies.iterrows():
        movie_vectors.loc[row['movieId'], row['genres']] = 1
    
    return movie_vectors

def create_movie_value_vectors():
    # 这里我们创建一个不同的表示作为Value
    # 例如,我们可以使用电影的平均评分和评分数量
    movie_stats = ratings.groupby('movieId').agg({
        'rating': ['mean', 'count']
    })
    movie_stats.columns = ['avg_rating', 'rating_count']
    
    # 创建一个与电影Key特征矩阵相同的索引
    all_movie_ids = movies['movieId'].unique()
    movie_stats = movie_stats.reindex(all_movie_ids, fill_value=0)  # 补齐缺失的电影ID, 默认值为0
    
    # 标准化这些特征
    scaler = MinMaxScaler()
    movie_stats = pd.DataFrame(
        scaler.fit_transform(movie_stats),
        index=movie_stats.index,
        columns=movie_stats.columns
    )
    
    return movie_stats

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
    
    # 检查电影Key向量文件是否存在
    if os.path.exists(MOVIE_KEY_VECTORS_FILE):
        print("正在从本地加载电影Key向量...")
        movie_key_vectors = load_vectors(MOVIE_KEY_VECTORS_FILE)
    else:
        print("正在创建电影Key向量...")
        movie_key_vectors = create_movie_key_vectors()
        save_vectors(movie_key_vectors, MOVIE_KEY_VECTORS_FILE)
    
    # 检查电影Value向量文件是否存在
    if os.path.exists(MOVIE_VALUE_VECTORS_FILE):
        print("正在从本地加载电影Value向量...")
        movie_value_vectors = load_vectors(MOVIE_VALUE_VECTORS_FILE)
    else:
        print("正在创建电影Value向量...")
        movie_value_vectors = create_movie_value_vectors()
        save_vectors(movie_value_vectors, MOVIE_VALUE_VECTORS_FILE)
    
    return user_vectors, movie_key_vectors, movie_value_vectors


def attention_mechanism(query, key, value):
    """
    实现注意力机制。

    在注意力机制中, Q、K、V 分别代表 Query(查询) 、Key(键) 和 Value(值) : 
    
    - Query (Q): 代表当前我们关注的对象或想要查找的信息。在推荐系统中, 
      Q 通常代表用户的兴趣或当前的查询内容。可以理解为"我们正在寻找什么"。
    
    - Key (K): 代表可以被查询的信息或可以与查询进行匹配的内容。在推荐系统中, 
      K 通常代表物品( 如电影) 的特征或属性。可以理解为"可以用来匹配查询的信息"。
    
    - Value (V): 代表与 Key 相关联的实际内容或信息。在得到 Q 和 K 的匹配度之后, 
      V 用来计算最终的输出。可以理解为"我们实际上想要获取的信息"。

    计算过程: 
    1. 计算 Q 和 K 之间的相似度或匹配度( 通过点积操作) 。
    2. 对计算出的匹配度进行 softmax 操作, 得到注意力权重。
    3. 用这些权重对 V 进行加权求和, 得到最终的输出。

    在推荐系统中, Q 可能代表用户的兴趣特征, K 和 V 可能代表电影的特征。
    通过这种机制, 系统可以根据用户兴趣( Q) 和电影特征( K) 的匹配程度, 
    为用户推荐最相关的电影内容( V) 。

    注意力机制的优势在于能够动态地为不同的输入分配不同的重要性, 
    使得模型能够"关注"最相关的信息, 从而提高推荐的准确性。

    参数:
    query (np.ndarray): 形状为 (num_users, num_features) 的查询矩阵
    key (np.ndarray): 形状为 (num_movies, num_features) 的键矩阵
    value (np.ndarray): 形状为 (num_movies, num_features) 的值矩阵

    维度含义：
    - attention_scores 的形状为 (num_users, num_movies), 表示每个用户对每部电影的注意力分数。
    - attention_weights 的形状与 attention_scores 相同, 即 (num_users, num_movies)。
    
    各维度的具体含义：
    1. 第一个维度 (num_users):
       表示用户的数量。每一行对应一个用户。
    2. 第二个维度 (num_movies):
       表示电影的数量。每一列对应一部电影。
    
    attention_weights 的含义：
    attention_weights 中的每个元素表示特定用户对特定电影的注意力权重。
    权重是通过对 attention_scores 应用 softmax 函数计算得出的, 确保每个用户的权重和为1。
    这意味着对于每个用户, 所有电影的注意力权重加起来等于1, 反映了该用户对不同电影的相对关注程度。

    weighted_sum 的维度含义：
    
    1. 形状：(num_users, num_features)
       - num_users: 用户数量
       - num_features: 电影特征的数量（与 value 矩阵的第二个维度相同）
    
    2. 计算过程：
       - attention_weights 的形状为 (num_users, num_movies)
       - value 的形状为 (num_movies, num_features)
       - np.dot 操作将这两个矩阵相乘, 结果是 (num_users, num_features)
    
    3. 含义：
       - weighted_sum 中的每一行代表一个用户
       - 每一行包含了该用户对所有电影特征的加权和
       - 权重来自 attention_weights, 表示用户对每部电影的关注度
       - 这个加权和反映了用户对不同电影特征的综合偏好
    
    4. 使用：
       - weighted_sum 可以用于后续计算用户对电影的兴趣得分
       - 它捕捉了用户兴趣与电影特征之间的复杂关系
    
    返回:
    np.ndarray: 形状为 (num_users, num_features) 的注意力加权结果
    """
    # 计算注意力分数
    attention_scores = np.dot(query, key.T)
    
    # 应用缩放因子
    attention_scores = attention_scores / np.sqrt(query.shape[1])
    
    # 应用 softmax 获取注意力权重    
    attention_weights = np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), 1, attention_scores)
    
    # 计算加权和
    weighted_sum = np.dot(attention_weights, value)
    
    return weighted_sum

def calculate_interest_scores(user_vectors, movie_key_vectors, movie_value_vectors):
    """
    计算所有用户对所有电影的兴趣得分。

    参数:
    user_vectors (pd.DataFrame): 用户特征向量
    movie_key_vectors (pd.DataFrame): 电影Key特征向量
    movie_value_vectors (pd.DataFrame): 电影Value特征向量

    返回:
    np.ndarray: 形状为 (num_users, num_movies) 的兴趣得分矩阵
    """
    # 将用户向量作为查询 (Q)
    query = user_vectors.values
    
    # 将电影Key向量作为键 (K)
    key = movie_key_vectors.values
    
    # 将电影Value向量作为值 (V)
    value = movie_value_vectors.values
    
    # 应用注意力机制
    weighted_sum = attention_mechanism(query, key, value)
    
    # 计算最终的兴趣得分
    scores = np.dot(weighted_sum, value.T)
    
    return scores


if __name__ == "__main__":
    user_vectors, movie_key_vectors, movie_value_vectors = get_or_create_vectors()
    scores = calculate_interest_scores(user_vectors, movie_key_vectors, movie_value_vectors)

    # 随机选择一个用户ID
    random_user_id = np.random.choice(user_vectors.index)

    # 随机选择一部电影ID
    random_movie_id = np.random.choice(movie_key_vectors.index)

    # 获取该用户对该电影的兴趣得分
    user_index = user_vectors.index.get_loc(random_user_id)
    movie_index = movie_key_vectors.index.get_loc(random_movie_id)
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