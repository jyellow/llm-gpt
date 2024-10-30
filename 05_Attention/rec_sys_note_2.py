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

# 注意力机制
def attention_mechanism(user_vectors, movie_vectors):
    # 转置电影向量矩阵以便进行矩阵乘法
    # 计算注意力分数
    attention_scores = np.dot(user_vectors, movie_vectors.T)
    
    # 应用缩放因子
    attention_scores = attention_scores / np.sqrt(user_vectors.shape[1])
    
    # 对每个用户的分数应用 softmax
    attention_weights = np.apply_along_axis(lambda x: np.exp(x) / np.sum(np.exp(x)), 1, attention_scores)
    
    return attention_weights



# 计算用户对电影的兴趣度得分
def calculate_interest_scores(attention_weights, user_vectors, movie_vectors):
    """
    计算所有用户对所有电影的兴趣得分。

    该函数使用注意力权重、用户特征向量和电影特征向量来计算兴趣得分。
    它利用NumPy的广播机制和向量化操作来高效地进行计算。

    参数:
    attention_weights (np.ndarray): 形状为(num_users, num_movies)的注意力权重矩阵
    user_vectors (pd.DataFrame): 形状为(num_users, num_features)的用户特征向量矩阵
    movie_vectors (pd.DataFrame): 形状为(num_movies, num_features)的电影特征向量矩阵

    返回:
    np.ndarray: 形状为(num_users, num_movies)的兴趣得分矩阵, 每个元素代表一个用户对一部电影的兴趣得分

    计算步骤:
    1. 扩展注意力权重矩阵为(num_users, num_movies, 1)
    2. 扩展用户向量矩阵为(num_users, 1, num_features)
    3. 扩展电影向量矩阵为(1, num_movies, num_features)
    4. 利用NumPy的广播机制将这三个张量广播到兼容的形状(num_users, num_movies, num_features)
    5. 进行元素级别的乘法
    6. 在特征维度（第三个维度）上进行求和, 得到最终的兴趣得分矩阵

    等价的循环实现:
    def calculate_interest_scores_loop(attention_weights, user_vectors, movie_vectors):
        num_users, num_movies = attention_weights.shape
        num_features = user_vectors.shape[1]
        
        scores = np.zeros((num_users, num_movies))
        
        for u in range(num_users):
            for m in range(num_movies):
                score = 0
                for f in range(num_features):
                    score += (attention_weights[u, m] * 
                              user_vectors.values[u, f] * 
                              movie_vectors.values[m, f])
                scores[u, m] = score
        
        return scores
    
    验证循环实现和向量化实现的等价性:
    
    import time

    # 向量化版本
    start_time = time.time()
    scores_vectorized = calculate_interest_scores(attention_weights, user_vectors, movie_vectors)
    vectorized_time = time.time() - start_time

    # 循环版本
    start_time = time.time()
    scores_loop = calculate_interest_scores_loop(attention_weights, user_vectors, movie_vectors)
    loop_time = time.time() - start_time

    # 验证结果是否相同
    assert np.allclose(scores_vectorized, scores_loop), "结果不一致"

    print(f"向量化版本耗时: {vectorized_time:.4f} 秒")
    print(f"循环版本耗时: {loop_time:.4f} 秒")
    print(f"向量化版本比循环版本快 {loop_time / vectorized_time:.2f} 倍")

    示例:
    >>> scores = calculate_interest_scores(attention_weights, user_vectors, movie_vectors)
    >>> print(scores.shape)
    (num_users, num_movies)

    详细计算案例:
    假设我们有2个用户、3部电影和电影总共有4个特征。那么: 
    - attention_weights 形状为 (2, 3, 1)
    - user_vectors 形状为 (2, 1, 4)
    - movie_vectors 形状为 (1, 3, 4)

    广播后, 它们都会变成 (2, 3, 4) 的形状, 然后进行元素级别的乘法。
    例如, 对于用户0、电影1、特征2: 
    结果[0, 1, 2] = attention_weights[0, 1] * user_vectors[0, 2] * movie_vectors[1, 2]

    这个计算会对所有的用户、电影和特征组合进行。最后, 沿着特征轴求和, 
    得到形状为 (2, 3) 的结果, 表示每个用户对每部电影的兴趣得分。
    
    流程图: 
        https://www.mermaidchart.com/app/projects/0fa7df68-fa72-46b5-b5d2-9b45303a9997/diagrams/0363f67e-675c-4a95-9ecd-9c674bcb6baa/version/v0.1/edit
    """
    scores = np.sum(
        attention_weights[:, :, np.newaxis] *
        user_vectors.values[:, np.newaxis, :] *
        movie_vectors.values[np.newaxis, :, :],
        axis=2
    )
    return scores



# 主程序
if __name__ == "__main__":
    user_vectors, movie_vectors = get_or_create_vectors()
    scores = calculate_interest_scores(user_vectors, movie_vectors)
    # 使用示例
    # attention_weights = attention_mechanism(user_vectors.values, movie_vectors.values)

    # print("注意力权重矩阵形状:", attention_weights.shape)
    # print("注意力权重示例 (前5个用户对前5部电影):")
    # print(attention_weights[:5, :5])
    # print(user_vectors.shape)
    # print(movie_vectors.shape)
    # print("用户兴趣向量示例:")
    # print(user_vectors.head())
    # print("\n电影特征向量示例:")
    # print(movie_vectors.head())


    # print("兴趣得分矩阵形状:", scores.shape)
    # print("兴趣得分示例 (前5个用户对前5部电影):")
    # print(scores[:5, :5])
    # 随机选择一个用户ID
    random_user_id = np.random.choice(user_vectors.index)

    # 随机选择一部电影ID
    random_movie_id = np.random.choice(movie_vectors.index)

    # ��取该用户对该电影的兴趣得分
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