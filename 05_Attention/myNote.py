import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 假设我们使用 MovieLens 数据集
movies = pd.read_csv('./ml-latest-small/movies.csv')
ratings = pd.read_csv('./ml-latest-small/ratings.csv')

# 定义用户兴趣向量 U 和电影特征向量 M
# def create_user_vector():
#     # 用户兴趣向量: [喜欢爱情片, 喜欢动作片, 喜欢恐怖片, 喜欢喜剧片]
#     return np.array([0.8, 0.6, 0.2, 0.7])

# def create_movie_vector():
#     # 电影特征向量: [动作指数, 浪漫指数, 恐怖指数, 喜剧指数]
#     return np.array([0.7, 0.3, 0.1, 0.5])
def create_user_vector(user_id):
    # 获取用户的评分历史
    user_ratings = ratings[ratings['userId'] == user_id]
    
    # 计算用户对每种类型电影的平均评分
    user_genre_ratings = user_ratings.merge(movies, on='movieId')
    user_genre_ratings['genres'] = user_genre_ratings['genres'].str.split('|')
    user_genre_ratings = user_genre_ratings.explode('genres')
    user_vector = user_genre_ratings.groupby('genres')['rating'].mean()
    
    # 标准化评分
    scaler = MinMaxScaler()
    user_vector = scaler.fit_transform(user_vector.values.reshape(-1, 1)).flatten()
    
    return user_vector

def create_movie_vector(movie_id):
    # 获取电影的类型信息
    movie_genres = movies[movies['movieId'] == movie_id]['genres'].iloc[0].split('|')
    
    # 创建电影特征向量
    all_genres = movies['genres'].str.split('|', expand=True).stack().unique()
    movie_vector = np.zeros(len(all_genres))
    for genre in movie_genres:
        if genre in all_genres:
            movie_vector[np.where(all_genres == genre)[0][0]] = 1
    
    return movie_vector

# 注意力机制
def attention_mechanism(user_vector, movie_vector):
    # 计算注意力权重，使用点积注意力机制，同时除以向量长度的平方根以稳定训练过程
    attention_weights = np.dot(user_vector, movie_vector) / np.sqrt(len(user_vector))
    # 应用softmax使权重和为1
    attention_weights = np.exp(attention_weights) / np.sum(np.exp(attention_weights))
    return attention_weights

# 计算用户对电影的兴趣度得分
def calculate_interest_score(user_vector, movie_vector):
    attention_weights = attention_mechanism(user_vector, movie_vector)
    score = np.sum(attention_weights * user_vector * movie_vector)
    return score

# 主函数
def main():
    user_vector = create_user_vector()
    movie_vector = create_movie_vector()
    
    interest_score = calculate_interest_score(user_vector, movie_vector)
    
    print("用户兴趣向量:", user_vector)
    print("电影特征向量:", movie_vector)
    print("注意力权重:", attention_mechanism(user_vector, movie_vector))
    print("用户对电影的兴趣度得分:", interest_score)
    
    if interest_score > 0.5:
        print("系统可能会推荐这部电影给用户。")
    else:
        print("系统可能不会推荐这部电影给用户。")

if __name__ == "__main__":
    main()
