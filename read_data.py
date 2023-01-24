import os
import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections import Counter
from copy import deepcopy
import argparse
from sklearn.model_selection import train_test_split
import scipy


class DatasetLoader(object):
    def load(self):
        """Minimum condition for dataset:
          * All users must have at least one item record.
          * All items must have at least one user record.
        """
        raise NotImplementedError

class MovieLens100k(DatasetLoader):
    def _init_(self, data_dir):
        self.fpath_rate = os.path.join(data_dir, 'u.data')
        self.fpath_gender = os.path.join(data_dir, 'u.user')
        self.fpath_genre = os.path.join(data_dir, 'u.item')

    def load(self):
        # Load data
        df_rate = pd.read_csv(self.fpath_rate,
                              sep='\t',
                              engine='python',
                              names=['user', 'item', 'rate', 'time'],
                              usecols=['user', 'item', 'rate'])

        df_gender = pd.read_csv(self.fpath_gender,
                                sep='|',
                                engine='python',
                                names=['user', 'age', 'gender', 'occupation', 'zip'],
                                usecols=['user', 'gender'])

        df_rate['user'] = df_rate['user'].astype(str)
        df_gender['user'] = df_gender['user'].astype(str)

        df = pd.merge(df_rate, df_gender, on='user')
        df = df.dropna(axis=0, how='any')

        # TODO: Remove negative rating?
        df = df[df['rate'] > 3]
        df = df.reset_index().drop(['index'], axis=1)

        df, item_mapping = convert_unique_idx(df, 'item')

        return df, item_mapping


class LibraryThing(DatasetLoader):
    def __init__(self, data_dir):
        self.path = os.path.join(data_dir, 'reviews.txt')
        self.ndatapoints = 100
    
    def load(self):
        df = pd.DataFrame(columns = ['item', 'flags', 'rate', 'nhelpful',  'user', 'commentlength'], 
                   index = np.arange(1, self.ndatapoints, 1))
        file = open(self.path, 'r')
        lines = file.readlines()[1:]

        linecount = 0
        for line in lines:
            try:
                try:
                    line = line.split('=')
                    line = line[1]
                    comment = line.split(", 'nhelpful':")[0]
                    df['commentlength'].iloc[linecount] = len(comment[14:-1].split(" "))
                    metadata = line.split(", 'nhelpful':")[1]
                    #df['comment'].iloc[linecount] = comment[14:-1]
                    metadata = metadata.split(':')    
                    df['nhelpful'].iloc[linecount] = float(metadata[0].split(',')[0][1:])
                    df['item'].iloc[linecount] = int(metadata[2].split(',')[0][2:-1])
                    df['flags'].iloc[linecount] =  metadata[3].split(',')[0]
                    df['user'].iloc[linecount] = metadata[4].split(',')[0][2:-1]
                    df['rate'].iloc[linecount] = float(metadata[5].split(',')[0])
                    linecount +=1
                    if linecount > self.ndatapoints:
                        break
                except ValueError:
                    pass
            except IndexError:
                pass
        
        df, user_mapping = convert_unique_idx(df, 'user')
        df, item_mapping = convert_unique_idx(df, 'item')
        return df, item_mapping


        return df, item_mapping


class MovieLens1M(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath_rate = os.path.join(data_dir, 'ratings.dat')
        self.fpath_user = os.path.join(data_dir, 'users.dat')
        self.fpath_item = os.path.join(data_dir, 'movies.dat')

    def load(self):
        """ 
        Load datasets from rate, user and item sources
        """
        # O: Load movie rating information 
        df_rate = pd.read_csv(self.fpath_rate,
                              sep='::',
                              engine='python',
                              names=['user', 'item', 'rate', 'time'],
                              usecols=['user', 'item', 'rate'])
        # O: Load user demographic information
        df_user = pd.read_csv(self.fpath_user,
                              sep='::',
                              engine='python',
                              names=['user', 'gender', 'age', 'occupation', 'Zip-code'],
                              usecols=['user', 'gender', 'age'])
        # O: Load movie item information
        df_item = pd.read_csv(self.fpath_item,
                              sep='::',
                              engine='python',
                              names=['item', 'title', 'genre'],
                              usecols=['item', 'genre'],
                              encoding='unicode_escape')

        # O: Merge all data sources and clean dataframe
        df = pd.merge(df_rate, df_user, on='user')
        df = df.dropna(axis=0, how='any')

        # df = df[df['rate'] > 3]
        
        df = df.reset_index().drop(['index'], axis=1)
        df = pd.merge(df, df_item, on='item')
        
        # O: Reset index of users to zero 
        df.user = df.user - 1


        # O: Reassign movie indices
        df, item_mapping = convert_unique_idx(df, 'item')

        return df, item_mapping


def convert_unique_idx(df, column_name):
    """ O: Switch the index notation of the movie reviews in DataFrame into an ordered system. 
    Return:
        df: Dataframe with new index system
        column_dict: Mapping between original index and new ordered index
    """
    # O: Build dictionary of index mappings from original definition to ordered definition
    column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    # O: Change index of dataframe
    df[column_name] = df[column_name].apply(column_dict.get)
    df[column_name] = df[column_name].astype('int')
    
    # O: Check that new index system is of expected size
    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dict) - 1

    
    return df, column_dict


def gender_index(df):
    """ O: Find index per gender 
    return:
        index_F: List of index of Females
        index_M: List of index of Males
    """
    gender_dic = df.groupby('user')['gender'].apply(list).to_dict()
    index_F = []
    index_M = []
    for i in range(0, len(gender_dic)):
        if 'f' in gender_dic[i] or 'F' in gender_dic[i]:
            index_F.append(i)
        else:
            index_M.append(i)
    index_F = np.array(index_F) #1709 women
    index_M = np.array(index_M) #4331 men
    
    return index_F, index_M


def age_mapping_ml100k(age):
    if age < 18:
        return 0
    elif age < 25:
        return 1
    elif age < 35:
        return 2
    elif age < 45:
        return 3
    elif age < 50:
        return 4
    elif age < 56:
        return 5
    elif age >= 56:
        return 6
    else:
        print('Error in age data, age set = ', age)

def age_mapping_ml1m(age):
    if age == 1:
        return 0
    elif age == 18:
        return 1
    elif age == 25:
        return 2
    elif age == 35:
        return 3
    elif age == 45:
        return 4
    elif age == 50:
        return 5
    elif age == 56:
        return 6
    else:
        print('Error in age data, age set = ', age)

def help_mapping(nhelpful):
    if nhelpful == 0 :
        return 0
    elif nhelpful == 1:
        return 1
    elif nhelpful == 2:
        return 2
    elif nhelpful == 3:
        return 3
    elif nhelpful == 4:
        return 4
    elif nhelpful == 5:
        return 5
    elif nhelpful >= 6:
        return 6
    else:
        print('Error in nhelpful data, nhelpful set = ', nhelpful)

def age_index(df, user_size, data):
    """ 
    """
    if data in ('ml-100k', 'ml-1m'):
        dic = df.groupby('user')['age'].apply(list).to_dict()
    else:
        dic = df.groupby('user')['nhelpful'].apply(list).to_dict()

    for id, attribute in dic.items():
        if data == 'ml-100k':
            dic[id] = age_mapping_ml100k(attribute[0])
        elif data == 'ml-1m':
            dic[id] = age_mapping_ml1m(attribute[0])
        else:
            dic[id] = help_mapping(attribute[0])

    index_age = [[], [], [], [], [], [], []]
    
    for i in range(0, len(dic)):
        if 0 == dic[i]:
            index_age[0].append(i)
        elif 1 == dic[i]:
            index_age[1].append(i)
        elif 2 == dic[i]:
            index_age[2].append(i)
        elif 3 == dic[i]:
            index_age[3].append(i)
        elif 4 == dic[i]:
            index_age[4].append(i)
        elif 5 == dic[i]:
            index_age[5].append(i)
        elif 6 == dic[i]:
            index_age[6].append(i)

    for i in range(len(index_age)):
        index_age[i] = np.array(index_age[i])

    age_type = 7   
    age_mask = torch.zeros(age_type, user_size)
    for i in range(user_size):
        for k in range(age_type):
            if i in index_age[k]:
                age_mask[k][i] = 1


    return index_age, age_mask


#i am not sure i think it divides movies in 5 categories from most common to less
def pop_index(df):
    # count number of reviews per movie
    count = Counter(df['item'])
    common = count.most_common() #returns list of tuples of (element, count) sorted by counts(mostly commonly viewed rated movie)
    item_size = len(set(df['item'])) # = len(common) = 3706

    index_pop = [[], [], [], [], []]
    
    for i in range(item_size):
        #compares the number of occurences of each movie with an occurence in common idk why
        if count[i] > common[int(0.2 * len(common))][1]:
            index_pop[0].append(i)
        elif count[i] > common[int(0.4 * len(common))][1]:
            index_pop[1].append(i)
        elif count[i] > common[int(0.6 * len(common))][1]:
            index_pop[2].append(i)
        elif count[i] > common[int(0.8 * len(common))][1]:
            index_pop[3].append(i)
        else:
            index_pop[4].append(i)
  
    for i in range(len(index_pop)):
        index_pop[i] = torch.tensor(index_pop[i])

    pop_size = 5
    pop_mask = torch.zeros(pop_size, item_size) #[5, 3706] 
    for i in range(item_size):
        for k in range(pop_size):
            if i in index_pop[k]:
                pop_mask[k][i] = 1

    return index_pop, pop_mask


def genre_ml100k_index(df):
    df_genre = df[
        ['item', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9', 'g10', 'g11', 'g12', 'g13', 'g14', 'g15',
         'g16', 'g17', 'g18']]
    df_genre = df_genre.drop_duplicates(subset=['item'], keep='first').reset_index(drop=True).drop(columns=['item'])
    print('df genre', df_genre)
    genre_name = ['g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9', 'g10', 'g11', 'g12', 'g13', 'g14', 'g15',
                  'g16', 'g17', 'g18']
    index_genre = []
    for genre in genre_name:
        index_genre.append(torch.tensor(np.flatnonzero(df_genre[genre])).long())

    genre_mask = df_genre.to_numpy().T
    genre_mask = torch.FloatTensor(genre_mask)

    return index_genre, genre_mask

def engagement_index(df):
    df_length = df[['item', 'commentlength']]
    df_length = df_length.drop_duplicates(subset=['item'], keep='first').reset_index(drop=True)
    ls = df_length['commentlength'].tolist()

    for i in range(len(ls)):
        # for j in range(len(ls[i])):
        if ls[i] <= 50:
            ls[i] = 0
        elif 50 < ls[i] <= 100:
            ls[i] = 1
        elif 100 < ls[i] <= 150:
            ls[i]= 2
        elif 150 < ls[i] <= 200:
            ls[i] = 3
        elif 200 < ls[i] <= 250:
            ls[i] = 4
        elif 250 < ls[i] <= 300:
            ls[i] = 5
        elif 300 < ls[i] <= 350:
            ls[i] = 6
        elif 350 < ls[i] <= 400:
            ls[i] = 7
        elif 400 < ls[i] <= 450:
            ls[i] = 8
        elif 450 < ls[i] <= 500:
            ls[i] = 9
        elif 500 < ls[i] <= 550:
            ls[i] = 10
        elif 550 < ls[i] <= 600:
            ls[i] = 11
        elif 600 < ls[i] <= 650:
            ls[i] = 12
        elif 650 < ls[i] <= 700:
            ls[i] = 13
        elif 700 < ls[i] <= 750:
            ls[i] = 14
        elif 750 < ls[i] <= 800:
            ls[i] = 15
        elif 800 < ls[i] <= 850:
            ls[i] = 16
        elif ls[i] > 850:
            ls[i] = 17

    genre_mask = torch.zeros(18, len(df_length)) # [18, 3706]
    for i in range(len(df_length)):
        for k in range(0, 18):
            if k in ls:
                genre_mask[k] = 1

    index_genre = [] #list 18
    for i in range(genre_mask.shape[0]):
        index_genre.append(torch.tensor(np.where(genre_mask[i] == 1)[0]).long())
    
    return index_genre, genre_mask


def genre_ml1m_index(df):
    df_genre = df[['item', 'genre']]
    df_genre = df_genre.drop_duplicates(subset=['item'], keep='first').reset_index(drop=True)
    # ls = df_genre['commentlength'].tolist()
    ls = df_genre['genre'].tolist()
    for i in range(len(ls)):
        ls[i] = ls[i].split("|")
    
    for i in range(len(ls)):
        for j in range(len(ls[i])):
            if ls[i][j] == 'Action':
                ls[i][j] = 0
            elif ls[i][j] == 'Adventure':
                ls[i][j] = 1
            elif ls[i][j] == 'Animation':
                ls[i][j] = 2
            elif ls[i][j] == "Children's":
                ls[i][j] = 3
            elif ls[i][j] == 'Comedy':
                ls[i][j] = 4
            elif ls[i][j] == 'Crime':
                ls[i][j] = 5
            elif ls[i][j] == 'Documentary':
                ls[i][j] = 6
            elif ls[i][j] == 'Drama':
                ls[i][j] = 7
            elif ls[i][j] == 'Fantasy':
                ls[i][j] = 8
            elif ls[i][j] == 'Film-Noir':
                ls[i][j] = 9
            elif ls[i][j] == 'Horror':
                ls[i][j] = 10
            elif ls[i][j] == 'Musical':
                ls[i][j] = 11
            elif ls[i][j] == 'Mystery':
                ls[i][j] = 12
            elif ls[i][j] == 'Romance':
                ls[i][j] = 13
            elif ls[i][j] == 'Sci-Fi':
                ls[i][j] = 14
            elif ls[i][j] == 'Thriller':
                ls[i][j] = 15
            elif ls[i][j] == 'War':
                ls[i][j] = 16
            elif ls[i][j] == 'Western':
                ls[i][j] = 17


    genre_mask = torch.zeros(18, len(df_genre)) # [18, 3706]
    for i in range(len(df_genre)):
        for k in range(0, 18):
            if k in ls[i]:
                genre_mask[k][i] = 1

    index_genre = [] #list 18
    for i in range(genre_mask.shape[0]):
        index_genre.append(torch.tensor(np.where(genre_mask[i] == 1)[0]).long())
    
    return index_genre, genre_mask


def preprocessing(args):
    """ O: Arranging review data
    1. Load data from directory 
    2. Construct sparse matrix of labels
    """
    data_dir = os.path.join('./data', args.data)
    
    #following line is commented out because otherwise there is a local var df referenced before assignment error
    if args.data == 'ml-1m':
        df, item_mapping = MovieLens1M(data_dir).load()
    elif args.data == 'lt':
        df, item_mapping = LibraryThing(data_dir).load()
    else:
        df, item_mapping = LibraryThing(data_dir).load()

   
    user_size = len(df['user'].unique())
    item_size = len(df['item'].unique())
    
    print("user_size:", user_size)
    print("item_size:", item_size)

    #print(df.head(10))

    """construct matrix_label"""
    #if the rating is >3 the user would watch it
    #matrix label is the rating matrix: a 6040(#users)X3706(#items) matrix needed for matrix factorisation

    df_rate = df[["user", "item", "rate"]]
    # O: Keep only movies with ratings larger than 3 
    df_rate = df_rate[df_rate['rate'] > 3]
    df_rate = df_rate.reset_index().drop(['index'], axis=1)
    
    # O: set al rates to 1 
    df_rate['rate'] = 1
    # O: (user, item), rate
    matrix_label = scipy.sparse.csr_matrix(
        (np.array(df_rate['rate']), (np.array(df_rate['user']), np.array(df_rate['item']))))


    return df, item_mapping, matrix_label, user_size, item_size


def obtain_group_index(df, args):
    """
    """
    
    user_size = len(df['user'].unique())
    
    #matrices of where in the df there is an index for each case
    index_F, index_M = gender_index(df)
    #list of size 2 that has the #women and #men
    index_gender = [torch.tensor(index_F).long(), torch.tensor(index_M).long()]
    #an array of arrays for all 7 age groups and an array that has 1 if the user belongs to a specific age group
    index_age, age_mask = age_index(df, user_size, args.data)
    index_pop, pop_mask = pop_index(df)
    # print("index_age:", index_age)
    # print("index_pop:", index_pop)
    index_genre = []
    if args.data == 'ml-100k':
        index_genre, genre_mask = genre_ml100k_index(df)
    elif args.data == 'ml-1m':
        index_genre, genre_mask = genre_ml1m_index(df)

    # print("pop_mask:", pop_mask, pop_mask.shape)
    # print("genre_mask:", genre_mask, genre_mask.shape)

    return index_F, index_M, index_gender, index_age, index_genre, index_pop, age_mask, pop_mask, genre_mask

def obtain_group_index_tl(df, args):
    user_size = len(df['user'].unique())
    #matrices of where in the df there is an index for each group
    index_engagement, engagement_mask = engagement_index(df)
    index_helpful, helpful_mask = age_index(df, user_size, args.data)

    return index_engagement, index_helpful, engagement_mask, helpful_mask

def obtain_group_index_tl(df, args):
    user_size = len(df['user'].unique())
    #matrices of where in the df there is an index for each group
    index_engagement, engagement_mask = engagement_index(df)
    index_helpful, helpful_mask = helfulness_index(df)

    return index_engagement, index_helpful, engagement_mask, helpful_mask



def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="lt")
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args()
    df, item_mapping, matrix_label, user_size, item_size = preprocessing(args)
    index_engagement, index_helpful, engagement_mask, helpful_mask = obtain_group_index_tl(df, args)
    # print("matrix_label:", matrix_label.todense().shape)
    # print(df)
