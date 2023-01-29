import os 
import pandas as pd 

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
                              usecols=['user', 'gender', 'occupation', 'age'])
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