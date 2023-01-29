import os 
import pandas as pd 
from Dataloders.MovieLensLoader import convert_unique_idx

class DatasetLoader(object):
    def load(self):
        """Minimum condition for dataset:
          * All users must have at least one item record.
          * All items must have at least one user record.
        """
        raise NotImplementedError

class LibraryThing(DatasetLoader):
    def __init__(self, data_dir, ndatapoints):
        self.path = os.path.join(data_dir, 'reviews.txt')
        self.ndatapoints = ndatapoints
    
    def load(self):
        df = pd.DataFrame([], columns = ['comment','nhelpful', 'unixtime', 'work', 'flags', 'user', 'stars', 'time'])
        file = open(self.path, 'r')
        lines = file.readlines()[1:]

        extracted_data = []
        linecount = 0
        for line in lines:
            line = line.split('] = ')
            line = line[1]
            reviews = eval(line)
            linecount +=1
            extracted_data.append([reviews.get('comment', ''), reviews.get('nhelpful', '0'), reviews.get('unixtime', '0'), reviews.get('work', ''), reviews.get('flags', ''), reviews.get('user', ''), reviews.get('stars', '0'), reviews.get('time', '')])
        
            if linecount > self.ndatapoints:
                    break

        df = pd.DataFrame(extracted_data, columns=['comment', 'nhelpful', 'unixtime', 'work', 'flags', 'user', 'stars', 'time'])
        df['commentlength'] = df['comment'].str.split().apply(len)
        df.rename(columns = {'work':'item', 'stars':'rate'}, inplace = True)
        df['rate'] = df['rate'].astype(float)
        df['nhelpful'] = df['nhelpful'].astype(float)
        df['item'] = df['item'].astype(int)

        
        df, user_mapping = convert_unique_idx(df, 'user')
        df, item_mapping = convert_unique_idx(df, 'item')
        return df, item_mapping
