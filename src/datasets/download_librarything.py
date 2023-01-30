import gdown
import tarfile 
import os

if os.path.exists('lt/reviews.txt'):
    print('Dataset Librarything Already Downloaded ....')
else:
    print('Downloading Librarything Dataset')
    url = 'https://drive.google.com/u/0/uc?id=1wgXv14TyrRD5DH6ZfJqApFymQv7_vuho&export=download'
    output = 'lthing_data.tar.gz'
    gdown.download(url, output, quiet = False)

    mytar = tarfile.open(output)
    mytar.extractall('./lt')
    mytar.close()

    os.rename('lt/lthing_data/reviews.txt', 'lt/reviews.txt')
    os.remove('lt/lthing_data/edges.txt')
    os.rmdir('lt/lthing_data')
    os.remove('lthing_data.tar.gz')