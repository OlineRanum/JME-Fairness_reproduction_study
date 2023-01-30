# Download dataset Librarything
python3 src/datasets/download_librarything.py

# Create environment
conda env create -f fact.yml

# Activate environment 
conda activate fact 

python << END
print('------- JME-Fairness Reproducability Study -------\n')
print('Hello, welcome to the JME Reproducability Study code repository\n\n\
To run all experiments:\n   \
python3 src/run_experiments.sh \n')
END