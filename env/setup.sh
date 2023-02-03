# Create environment
conda env create -f env/jme.yml

# Install Gdown
pip3 install gdown

# Download dataset Librarything
python3 src/datasets/download_librarything.py

python << END
print('------- JME-Fairness Reproducibility Study -------\n')
print('Hello, welcome to the JME Reproducability Study code repository\n\n\
To run all experiments:\n   \
python3 src/run_experiments.sh \n')
END