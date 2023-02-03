# Create environment
conda env create -f env/jme.yml

# Install Gdown
pip3 install gdown

# Download dataset Librarything
python3 src/datasets/download_librarything.py

python << END
print('------- JME-Fairness Reproducibility Study -------\n')
print('Hello, welcome to the JME Reproducability Study code repository\n\n\
To plot all figures in our reproducibility study:\n   \
conda activate jme\n
bash experiments/run_files/plot_reproduction_results.sh \n')
END