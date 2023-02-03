# Create environment
conda env create -f env/jme.yml
conda activate jme
# Download dataset Librarything
python3 src/BERT4Rec/download_pretrained.py

python << END
print('------- JME-Fairness Reproducibility Study -------\n')
print('Hello, welcome to the JME Reproducability Study code repository\n\n\
To run all experiments:\n   \
python3 src/run_experiments.sh \n')
END