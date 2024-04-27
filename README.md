# Large Language Models are Interpreters of Text Clustering Results
This is a course project for IS 557 Applied Machine Learning: Team Project.

The [dataset](data/mtop.csv) used for this project is based on the ClusterLLM (Zhang et al., 2023), which preprocessed the MTOP dataset (Li et al., 2021) by removing intents with only a few instances and keeping English-only data. Only the music domain has been selected for further experimentation which consists of 1341 samples and 24 intentions.

The `src` folder contains four *Python* scripts:

- [1_baselines.py](src/1_baselines.py)

- [2_1_musician_ds_one_model.py](src/2_1_musician_ds_one_model.py)

- [2_2_musician_ds_multi_models.py](src/2_2_musician_ds_multi_models.py)

- [3_llama3_mistral.py](src/3_llama3_mistral.py)

**Environment setup**

1. Download [Ollama](https://ollama.com/download) and install pre-trained LLMs: 
```
# go to the command line and run the following commands
# to pull pre-trained LLMs to your local machine
ollama run llama3
ollama run mistral
``` 

2. Install [brew](https://brew.sh/)

3. Install *miniconda* (restart your terminal afterwards)
```
brew install --cask miniconda
conda init zsh # (or conda init bash)
```

3. Setup *Python* virtual environment
```
conda env create -f environment.yml -n is577
conda activate is577
```

4. Run scripts
```
# Pick the script
python src/1_baselines.py
```

References

Yuwei Zhang, Zihan Wang, and Jingbo Shang. 2023. ClusterLLM: Large Language Models as a Guide for Text Clustering. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 13903–13920, Singapore. Association for Computational Linguistics.

Haoran Li, Abhinav Arora, Shuohui Chen, Anchit Gupta, Sonal Gupta, and Yashar Mehdad. 2021. MTOP: A comprehensive multilingual task-oriented semantic parsing benchmark. In *Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume*, pages 2950–2962, Online. Association for Computational Linguistics.
