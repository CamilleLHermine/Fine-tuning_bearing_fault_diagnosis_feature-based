# Fine-tuning_bearing_fault_diagnosis_feature-based

This repository contains a PyTorch implementation of the feature-based Large Language Model (LLM) framework for bearing fault diagnosis, as proposed in the paper by Tao et al. (2025) published in Mechanical Systems and Signal Processing.
The notebook Finetuning_bearingfaultdiagnosis_CWRU_feature_based_FINAL.ipynb does a fine-tuning of the ChatGLM2-6B model on the CWRU bearing dataset using textualized vibration features.

Accurately diagnosing bearing faults is crucial for industrial maintenance. However, traditional methods struggle with varying operating conditions, limited data, and generalizing across different datasets. This project leverages the advanced sequential data processing capabilities of LLMs to create a more robust and adaptable fault diagnosis system.

This notebook specifically implements the feature-based LLM fault diagnosis approach detailed in the paper. The pipeline involves transforming raw vibration signals into structured text that an LLM can understand and learn from.


**The workflow is as follows:**
1.	Data Loading: The notebook loads the Case Western Reserve University (CWRU) Bearing Dataset.
2.	Signal Preprocessing: Raw vibration signals are segmented into windows of 2048 data points with a step size of 512 to create uniform samples.
3.	Feature Extraction: For each window, 24 distinct features are calculated2:
o	12 time-domain features (e.g., mean, standard deviation, kurtosis).
o	12 frequency-domain features (e.g., frequency mean, gravity frequency, spectral kurtosis).
4.	Feature Textualization: The 24 numerical features are converted into a natural language prompt. This "textualization" step frames the diagnosis task as a question-and-answer problem for the LLM.
   An example input is structured as:
"instruction": "You are a bearing fault diagnosis expert. Based on the following features, you need to conduct fault diagnosis:",
"input": "The time-domain mean of the vibration signal is... ",
"output": "There is an inner ring fault."

5.	LLM Fine-Tuning: The THUDM/chatglm2-6b model is fine-tuned using Low-Rank Adaptation (LoRA), an efficient technique that significantly reduces the number of trainable parameters. This allows the LLM to learn the patterns connecting the textualized features to specific fault conditions (Normal, Ball, Inner Race, Outer Race).



**Description of the Notebook key steps:**

_Data Science & Signal Processing:_
o	**Time-Series Analysis:** Processed and manipulated raw vibration signals from .mat files using NumPy.
o	**Feature Engineering**: Extracted 24 distinct time-domain and frequency-domain features (e.g., mean, standard deviation, kurtosis, spectral skewness) to create a robust representation of the signal state.
o	**Data Segmentation:** Implemented a sliding window technique to segment continuous time-series data into uniform samples for model training.
_Natural Language Processing (NLP) & LLMs:_
o	**Feature Textualization:** Engineered a method to convert numerical features into structured, descriptive natural language prompts, effectively bridging the gap between traditional signal processing and modern LLMs .
o	**LLM Fine-Tuning:** Leveraged the transformers library to fine-tune the ChatGLM2-6B model, a 6-billion parameter LLM, for a specialized sequence classification task.
o	**Parameter-Efficient Fine-Tuning :** Implemented Low-Rank Adaptation (LoRA) to efficiently train the LLM, reducing trainable parameters by over 99.9% while retaining performance. (memory-efficient finetuning strategy).
_•	Training :_
o	Utilized the Hugging Face Trainer API to manage the training pipeline, including setting TrainingArguments and defining performance metrics.
o	Developed custom Dataset classes in PyTorch to handle the tokenization, padding, and batching of textualized data for the LLM.


**How to Run:**
1. Setup
Clone this repository and install the required dependencies.
2. Download the CWRU Bearing Dataset. 
	The notebook expects the data in a zip file named CWRUDataset.zip with .mat files inside.
  Upload this zip file to your environment.
3. Execution
  The notebook will handle data extraction, preprocessing, feature calculation, and model training.


**Expected Results:**
This notebook performs a single-dataset training and evaluation task. According to the paper, the feature-based LLM approach achieves a final accuracy of 96.85% on the CWRU dataset after 10 epochs of training.
Note: The TrainingArguments in this notebook are set to num_train_epochs=1. For full replication, you may need to adjust this.
