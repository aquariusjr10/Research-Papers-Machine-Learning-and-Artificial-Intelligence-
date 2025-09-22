# 📚 Machine Learning Papers Re-Implemented

# 

# This repository is a personal learning project where I re-implement landmark machine learning research papers from scratch.

# The goal is to deepen my understanding of both foundational and cutting-edge concepts in machine learning, natural language processing, and computer vision.

# 

# 📝 Papers Included

# 1\. Attention Is All You Need (2017)

# 

# Introduced the Transformer architecture, which replaced recurrent and convolutional models for sequence transduction.

# 

# Key Contributions: Scaled Dot-Product Attention, Multi-Head Attention, Positional Encoding.

# 

# 2\. Handwritten Digit Recognition with a Back-Propagation Network (1989)

# 

# Early application of neural networks for digit recognition (precursor to MNIST benchmarks).

# 

# Key Contributions: Demonstrated backpropagation’s potential in supervised learning tasks.

# 

# 3\. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT, 2021)

# 

# Proposed Vision Transformers (ViTs) for image classification.

# 

# Key Contributions: Split images into patches, treated as tokens, processed with Transformer architecture.

# 

# 4\. LoRA: Low-Rank Adaptation of Large Language Models (2021)

# 

# Lightweight fine-tuning method for large models.

# 

# Key Contributions: Adaptation via low-rank decomposition, reducing memory and compute costs.

# 

# 5\. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG, 2021)

# 

# Combined retrieval with generation for knowledge-heavy tasks.

# 

# Key Contributions: Dense passage retrieval + generative models for improved factuality and grounding.

# 

# 🎯 Objectives

# 

# Reproduce experiments and results from each paper.

# 

# Compare performance against reported baselines.

# 

# Build a stronger intuition for architecture design choices.

# 

# Document implementation details and challenges for each paper.

# 

# 📂 Repository Structure

├── 1 - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks/

│   └── implementation.ipynb

├── 2 - LORA (LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS)/

│   └── implementation.ipynb

├── 3 - An Image is Worth 16 X 16 Words/

│   └── implementation.ipynb

├── 4 - Handwritten Digit Recognition with a Back-Propagation Network/

│   └── implementation.ipynb

├── 5 - Attention is all you need/

│   └── implementation.ipynb

└── README.md



⚙️ Requirements



Python 3.9+



PyTorch / TensorFlow (depending on the experiment)



Hugging Face Transformers (for LLMs and RAG)



NumPy, Pandas, Matplotlib, Scikit-learn



You can install the dependencies with:

pip install -r requirements.txt



🚀 Usage



Each subdirectory contains:



Paper-specific implementation (.ipynb or .py)



Documentation on methodology



Experiments and results



Run any notebook directly to reproduce experiments.



📌 Notes



This project is for self-learning and educational purposes.



The implementations may not be fully optimized but are focused on clarity and correctness.



Contributions, discussions, and suggestions are always welcome.



📖 References



Vaswani et al., Attention Is All You Need (2017)



LeCun et al., Handwritten Digit Recognition with a Back-Propagation Network (1989)



Dosovitskiy et al., An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (2021)



Hu et al., LoRA: Low-Rank Adaptation of Large Language Models (2021)



Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2021)

