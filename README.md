# 📚 Machine Learning Papers Re-Implemented


This repo is basically me going down the rabbit hole of famous machine learning papers 🐇📄.
Instead of just reading them, I’m re-implementing everything myself — experiments, code, and (hopefully) reproducing some results.

Expected Outcomes:

✅ learning by doing

✅ building intuition

✅ having fun breaking/rebuilding cool ML ideas

#🔬 Papers I’m tackling

Attention is All You Need (2017)
Transformers before they were cool. Multi-head attention, positional encodings, the whole deal.

Handwritten Digit Recognition with a Back-Propagation Network (1989)
Old-school LeCun vibes 🕹️. The OG neural net that paved the way for MNIST benchmarks.

An Image is Worth 16x16 Words (ViT, 2021)
Cut images into patches → treat them like tokens → run a Transformer → magic ✨.

LoRA: Low-Rank Adaptation of LLMs (2021)
Fine-tune giant models without setting your GPU on fire 🔥. Low-rank matrices save the day.

RAG: Retrieval-Augmented Generation (2021)
What if your model could Google stuff before answering? That’s basically RAG.

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




#⚙️ Requirements


Python 3.9+



PyTorch / TensorFlow (depending on the experiment)



Hugging Face Transformers (for LLMs and RAG)



NumPy, Pandas, Matplotlib, Scikit-learn



You can install the dependencies with:

pip install -r requirements.txt



#🚀 Usage


Each subdirectory contains:

Paper-specific implementation (.ipynb or .py)

Documentation on methodology

Experiments and results

Run any notebook directly to reproduce experiments.

#📌 Notes
This project is for self-learning and educational purposes.

The implementations may not be fully optimized but are focused on clarity and correctness.

Contributions, discussions, and suggestions are always welcome.


#📖 References

Vaswani et al., Attention Is All You Need (2017)



LeCun et al., Handwritten Digit Recognition with a Back-Propagation Network (1989)



Dosovitskiy et al., An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (2021)



Hu et al., LoRA: Low-Rank Adaptation of Large Language Models (2021)



Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2021)

