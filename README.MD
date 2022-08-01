## Implementation of the paper ["Extracting Training Data from Large Language Models"(Carlini et al, 2020)](https://arxiv.org/abs/2012.07805)

### How to Run

- (Optional) Change model type and hyperparameters at `config.yaml`
- Run `python inference.py` for text sampling from the victim language model
- Run `python rerank.py` to retrieve possibly memorized text sequence candidates

### References

- [Authors' Implementation](https://github.com/ftramer/LM_Memorization)
- [Revised Implementation on Sampling Method and on Metrics by @shreyansh26](https://github.com/shreyansh26/Extracting-Training-Data-from-Large-Langauge-Models)

### Contribution

- [ ] Supports T5 Encoder-Decoder as the victim model
- [x] Prevents oversampling during the prefix selection
- [ ] Speeds up the inference with parallel Multi-GPU usage
- [ ] Speeds up the reranking with parallel Multi-GPU usage
- [x] Clears up GPU VRAM memory usage after the corresponding task
- [x] Rules out 'low-quality repeated generations' with repetition penalty and with ngram restriction
