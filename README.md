# DRAGON RoBERTa Large Mixed-domain Algorithm

Adaptation of [DRAGON baseline](https://github.com/DIAGNijmegen/dragon_baseline) (version `0.2.1`), with pretrained foundational model `joeranbosma/dragon-roberta-large-mixed-domain`.

For details on the pretrained foundational model, check out HuggingFace: [huggingface.co/joeranbosma/dragon-roberta-large-mixed-domain](https://huggingface.co/joeranbosma/dragon-roberta-large-mixed-domain).

This algorithm is used for the pretraining experiment in the DRAGON manuscript [1], according to the pre-specified statistical analysis plan [2]. See [dragon.grand-challenge.org/manuscript/](https://dragon.grand-challenge.org/manuscript/) for the latest info on the DRAGON manuscript.

The following adaptations were made to the DRAGON baseline:

```python
model_name = "joeranbosma/dragon-roberta-large-mixed-domain"
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
gradient_checkpointing = False
max_seq_length = 512
learning_rate = 1e-05
```


**References:**

[1] J. S. Bosma, K. Dercksen, L. Builtjes, R. André, C, Roest, S. J. Fransen, C. R. Noordman, M. Navarro-Padilla, J. Lefkes, N. Alves, M. J. J. de Grauw, L. van Eekelen, J. M. A. Spronck, M. Schuurmans, A. Saha, J. J. Twilt, W. Aswolinskiy, W. Hendrix, B. de Wilde, D. Geijs, J. Veltman, D. Yakar, M. de Rooij, F. Ciompi, A. Hering, J. Geerdink, and H. Huisman on behalf of the DRAGON consortium. The DRAGON benchmark for clinical NLP. *npj Digital Medicine* 8, 289 (2025). [https://doi.org/10.1038/s41746-025-01626-x](https://doi.org/10.1038/s41746-025-01626-x)

[2] J. S. Bosma, K. Dercksen, L. Builtjes, R. André, C, Roest, S. J. Fransen, C. R. Noordman, M. Navarro-Padilla, J. Lefkes, N. Alves, M. J. J. de Grauw, L. van Eekelen, J. M. A. Spronck, M. Schuurmans, A. Saha, J. J. Twilt, W. Aswolinskiy, W. Hendrix, B. de Wilde, D. Geijs, J. Veltman, D. Yakar, M. de Rooij, F. Ciompi, A. Hering, J. Geerdink, H. Huisman, DRAGON Consortium (2024). DRAGON Statistical Analysis Plan (v1.0). Zenodo. https://doi.org/10.5281/zenodo.10374512
