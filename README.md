# [ACL 2025] Hierarchical Safety Realignment: Lightweight Restoration of Safety in Pruned Large Vision-Language Models

Because the paper is still under double-blind review and the code needs time to sort out, the code file has not been uploaded for the time being.

## 🔔 News

- \[2025.05.16\]. Our work is accepted by ACL 2025 (Findings)!

## Environment Preparation

```python
conda create --name hsr python=3.9
conda activate hsr
pip install -r requirements.txt
```

## Citation
If you find our work useful, please consider citing our paper:
```

```

Our codebase is built upon on the following works:
```
@misc{zhou2025roleattentionheadslarge,
      title={On the Role of Attention Heads in Large Language Model Safety}, 
      author={Zhenhong Zhou and Haiyang Yu and Xinghua Zhang and Rongwu Xu and Fei Huang and Kun Wang and Yang Liu and Junfeng Fang and Yongbin Li},
      year={2025},
      eprint={2410.13708},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.13708}, 
}
```
```
@InProceedings{pmlr-v235-wei24f,
  title = 	 {Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications},
  author =       {Wei, Boyi and Huang, Kaixuan and Huang, Yangsibo and Xie, Tinghao and Qi, Xiangyu and Xia, Mengzhou and Mittal, Prateek and Wang, Mengdi and Henderson, Peter},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {52588--52610},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/wei24f/wei24f.pdf},
  url = 	 {https://proceedings.mlr.press/v235/wei24f.html},
  abstract = 	 {Large language models (LLMs) show inherent brittleness in their safety mechanisms, as evidenced by their susceptibility to jailbreaking and even non-malicious fine-tuning. This study explores this brittleness of safety alignment by leveraging pruning and low-rank modifications. We develop methods to identify critical regions that are vital for safety guardrails, and that are disentangled from utility-relevant regions at both the neuron and rank levels. Surprisingly, the isolated regions we find are sparse, comprising about $3$ % at the parameter level and $2.5$ % at the rank level. Removing these regions compromises safety without significantly impacting utility, corroborating the inherent brittleness of the model’s safety mechanisms. Moreover, we show that LLMs remain vulnerable to low-cost fine-tuning attacks even when modifications to the safety-critical regions are restricted. These findings underscore the urgent need for more robust safety strategies in LLMs.}
}
```
