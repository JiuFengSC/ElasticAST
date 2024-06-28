# ElasticAST
This is the repository of the INTERSPEECH 2024 paper [ElasticAST: An Audio Spectrogram Transformer for All Length and Resolutions](https://github.com/JiuFengSC/ElasticAST).


This paper introduces an approach that enables the use of variable-length audio inputs with AST models during both training and inference. By employing sequence packing, our method ElasticAST, accommodates any audio length during training, thereby offering flexibility across all lengths and resolutions at the inference. This flexibility allows ElasticAST to maintain evaluation capabilities at various lengths or resolutions and achieve similar performance to standard ASTs trained at specific lengths or resolutions.


<div align=center>
<img width="300" alt="image" src="https://github.com/JiuFengSC/ElasticAST/blob/main/assets/ElasticAST.png?raw=true">
</div>

# Evaluation


The evaluation code has been released (the jupyter notebooks in `./src/`), you can use the weights below to reproduce the results.

The training code will be released soon.


# Pretrained Weights

The weights can be accessed by Google Drive, click the link to download.

1. [ElasctiAST_VoxCeleb](https://drive.google.com/file/d/1Sl5svJVQyICzKBQIrVoINrklFaaq86X0/view?usp=sharing)
2. [ElasctiAST_EpicSound](https://drive.google.com/file/d/1DNk9Bzwk8TqTBOmNFT0AUBREtxicQq_M/view?usp=sharing)
3. [ElasticAST_AudioSet](https://drive.google.com/file/d/1AXhKdBbtD8R1Ie68LNp3pLauIKk4Cs6o/view?usp=sharing)
4. [ElasctiAST_VGGSound](https://drive.google.com/file/d/15sCRT-h4PivlwzlmquPXHldebuRNJmEH/view?usp=sharing)


# Environment

```
conda create -n elasticast
pip install -r requirements.txt
```

# Citation
Please cite our paper if you find this repository useful. 
```
@inproceedings{feng2024elasticast,
  author={Feng, Jiu and Erol, Mehmet Hamza and Chung, Joon Son and Senocak, Arda},
  title={ElasticAST: An Audio Spectrogram Transformer for All Length and Resolutions},
  year=2024,
  booktitle={Proc. Interspeech 2024},
}
```