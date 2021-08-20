## The Effect of Domain and Diacritics in Yorùbá--English Neural Machine Translation

In this [paper](https://arxiv.org/abs/2103.08647), we present MENYO-20k,  the first multi-domain parallel corpus for Yorùbá--English (yo-en) language pair that can be used to benchmark MT systems. We also provide the code (or links to the code) to train the several neural MT (NMT) models we employed in the paper, and links to our trained MT models. 

MENYO-20k is a multi-domain parallel dataset with texts obtained from news articles, ted talks, movie transcripts, radio transcripts, science and technology texts, and other short articles curated from the web and professional translators.  The dataset has 20,100 parallel sentences split into 10,070 training sentences, 3,397 development sentences, and 6,633 test sentences (3,419 multi-domain, 1,714 news domain, and 1,500 ted talks speech transcript domain). See menyo_data_collection.pdf for the detailed description of the data collection. 

#### License
For non-commercial use because some of the data sources like [Ted talks](https://www.ted.com/about/our-organization/our-policies-terms/ted-talks-usage-policy) and [JW news](https://www.jw.org/en/terms-of-use/#link0) requires permission for commercial use. 

#### Paper:
We provide a detailed explanation of the dataset and some benchmark experiments in our [paper](https://arxiv.org/abs/2103.08647)

#### Models
* Fine-tuned MT5-base models
	* [EN-YO](https://huggingface.co/Davlan/mt5_base_eng_yor_mt)
	* [YO-EN](https://huggingface.co/Davlan/mt5_base_yor_eng_mt)
* Supervised
	* [EN-YO](https://drive.google.com/drive/folders/11AFrnCJ4JUbCwAHibBVG8pQQwM0SfXAH)
	* [YO-EN](https://drive.google.com/drive/folders/1oWUdYN38OcMfffQmaIJ4Sgi28R3KnFG4)
* Semi-supervised
	* [EN-YO](https://drive.google.com/drive/folders/1dXbBtilyd77SEH_bMbkVtO3Y5yE6W6c7)
	* [YO-EN](https://drive.google.com/drive/folders/1Pr24Ectz2iU1LtopTI6xIPG1h1PxXd9a)

Supervised and Semi-supervised are the models C4+Transfer and C4+Transfer+BT respectively. These two models were trained using [Fairseq](https://github.com/pytorch/fairseq). Therefore to generate translations using these models, you need to have installed Fairseq. 

```
CUDA_VISIBLE_DEVICES="$devices" fairseq-interactive \
	$DATADIR/$bpename \
	--path $model/$checkpoint \
	--beam 5 --source-lang $tgt --target-lang $src \
	--buffer-size 2048 \
	--max-sentences 64 \
	--remove-bpe
	< $BPEDIR/$bpename/test/test.$src-$tgt.$tgt.bpe \
	| grep -P "D-[0-9]+" | cut -f3 \
	> $evaldir/test.${tgt}2${src}.mtout
```

where `$src` and `$tgt` refers to the source and target languages respectively. And `test.$src-$tgt.$tgt.bpe` is the input file. The input file to the model should contain already pre-processed source language texts. We provided our [Truecase](https://drive.google.com/drive/folders/1zgXnGNfCFf-e7QSIeEylq_r2c5saOVtG) and [BPE](https://drive.google.com/drive/folders/1O3GcZFGEs5v91EYQuIkUIMDNYN9CuG4B) models for use. For more information on using the Fairseq framework, visit their github page. 

### Acknowledgement:

This project was partially supported by the [AI4D language dataset fellowship](https://www.k4all.org/project/language-dataset-fellowship/) through K4All and Zindi Africa

If you use this dataset, please cite this paper
```
@inproceedings{adelani-etal-2021-effect,
    title = "The Effect of Domain and Diacritics in {Y}oruba{--}{E}nglish Neural Machine Translation",
    author = "Adelani, David  and
      Ruiter, Dana  and
      Alabi, Jesujoba  and
      Adebonojo, Damilola  and
      Ayeni, Adesina  and
      Adeyemi, Mofe  and
      Awokoya, Ayodele Esther  and
      Espa{\~n}a-Bonet, Cristina",
    booktitle = "Proceedings of the 18th Biennial Machine Translation Summit (Volume 1: Research Track)",
    month = aug,
    year = "2021",
    address = "Virtual",
    publisher = "Association for Machine Translation in the Americas",
    url = "https://aclanthology.org/2021.mtsummit-research.6",
    pages = "61--75",
    abstract = "Massively multilingual machine translation (MT) has shown impressive capabilities and including zero and few-shot translation between low-resource language pairs. However and these models are often evaluated on high-resource languages with the assumption that they generalize to low-resource ones. The difficulty of evaluating MT models on low-resource pairs is often due to lack of standardized evaluation datasets. In this paper and we present MENYO-20k and the first multi-domain parallel corpus with a especially curated orthography for Yoruba{--}English with standardized train-test splits for benchmarking. We provide several neural MT benchmarks and compare them to the performance of popular pre-trained (massively multilingual) MT models both for the heterogeneous test set and its subdomains. Since these pre-trained models use huge amounts of data with uncertain quality and we also analyze the effect of diacritics and a major characteristic of Yoruba and in the training data. We investigate how and when this training condition affects the final quality of a translation and its understandability.Our models outperform massively multilingual models such as Google ($+8.7$ BLEU) and Facebook M2M ($+9.1$) when translating to Yoruba and setting a high quality benchmark for future research.",
}
```
