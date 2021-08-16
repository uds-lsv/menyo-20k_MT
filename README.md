## The Effect of Domain and Diacritics in Yorùbá--English Neural Machine Translation

In this [paper](https://arxiv.org/abs/2103.08647), we present MENYO-20k,  the first multi-domain parallel corpus for Yorùbá--English (yo-en) language pair that can be used to benchmark MT systems. We also provide the code (or links to the code) to train the several neural MT (NMT) models we employed in the paper, and links to our trained MT models. 

MENYO-20k is a multi-domain parallel dataset with texts obtained from news articles, ted talks, movie transcripts, radio transcripts, science and technology texts, and other short articles curated from the web and professional translators.  The dataset has 20,100 parallel sentences split into 10,070 training sentences, 3,397 development sentences, and 6,633 test sentences (3,419 multi-domain, 1,714 news domain, and 1,500 ted talks speech transcript domain). See menyo_data_collection.pdf for the detailed description of the data collection. 

#### License
For non-commercial use because some of the data sources like [Ted talks](https://www.ted.com/about/our-organization/our-policies-terms/ted-talks-usage-policy) and [JW news](https://www.jw.org/en/terms-of-use/#link0) requires permission for commercial use. 

#### Paper:
We provide a detailed explanation of the dataset and some benchmark experiment in our [paper](https://arxiv.org/abs/2103.08647)

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

where `$rc` and `$tgt` refers to the source and target languages respectively. And `test.$src-$tgt.$tgt.bpe` is the input file. The input file to the model should contain already pre-processed source language texts. We provided our [Truecase](https://drive.google.com/drive/folders/1zgXnGNfCFf-e7QSIeEylq_r2c5saOVtG) and [BPE](https://drive.google.com/drive/folders/1O3GcZFGEs5v91EYQuIkUIMDNYN9CuG4B) models for use. For more information on using the Fairseq framework, visit the github page. 

### Acknowledgement:

This project was partially supported by the [AI4D language dataset fellowship](https://www.k4all.org/project/language-dataset-fellowship/) through K4All and Zindi Africa

If you use this dataset, please cite this paper
```
@inproceedings{adelani_menyo20k,
    author = {Adelani, David and Ruiter, Dana and Alabi, Jesujoba and Adebonojo, Damilola and Ayeni, Adesina and Adeyemi, Mofe and Awokoya, Ayodele and España-Bonet, Cristina},
    title = {The Effect of Domain and Diacritics in Yoruba--English Neural Machine Translation},
    booktitle = {Proceedings of the 18th Biennial Machine Translation Summit. Machine Translation Summit (MT Summit-2021), located at Conference of the Association for Machine Translation in the Americas, August 16-20, Orlando, Florida, United States},
    year = {2021},
    publisher = {...}
}
```
