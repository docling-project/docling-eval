## [v1.1.0](https://github.com/docling-project/docling-eval/releases/tag/v1.1.0) - 2026-04-13

### Feature

* Dev/add datasetrecord ([#207](https://github.com/docling-project/docling-eval/issues/207)) ([`412c43a`](https://github.com/docling-project/docling-eval/commit/412c43aa962065989d08591b8267bc939e78579f))
* Flat-layout CVAT campaign tools and resilient shard writing ([#206](https://github.com/docling-project/docling-eval/issues/206)) ([`5c9f3fa`](https://github.com/docling-project/docling-eval/commit/5c9f3fadf24224dba6a6061844bcb7099baaf6e8))
* New dataset builder - DoclingSDGDatasetBuilder ([#205](https://github.com/docling-project/docling-eval/issues/205)) ([`e761bcc`](https://github.com/docling-project/docling-eval/commit/e761bcc9ddfb71bb3c8208cd0071c06f50939b65))

### Fix

* PIL Image Memory Leaks in Dataset Builders ([#194](https://github.com/docling-project/docling-eval/issues/194)) ([`55fd3eb`](https://github.com/docling-project/docling-eval/commit/55fd3eb60474618c4ffad2cefbac3c9a3694d44f))

## [v1.0.1](https://github.com/docling-project/docling-eval/releases/tag/v1.0.1) - 2026-03-11

### Fix

* Remove hard pinning of docling-parse ([#203](https://github.com/docling-project/docling-eval/issues/203)) ([`901814d`](https://github.com/docling-project/docling-eval/commit/901814d35a5aec6cf9f4120a84a7716e058b926c))

## [v1.0.0](https://github.com/docling-project/docling-eval/releases/tag/v1.0.0) - 2026-03-11

### Feature

* Parallelize the evaluation of tables and cache the loading of external predictions ([#190](https://github.com/docling-project/docling-eval/issues/190)) ([`9d04a56`](https://github.com/docling-project/docling-eval/commit/9d04a56b936e0efd130653ab64a765f08020fcc5))
* Regression tests for CVAT to Docling conversion ([#193](https://github.com/docling-project/docling-eval/issues/193)) ([`8a10188`](https://github.com/docling-project/docling-eval/commit/8a101881774ea73b56a608090e5c9b51b43be966))
* CVAT box rotation support, structural cleanup ([#191](https://github.com/docling-project/docling-eval/issues/191)) ([`db068e9`](https://github.com/docling-project/docling-eval/commit/db068e9d88f4e08093af746ff8d89f81d57588ae))
* Improvements in user experience: Performance, error handling, logging ([#189](https://github.com/docling-project/docling-eval/issues/189)) ([`a850784`](https://github.com/docling-project/docling-eval/commit/a850784b4f9b26e94659085d6ea1f95473313f90))
* Visualizer tool and command for datasets ([#186](https://github.com/docling-project/docling-eval/issues/186)) ([`373f959`](https://github.com/docling-project/docling-eval/commit/373f959633077e0aa82b0970ccfd9f18c7d87292))
* Extend the evaluators to support external predictions stored in files ([#185](https://github.com/docling-project/docling-eval/issues/185)) ([`53dbd95`](https://github.com/docling-project/docling-eval/commit/53dbd955ae4da718ae419aae5e90963d8cb98b7f))
* Convert Docling JSON inputs to image streams in FileDatasetBuilder ([#184](https://github.com/docling-project/docling-eval/issues/184)) ([`15888fd`](https://github.com/docling-project/docling-eval/commit/15888fd25c6e7fe08ca04b07f0fd1fb8dd1c4e84))
* Allow subset to split routing in CVAT to HF exporter ([#182](https://github.com/docling-project/docling-eval/issues/182)) ([`ebb8800`](https://github.com/docling-project/docling-eval/commit/ebb88006413d7611299b1e7be85441ab966352eb))
* Ingest CVAT assets and filter submissions ([#180](https://github.com/docling-project/docling-eval/issues/180)) ([`b55b2ea`](https://github.com/docling-project/docling-eval/commit/b55b2ea40df8d1ffe18d644c8cee673336082a49))
* Runtime optimizations for MultiLabelConfusionMatrix ([#175](https://github.com/docling-project/docling-eval/issues/175)) ([`5084a4d`](https://github.com/docling-project/docling-eval/commit/5084a4d675d6cf2478b08b72fd3fbc9949c2de6d))
* Add more fine-grained control in the DoclingEvalCOCOExporter ([#149](https://github.com/docling-project/docling-eval/issues/149)) ([`8f33420`](https://github.com/docling-project/docling-eval/commit/8f33420d6a0d006fe521f5efabaa612b99699930))
* Remove legacy CvatDatasetBuilder code, use modernized code ([#174](https://github.com/docling-project/docling-eval/issues/174)) ([`693c224`](https://github.com/docling-project/docling-eval/commit/693c22445fff0704db86711b2d5a5a0477bf375b))
* Introduce the PixelLayoutEvaluator to produce confusion matrices for the multi-label layout analysis ([#173](https://github.com/docling-project/docling-eval/issues/173)) ([`a79bac5`](https://github.com/docling-project/docling-eval/commit/a79bac5d0265d054a16e32750cbedda18d1f2115))
* Review-bundle builder, fixes for GraphCell with merged elements and more ([#172](https://github.com/docling-project/docling-eval/issues/172)) ([`21341ce`](https://github.com/docling-project/docling-eval/commit/21341ce1bed43b337e3097449377dbf5368a082b))

### Fix

* Correct import path for TableStructureModel ([#199](https://github.com/docling-project/docling-eval/issues/199)) ([`a7e74a3`](https://github.com/docling-project/docling-eval/commit/a7e74a3b369f9ab8426818aa2591217d6b1b4df7))
* Fix the reporting of doc_id, true_md, pred_md in markdown_text_evaluator.py ([#196](https://github.com/docling-project/docling-eval/issues/196)) ([`3ce7591`](https://github.com/docling-project/docling-eval/commit/3ce75918724bf09a07d1ad2c82ae536e7fa63af2))
* PixelLayoutEvaluator: Set all-pixels background in case of a missing prediction and evaluate ([#183](https://github.com/docling-project/docling-eval/issues/183)) ([`4314091`](https://github.com/docling-project/docling-eval/commit/4314091abf4d1cdd4244ebb604579eba6bfffba8))
* Fix empty prediction handling in markdown evaluator ([#177](https://github.com/docling-project/docling-eval/issues/177)) ([`9b6df83`](https://github.com/docling-project/docling-eval/commit/9b6df83aea0bd9d2c2aac8d2ce6db2d5b126d64f))
* Consistenty and perf improvements ([#171](https://github.com/docling-project/docling-eval/issues/171)) ([`8fb3a16`](https://github.com/docling-project/docling-eval/commit/8fb3a169f6d45b055c1012a63fbdab667b704054))

### Breaking

* CvatDatasetBuilder now requires modern CVAT folder structure and uses convert_cvat_folder_to_docling() internally. ([`693c224`](https://github.com/docling-project/docling-eval/commit/693c22445fff0704db86711b2d5a5a0477bf375b))

## [v0.10.0](https://github.com/docling-project/docling-eval/releases/tag/v0.10.0) - 2025-11-05

### Feature

* Extend the CLI for create-eval to receive the vlm-options and max_new_tokens parameters when the provider is GraniteDocling ([#164](https://github.com/docling-project/docling-eval/issues/164)) ([`8be2e83`](https://github.com/docling-project/docling-eval/commit/8be2e8399b9fdefc3a9997176cc4f6d54024b39b))
* Harmonizing pic classes for cvat to docling conversion ([#167](https://github.com/docling-project/docling-eval/issues/167)) ([`740157d`](https://github.com/docling-project/docling-eval/commit/740157dba3d5f239e792029d3f2df6cb5b855368))
* Add more specific validation for reading-order, enhance validation report ([`5e5f2db`](https://github.com/docling-project/docling-eval/commit/5e5f2dbb369957921818042d369815777dcfde1f))
* Integrate textline_cells based OCR evaluation ([#156](https://github.com/docling-project/docling-eval/issues/156)) ([`3a9543c`](https://github.com/docling-project/docling-eval/commit/3a9543c865e31178306b5552102ab012c72c8583))

### Fix

* Validation fixes for list item impurity check ([#169](https://github.com/docling-project/docling-eval/issues/169)) ([`74e7b3e`](https://github.com/docling-project/docling-eval/commit/74e7b3e7dec671083cae15d98ec46725f03e3894))
* Don't report content-layer group violation multiple times ([`cb71009`](https://github.com/docling-project/docling-eval/commit/cb71009009aefb4175bb358f68fffdeb67dc34a2))
* Handle merged elements regarding inclusion, don't flag single element pages ([`c10fdfd`](https://github.com/docling-project/docling-eval/commit/c10fdfd8c46245376f64a792e481f7d9a6400179))
* Missing transform to storage_scale for some items and table cells ([`1eb6b4e`](https://github.com/docling-project/docling-eval/commit/1eb6b4ea766dfd38479fdc577907132dfbdbdf91))
* More CVAT validation and docling conversion fixes ([#163](https://github.com/docling-project/docling-eval/issues/163)) ([`6f59c7a`](https://github.com/docling-project/docling-eval/commit/6f59c7a8afadcd51816abdea16b113ee00e86229))
* Better control over scaling in CVAT transform, fixes for OCR ([#162](https://github.com/docling-project/docling-eval/issues/162)) ([`ef17b5a`](https://github.com/docling-project/docling-eval/commit/ef17b5a30b1727676c915e6d49cd5f28c348b8d8))
* Fixes for CVAT validation, OCR in CVAT pipeline, logging, and more ([#161](https://github.com/docling-project/docling-eval/issues/161)) ([`80e449d`](https://github.com/docling-project/docling-eval/commit/80e449de7ff0042a79707c9c1f4c59f45cbff269))

### Performance

* Consistenty and perf improvements ([#170](https://github.com/docling-project/docling-eval/issues/170)) ([`d4a0ef6`](https://github.com/docling-project/docling-eval/commit/d4a0ef619e1333163d2041aa811b0ce5d9a24715))

## [v0.9.0](https://github.com/docling-project/docling-eval/releases/tag/v0.9.0) - 2025-10-01

### Feature

* Exposed forced-ocr-option ([#157](https://github.com/docling-project/docling-eval/issues/157)) ([`ac21644`](https://github.com/docling-project/docling-eval/commit/ac21644d75e43c0814d3e0c4261d293b7825527f))
* Implementation of table structure conversion from CVAT to DoclingDocument ([`208cd14`](https://github.com/docling-project/docling-eval/commit/208cd14bdd41ca9403ec82a33d68b7526890c668))

## [v0.8.1](https://github.com/docling-project/docling-eval/releases/tag/v0.8.1) - 2025-09-16

### Fix

* Ocr visualization and add ocr recognition metrics ([#144](https://github.com/docling-project/docling-eval/issues/144)) ([`d63a439`](https://github.com/docling-project/docling-eval/commit/d63a439441ff8c3f8c51f0d442e2c352f8bbc8dc))

## [v0.8.0](https://github.com/docling-project/docling-eval/releases/tag/v0.8.0) - 2025-09-03



## [v0.7.0](https://github.com/docling-project/docling-eval/releases/tag/v0.7.0) - 2025-07-30

### Feature

* Add CLI arguments to control the docling layout model ([#136](https://github.com/docling-project/docling-eval/issues/136)) ([`3e134ae`](https://github.com/docling-project/docling-eval/commit/3e134ae1b08f82e9e6ecb9690b73a9420a528fb1))
* Campaign tools ([#139](https://github.com/docling-project/docling-eval/issues/139)) ([`af2c222`](https://github.com/docling-project/docling-eval/commit/af2c222af0bdf93230fa4e619dc45e388d48e7a5))
* Add KeyValueEvaluator ([#140](https://github.com/docling-project/docling-eval/issues/140)) ([`bc60093`](https://github.com/docling-project/docling-eval/commit/bc600938fc3452d0bffdd835bca420538c9f2fea))

### Fix

* Prevent crash from invalid bbox coordinates in HTML export ([#142](https://github.com/docling-project/docling-eval/issues/142)) ([`c31b107`](https://github.com/docling-project/docling-eval/commit/c31b107298f625721ab98aaac54f56d8c3f87a68))

## [v0.6.0](https://github.com/docling-project/docling-eval/releases/tag/v0.6.0) - 2025-07-02

### Feature

* Layout evaluation fixes, mode control and cleanup ([#133](https://github.com/docling-project/docling-eval/issues/133)) ([`629a451`](https://github.com/docling-project/docling-eval/commit/629a451d7b75e274352a1f21710316e47fc7a80a))
* Introduce utility to export layout predictions from HF parquet files into pycocotools format. ([#125](https://github.com/docling-project/docling-eval/issues/125)) ([`54f7c81`](https://github.com/docling-project/docling-eval/commit/54f7c81f8ad28b848372c4961a4f4b83763ffebe))
* Add specific language support for XFUND dataset builder ([#122](https://github.com/docling-project/docling-eval/issues/122)) ([`4ca6a0e`](https://github.com/docling-project/docling-eval/commit/4ca6a0e2ddb63d30d204c30549ec4bc56abbb972))
* Tooling for CVAT validation, to DoclingDocument transformation, new Evaluators ([#119](https://github.com/docling-project/docling-eval/issues/119)) ([`2ee1104`](https://github.com/docling-project/docling-eval/commit/2ee11049d7da313206f08e4e1a7adf20c4d27459))

### Fix

* Move ibm-cos to hyperscaler ([#135](https://github.com/docling-project/docling-eval/issues/135)) ([`9aff6c1`](https://github.com/docling-project/docling-eval/commit/9aff6c1a6a04f0b6d54ed9fd94207263452d35c5))
* Update hyperscalers to support multiple image file types ([#118](https://github.com/docling-project/docling-eval/issues/118)) ([`a34f264`](https://github.com/docling-project/docling-eval/commit/a34f2649abd01671b5da9a44d546e010d73b0d60))
* Misc fixes ([#131](https://github.com/docling-project/docling-eval/issues/131)) ([`518e1ba`](https://github.com/docling-project/docling-eval/commit/518e1ba342bee819d74f0bad266013074af052dd))
* **CVAT to DoclingDoc:** Ensure that nested list handling works across page boundaries ([#129](https://github.com/docling-project/docling-eval/issues/129)) ([`1b58377`](https://github.com/docling-project/docling-eval/commit/1b583779e73892b2a36aa54829f69c85928c6dc2))
* Important fixes for parquet serialization / deserialization, optimizations ([#128](https://github.com/docling-project/docling-eval/issues/128)) ([`53c22ef`](https://github.com/docling-project/docling-eval/commit/53c22efe749bcdfe8708b02ea56109de20ff124f))
* Fixes for the dataset visualizers ([#127](https://github.com/docling-project/docling-eval/issues/127)) ([`a127ea9`](https://github.com/docling-project/docling-eval/commit/a127ea9424d711b29bf1399aa3caec68d3ebfee1))

### Performance

* Improve parquet writing with plain pyarrow ([#134](https://github.com/docling-project/docling-eval/issues/134)) ([`c08950b`](https://github.com/docling-project/docling-eval/commit/c08950b4969748aa5a689a8e2ab0c51b658582db))

## [v0.5.0](https://github.com/docling-project/docling-eval/releases/tag/v0.5.0) - 2025-06-11

### Feature

* Integrate OCR visualization ([#121](https://github.com/docling-project/docling-eval/issues/121)) ([`b39f2e7`](https://github.com/docling-project/docling-eval/commit/b39f2e7932b4ed9b9a08ba0dda2be6af9d59daff))
* Add the segmentation layout evaluations in the consolidated excel report. Update mypy overrides. ([#120](https://github.com/docling-project/docling-eval/issues/120)) ([`c4e7de0`](https://github.com/docling-project/docling-eval/commit/c4e7de0c1777f86e68b7a3b6db6b2f56ab3ba127))
* Update OCREvaluator with additional metrics ([#78](https://github.com/docling-project/docling-eval/issues/78)) ([`17e9fde`](https://github.com/docling-project/docling-eval/commit/17e9fde84f4b01564d4a838443d876890948312c))

### Fix

* Add the bbox to TableData from annotations ([#123](https://github.com/docling-project/docling-eval/issues/123)) ([`c4fe51f`](https://github.com/docling-project/docling-eval/commit/c4fe51f46161305076269dda4291636690b78a60))
* Treat th and td as equal for TEDS calculation ([#114](https://github.com/docling-project/docling-eval/issues/114)) ([`dbf9db7`](https://github.com/docling-project/docling-eval/commit/dbf9db77349aa845b9cd5d7f337e91e53515cbaa))
* Add support for Google, AWS, and Azure prediction providers in cli ([#115](https://github.com/docling-project/docling-eval/issues/115)) ([`e8e7421`](https://github.com/docling-project/docling-eval/commit/e8e7421a9a830bbd15774ee9d26e98296f9dbd2c))

## [v0.4.0](https://github.com/docling-project/docling-eval/releases/tag/v0.4.0) - 2025-05-28

### Feature

* Extend the FileProvider and the CLI to accept parameters that control the source of the  prediction images ([#111](https://github.com/docling-project/docling-eval/issues/111)) ([`42e1615`](https://github.com/docling-project/docling-eval/commit/42e16152c55d1676214ef1fb1378975c67771f3b))
* Improvements for the MultiEvaluator ([#95](https://github.com/docling-project/docling-eval/issues/95)) ([`04fe2d9`](https://github.com/docling-project/docling-eval/commit/04fe2d916fbc5da915cfd5c53ebd322086f21a7f))
* Add extra args for docling-provider and default annotations for CVAT ([#98](https://github.com/docling-project/docling-eval/issues/98)) ([`7903b6a`](https://github.com/docling-project/docling-eval/commit/7903b6a1d9f3754a5283fcf567bdadb613348cf4))
* Introduce SegmentedPage for OCR ([#91](https://github.com/docling-project/docling-eval/issues/91)) ([`be0ff6a`](https://github.com/docling-project/docling-eval/commit/be0ff6a80c29dd2a0662adab1c348ed90c0e654a))
* Update CVAT for multi-page annotation, utility to create sliced PDFs ([#90](https://github.com/docling-project/docling-eval/issues/90)) ([`28d166d`](https://github.com/docling-project/docling-eval/commit/28d166d53100e285108bb35f139ee562ad5ccd93))
* Add area level f1 ([#86](https://github.com/docling-project/docling-eval/issues/86)) ([`54d013b`](https://github.com/docling-project/docling-eval/commit/54d013bc5e554c48974fb26f32176d264977c6cd))

### Fix

* Small fixes ([#108](https://github.com/docling-project/docling-eval/issues/108)) ([`0628fa6`](https://github.com/docling-project/docling-eval/commit/0628fa6c404dae780f0952835c99a6cbb3e01029))
* Layout text not correctly populated in AWS prediction provider, add tests ([#100](https://github.com/docling-project/docling-eval/issues/100)) ([`6441688`](https://github.com/docling-project/docling-eval/commit/6441688eb3c8e2c85ab73d22c15345323df53e72))
* Dataset feature spec fixes, cvat improvements ([#97](https://github.com/docling-project/docling-eval/issues/97)) ([`b79dd19`](https://github.com/docling-project/docling-eval/commit/b79dd1988cb391cc256d3a373551528e44618301))
* Update boto3 AWS client to accept service credentials ([#88](https://github.com/docling-project/docling-eval/issues/88)) ([`4e01d0b`](https://github.com/docling-project/docling-eval/commit/4e01d0bbe5c86700f65f1671802669d851f64612))
* Handle unsupported END2END evaluation and fix variable name in OCR ([#87](https://github.com/docling-project/docling-eval/issues/87)) ([`75311da`](https://github.com/docling-project/docling-eval/commit/75311da9bf480c12f70d4b1b150579a7746cf514))
* Propagate cvat parameters ([#82](https://github.com/docling-project/docling-eval/issues/82)) ([`1e2040a`](https://github.com/docling-project/docling-eval/commit/1e2040a6293c2f157ae2214ab8d650669b6fbbf0))

### Documentation

* Update README.md ([#84](https://github.com/docling-project/docling-eval/issues/84)) ([`518f684`](https://github.com/docling-project/docling-eval/commit/518f684fb5f3bf89a214bce162e61cb81e272f95))
