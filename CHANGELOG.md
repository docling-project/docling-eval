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
