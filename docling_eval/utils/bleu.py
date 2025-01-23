# from nltk.translate.bleu_score import corpus_bleu


# def compute_bleu_scores(
#     targets: list[list[str]], predictions: list[list[str]]
# ) -> tuple[list[float], float]:
#     r"""
#     Compute the BLEU score for the given targets and predictions.

#     Parameters
#     ----------
#     targets : List[List[str]]
#         The ground truth target sequences.
#     predictions : List[List[str]]
#         The predicted sequences. Each prediction is a list of tokens.

#     Returns
#     -------
#     Tuple[List[float], float]
#         A tuple containing:
#         - bleu_scores (List[float]): The BLEU scores of the predictions.
#         - avg_bleu_score (float): The average BLEU score across all predictions.
#     """
#     weights = (0.25, 0.25, 0.25, 0.25)

#     bleu_scores = [
#         corpus_bleu([[tg]], [pred], weights=weights)
#         for tg, pred in zip(targets, predictions)
#     ]
#     return bleu_scores, sum(bleu_scores) / len(bleu_scores)


# def compute_bleu_score(target: list[str], prediction: list[str]) -> float:
#     r"""
#     Compute the BLEU score for the given targetand prediction text.

#     Parameters
#     ----------
#     targets : List[List[str]]
#         The ground truth target sequences.
#     predictions : List[List[str]]
#         The predicted sequences. Each prediction is a list of tokens.

#     Returns
#     -------
#     bleu_score
#     """
#     weights = (0.25, 0.25, 0.25, 0.25)

#     # reference: Ground truth (in BLEU there can be many references)
#     # hypothesis: prediction
#     bleu = corpus_bleu([[target]], [prediction], weights=weights)
#     return bleu
