from typing import Optional


class EvaluationError(Exception):
    r"""
    Evaluation error class
    """

    def __init__(self, msg: Optional[str] = None):
        Exception.__init__(self, msg)
