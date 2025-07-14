from functools import partial
import pandas as pd
import random
import re

random.seed(45)
from typing import List, Dict
from multiprocessing import Pool, cpu_count
import argparse
from pathlib import Path


class RegexTokenizer:
    def __init__(self, tokens: List[str]):
        # Escape tokens to ensure special characters are treated as literals
        escaped_tokens = [
            re.escape(token) for token in sorted(tokens, key=len, reverse=True)
        ]
        # Create a regex pattern that matches any of the tokens
        self.pattern = re.compile("|".join(escaped_tokens))
        # Create mapping dictionaries
        self.token_to_id = {token: idx for idx, token in enumerate(tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def encode(self, text: str) -> List[int]:
        matches = self.pattern.findall(text)
        return [
            self.token_to_id[match] for match in matches if match in self.token_to_id
        ]

    def decode(self, token_ids: List[int]) -> str:
        return "".join(
            [
                self.id_to_token[token_id]
                for token_id in token_ids
                if token_id in self.id_to_token
            ]
        )

    def tokenize(self, text: str) -> List[str]:
        tokens = self.pattern.findall(text)
        return tokens

tokens = pd.read_csv("docling_eval/end_to_end/equation_config/tokens.csv")
all_tokens = set(tokens["token"].tolist())
tokens_to_keep = set(tokens.loc[tokens["policy"].isin([0])]["token"].tolist())
tokens_to_drop = set(tokens.loc[tokens["policy"] == 1]["token"].tolist())
tokens_to_replace = dict(
    zip(
        tokens.loc[tokens["policy"] == 2, "token"],
        tokens.loc[tokens["policy"] == 2, "replace"].fillna(""),
    )
)
remove_with_braces_tokens = set(tokens.loc[tokens["policy"] == 3]["token"].tolist())
one_braces_tokens = set(tokens.loc[tokens["policy"] == 4]["token"].tolist())
two_braces_tokens = set(tokens.loc[tokens["policy"] == 5]["token"].tolist())
remove_with_braces_tokens_keep_tokens = set(
    tokens.loc[tokens["policy"] == 6]["token"].tolist()
)
remove_content_tokens = set(tokens.loc[tokens["policy"] == 7]["token"].tolist())
tokenizer = RegexTokenizer(list(all_tokens))


def handle_spaces(tokenized_equation):
    in_text = False
    braces_counter = 0
    new_equation = []
    for token in tokenized_equation:
        if not in_text and token == " ":
            continue

        if token == r"\text":
            in_text = True

        if in_text and token == r"{":
            braces_counter += 1
        elif in_text and token == r"}":
            braces_counter -= 1
            if braces_counter == 0:
                in_text = False

        new_equation.append(token)

    return new_equation

def equation_join(split_equation: List[str]) -> str:
    """
    Joins a list of LaTeX tokens into a single string, adding spaces between tokens except
    when inside a '\\text{...}' block, where tokens are joined without spaces.

    Parameters:
    - split_equation (list of str): A list of LaTeX tokens.

    Returns:
    - str: The joined LaTeX equation as a single string.

    The function handles '\\text{...}' blocks by tracking the nesting of braces to ensure
    that all content within the braces is joined without additional spaces. This preserves
    the intended formatting of text within LaTeX equations.

    Example:
    >>> split_equation = [
    ...     "A", "_", "7", "+", "A", "_", "8", "-", "2", r"\text", "{",
    ...     "A", " ", "i", "s", " ", "a", " ", "p", "a", "r", "a", "m", "e", "t", "e", "r", "}"
    ... ]
    >>> equation_join(split_equation)
    'A _ 7 + A _ 8 - 2 \\text{A is a parameter}'
    """
    in_text = False
    braces_counter = 0
    joined_equation = ""
    for token in split_equation:
        if token == r"\text":
            in_text = True
            joined_equation += token + " "
            continue

        if in_text and token == r"{":
            braces_counter += 1
        elif in_text and token == r"}":
            braces_counter -= 1
            if braces_counter == 0:
                in_text = False

        if in_text:
            joined_equation += token
        else:
            joined_equation += token + " "

    return joined_equation.strip()


def replace_tokens(
    equation_tokenized: List[str], to_replace_tokens: Dict[str, str]
) -> List[str]:
    equation_tokenized = [
        to_replace_tokens[token] if token in to_replace_tokens else token
        for token in equation_tokenized
    ]
    equation_tokenized = [token for token in equation_tokenized if token]

    return equation_tokenized


def remove_with_braces(
    equation: str, remove_with_braces_tokens: List[str], keep_tokens: bool
):
    for token in remove_with_braces_tokens:
        pattern = rf"\{token}\s*\{{"
        output = ""
        pos = 0
        while True:
            match = re.search(pattern, equation[pos:])
            if not match:
                output += equation[pos:]
                break
            start = pos + match.start()
            end = pos + match.end()
            if keep_tokens:
                output += (
                    equation[pos : start + len(token)] + " "
                )  # Append text before and including \command
            else:
                output += equation[pos:start]  # Append text before \command{
            # Now find the matching closing brace
            depth = 1
            i = end
            while i < len(equation) and depth > 0:
                if equation[i] == "{":
                    depth += 1
                elif equation[i] == "}":
                    depth -= 1
                i += 1
            if depth != 0:
                # print(f"Unbalanced braces after position {end}")
                return None
            content = equation[end : i - 1]  # Extract content inside braces
            output += content  # Append the content without \command{ and }
            pos = i  # Move the position forward

        equation = output

    return equation


def remove_tokens_and_content(
    equation: str,
    remove_content_tokens: List[str],
):
    for token in remove_content_tokens:
        pattern = rf"\{token}\s*\{{"
        output = ""
        pos = 0
        while True:
            match = re.search(pattern, equation[pos:])
            if not match:
                output += equation[pos:]
                break
            start = pos + match.start()
            end = pos + match.end()

            output += equation[pos:start]  # Append text before \command{
            # Now find the matching closing brace
            depth = 1
            i = end
            while i < len(equation) and depth > 0:
                if equation[i] == "{":
                    depth += 1
                elif equation[i] == "}":
                    depth -= 1
                i += 1
            if depth != 0:
                # print(f"Unbalanced braces after position {end}")
                return None
            pos = i  # Move the position forward

        equation = output

    return equation


def normalize_one_braces_tokens(equation_tokenized: List[str], one_braces_tokens):
    new_equation = []
    i = 0
    while i < len(equation_tokenized):
        token = equation_tokenized[i]
        new_equation.append(token)
        i = i + 1
        if i < len(equation_tokenized):
            next_token = equation_tokenized[i]
            if token in one_braces_tokens and next_token != r"{":
                new_equation.extend(["{", next_token, "}"])
                i = i + 1

    return new_equation


def normalize_two_braces_tokens(equation_tokenized: List[str], two_braces_tokens):
    new_equation = []
    i = 0
    while i < len(equation_tokenized):
        token = equation_tokenized[i]
        new_equation.append(token)
        i = i + 1
        if i < len(equation_tokenized) - 1:
            next_token = equation_tokenized[i]
            next_next_token = equation_tokenized[i + 1]
            if token in two_braces_tokens and next_token != r"{":
                new_equation.extend(["{", next_token, "}", "{", next_next_token, "}"])
                i = i + 2

    return new_equation


def normalize_left_right(equation: str) -> str:
    left_delimiters = [
        "(",
        "[",
        "\\{",
        "|",
        "\\|",
        "\\lceil",
        "\\lfloor",
        "\\langle",
    ]
    right_delimiters = [
        ")",
        "]",
        "\\}",
        "|",
        "\\|",
        "\\rceil",
        "\\rfloor",
        "\\rangle",
    ]

    words_pattern = "|".join(map(re.escape, left_delimiters))
    pattern = rf"\{{\s*\\left\s*({words_pattern})\s*\}}"
    tmp_equation = re.sub(pattern, r"\\left \1", equation)

    words_pattern = "|".join(map(re.escape, right_delimiters))
    pattern = rf"\{{\s*\\right\s*({words_pattern})\s*\}}"
    new_equation = re.sub(pattern, r"\\right \1", tmp_equation)

    return new_equation


def handle_big(equation_tokenized: List[str]) -> List[str]:
    """
    Replaces instances of \\Big followed by valid tokens ( (, [, {, ), ], } ) with \\left or \\right.
    """
    new_equation = []
    i = 0
    seen_left_bar = False
    counter_left = 0
    counter_right = 0
    while i < len(equation_tokenized) - 1:
        token = equation_tokenized[i]
        next_token = equation_tokenized[i + 1]

        if (
            token == "\\Big"
            and (next_token == "|" or next_token == "\\|")
            and not seen_left_bar
        ):
            new_equation.append("\\left")
            seen_left_bar = True
            counter_left += 1
        elif (
            token == "\\Big"
            and (next_token == "|" or next_token == "\\|")
            and seen_left_bar
        ):
            new_equation.append("\\right")
            seen_left_bar = False
            counter_right += 1
        elif token == "\\Big" and next_token in [
            "(",
            "[",
            r"\{",
            r"\lceil",
            r"\lfloor",
            r"\langle",
        ]:
            new_equation.append("\\left")
            counter_left += 1
        elif token == "\\Big" and next_token in [
            ")",
            "]",
            r"\}",
            r"\rceil",
            r"\rfloor",
            r"\rangle",
        ]:
            new_equation.append("\\right")
            counter_right += 1
        else:
            new_equation.append(token)

        i += 1
    # Add the last token
    if equation_tokenized and equation_tokenized[-1] != "\\Big":
        new_equation.append(equation_tokenized[-1])

    if counter_left != counter_right:
        return equation_tokenized

    return new_equation


def handle_prime(equation):
    pattern = r"(?:'\s*)+"

    def replacement(match):
        count = match.group().count("'")
        return "^{" + "\\prime" * count + "}"

    return re.sub(pattern, replacement, equation)


def handle_multiple_subsequent_braces(equation):
    """
    Simplify a LaTeX equation by removing unnecessary multiple curly braces.

    Args:
        equation (str): The LaTeX equation to simplify.

    Returns:
        str: The simplified LaTeX equation.
    """
    pattern = r"\{\s*\{([^{}]+)\}\s*\}"
    prev_equation = None
    while equation != prev_equation:
        prev_equation = equation
        # Replace redundant multiple braces with a single brace
        equation = re.sub(pattern, r"{\1}", equation)
    return equation

def handle_multiple_subsequent_verts(equation):
    """
    Simplify a LaTeX equation by removing unnecessary multiple curly braces.

    Args:
        equation (str): The LaTeX equation to simplify.

    Returns:
        str: The simplified LaTeX equation.
    """
    pattern = r"\\\|\s*\\\|([^\\\|]+)\\\|\s*\\\|"
    prev_equation = None
    while equation != prev_equation:
        prev_equation = equation
        # Replace redundant multiple braces with a single brace
        equation = re.sub(pattern, r"\\|\1\\|", equation)
    return equation


def normalize(equation: str) -> str:
    equation = (
        equation.replace("<doctag>", "")
        .replace("</doctag>", "")
        .replace("<formula>", "")
        .replace("</formula>", "")
        .replace("<loc_0><loc_0><loc_500><loc_500>", "")
    )
    equation = remove_with_braces(equation, remove_with_braces_tokens, False)
    if not equation:
        return None
    equation = remove_tokens_and_content(equation, remove_content_tokens)
    if not equation:
        return None
    equation = remove_with_braces(equation, remove_with_braces_tokens_keep_tokens, True)
    if not equation:
        return None
    equation = normalize_left_right(equation)

    equation = handle_multiple_subsequent_braces(equation)
    equation = handle_multiple_subsequent_verts(equation)
    equation = handle_prime(equation)

    if not equation:
        return None

    equation = equation.strip()
    equation_tokenized = tokenizer.tokenize(equation)

    if len(set(equation_tokenized).intersection(tokens_to_drop)) > 0:
        return None

    equation_tokenized = handle_spaces(equation_tokenized)

    if not equation_tokenized:
        return None

    equation_tokenized = replace_tokens(equation_tokenized, tokens_to_replace)
    equation_tokenized = normalize_one_braces_tokens(
        equation_tokenized, one_braces_tokens
    )
    equation_tokenized = normalize_two_braces_tokens(
        equation_tokenized, two_braces_tokens
    )
    equation_tokenized = handle_big(equation_tokenized)

    equation = equation_join(equation_tokenized)

    equation = equation.replace("  ", " ").replace(r"\text {", r"\text{").strip()

    equation_tokenized = tokenizer.tokenize(equation)
    equation_tokenized = handle_spaces(equation_tokenized)
    return equation_join(equation_tokenized)


def _process_file(file_path: Path, out_dir: Path) -> tuple[str, bool]:
    """
    Read one-line equation from *file_path*, normalize it, and write the result
    to *out_dir/file_path.name*.
    Returns (filename, True) on success, (filename, False) if normalization
    returned None.
    """
    try:
        eq = file_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        return (file_path.name, False)

    normalized_equation = normalize(eq)  # may be None
    flag = True
    if normalized_equation is None:
        normalized_equation = eq
        flag = False
    (out_dir / file_path.name).write_text(normalized_equation, encoding="utf-8")
    
    return (file_path.name, flag)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize LaTeX-equation files in parallel."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        help="Directory containing text files, each with a single equation.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to receive normalized files (created if absent).",
    )
    parser.add_argument(
        "-w",
        "--num_workers",
        type=int,
        default=cpu_count(),
        metavar="N",
        help="Number of worker processes (default: all logical cores).",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error(f"{args.input_dir} is not a directory.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in args.input_dir.iterdir() if p.is_file())
    if not files:
        parser.error(f"No files found in {args.input_dir}")

    with Pool(processes=min(args.num_workers, cpu_count())) as pool:
        process = partial(_process_file, out_dir=args.output_dir)
        results = pool.map(process, files)

    ok = sum(flag for _, flag in results)
    fail = len(results) - ok
    print(f"✓  Normalized {ok} file(s); ✗  skipped {fail} file(s).")


if __name__ == "__main__":
    main()
