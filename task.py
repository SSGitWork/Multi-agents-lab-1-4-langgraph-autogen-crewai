"""
task.py  –  The coding task used in all three framework implementations.

All three skeleton files import TASK_DESCRIPTION from here so you are
running an identical benchmark across frameworks.

Students do NOT modify this file.
"""

TASK_DESCRIPTION: str = """
Write a Python module called `word_freq.py` that contains a single function:

    def word_frequency(filepath: str) -> dict[str, int]:
        \"\"\"Return a dict mapping each word to its frequency in the file.\"\"\"

Requirements:
- Read the file at `filepath`.
- Normalise words: lowercase, strip punctuation.
- Return a dict sorted by frequency descending (most common first).
- Raise FileNotFoundError with a clear message if the file does not exist.
- Add type annotations and a one-sentence docstring to the function.

After writing the module:
1. Create a small test file called `sample.txt` with at least 20 words.
2. Execute `word_freq.py` with a short driver snippet to print the top-5 words.
3. Confirm the output looks correct, then return DONE.
"""
