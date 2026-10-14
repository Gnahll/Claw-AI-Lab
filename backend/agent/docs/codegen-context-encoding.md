# S11 optional context file encoding

S11 discovers local resources before generating experiment code. Dataset text
samples and codebase README files are optional background information, not the
training data, approved split manifest, experiment plan, or checkpoint contents.

These optional files are read strictly as UTF-8 (including BOM) first, then
GB18030, which also supports GBK text. The live code-generation log records the
file path when a fallback encoding is used. No replacement characters are
silently inserted.

If neither encoding works, the file contains NUL bytes, or an I/O error occurs,
S11 logs why the optional file was skipped and continues discovering resources.
An unreadable codebase README does not prevent trying a nested README. This does
not mark code generation or training successful, bypass required resource
validation, alter dataset files, or resume a stopped task automatically.

The original failure occurred while reading COVERAGE/readme.txt as UTF-8, before
code generation or training started. Regression tests cover UTF-8, UTF-8 BOM,
GBK, GB18030, unsupported bytes, binary content, and permission errors.

Run the focused regression suite from backend/agent:

```text
python -m pytest tests/test_stage09_stage11_regressions.py -q
```
