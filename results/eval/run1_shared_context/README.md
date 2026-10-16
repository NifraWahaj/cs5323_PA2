# Judging run 1 (do not use for headline numbers)

`llm.prompt()` inside a `kaggle_benchmarks` task keeps one conversation for the whole task. Input tokens per
request grew from 2.7k to 538k over 190 calls (see `*.run.json`), so each batch was judged with every
earlier batch in context. The judgments are therefore not independent: one AlpacaEval verdict mentions the
persona rubric, and 22 of 24 style verdicts were lost. Run 2 isolates each call with `kbench.chats.fork`.
These numbers are kept to compare against run 2 in the report.
