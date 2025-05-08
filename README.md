# CS410 Tech Review

Please note that this code may take a relative while to run the first time when creating the cache.

## Getting Started

This code uses `uv` to manage the dependencies. The documentation for `uv` can be found here: <https://docs.astral.sh/uv/>

Once it is installed, please download the dependencies using:

```
uv sync
```

It should automatically download GPU accelerated libraries for both Linux and Windows environments. Ideally, everything should "just work".

Download the spacy model

```
uv run python -m spacy download en_core_web_sm
```

Finally, start the application

```
uv run cs410-han
```

OR

```
uv run -m cs410_han
```

## Graphs

Create the graphs by running either of the below.

```
uv run scripts/graph1.py
```

```
uv run scripts/graph2.py
```

## Notes

- Used Spacy instead of Stanford CoreNLP due to the additional setup and dependencies.
