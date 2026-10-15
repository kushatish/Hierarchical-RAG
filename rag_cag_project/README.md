# Running the application

The [main README](../README.md) walks through setup, the demo, and the optional model integrations. The [technical guide](../TECHNICAL_GUIDE.md) covers the API, cache settings, and Docker configuration.

If your environment is already activated and the dependencies are installed, start the interface from this directory:

```bash
streamlit run app.py
```

For a quick terminal demo:

```bash
python cli.py ingest data/demo/handbook.md
python cli.py ask "What is the minimum attendance requirement?"
```

To run the tests and development benchmark:

```bash
python -m unittest discover -s tests -v
python -m evaluation.benchmark
```
