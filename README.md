# My Results and Criteria

See [merged_results.md](./merged_results.md).


# Run your own benchmark

The script can run regardless of your OS.
You will need Python 3, [`ffmpeg`](https://ffmpeg.org/) and the dependencies listed in [requirements.txt](./requirements.txt).
You can install the dependencies with `python3 -m pip install -r "requirements.txt"`

To run the benchmark:
1. Change audio sources in the `run.py` adding paths to the `files` dictionary
2. Change the codecs or adjut their parameters in the `codecs` dictionary in `run.py` if needed
3. Run `python run.py`
