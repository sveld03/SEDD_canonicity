# Canonicity in the Score Entropy Discrete Diffusion Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Acknowledgements

The bulk of this repo is cloned from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion, which contains a PyTorch implementation for the paper [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution
](https://arxiv.org/abs/2310.16834) by [Aaron Lou](https://aaronlou.com), [Chenlin Meng](https://cs.stanford.edu/~chenlin/) and [Stefano Ermon](https://cs.stanford.edu/~ermon/). See the SEDD_README folder for a description of the source repo.

The remainder of this repo is dedicated to investigating the properties and influence of non-canonical tokenizations in the SEDD model. This research is built upon Renato Geh's research on canonicity in autoregressive models, detailed in his [tokenization](https://github.com/RenatoGeh/tokenization) repository and the paper [Where is the signal in tokenization space?](https://arxiv.org/pdf/2408.08541), and has been advised and directed by Renato throughout the process. 

## Context

When generating text, language models will sometimes generate a non-canonical sequence -- that is, a sequence of tokens T that can be interpreted as a string S (e.g., "Alice forced Bob to move"), but is different from the token sequence T' that is generated by passing S through the canonical tokenizer (in our case, GPT2TokenizerFast). In the above example, the word "forced" may be canonically encoded as one token, but is broken into two tokens "for" and "ced" in the model's output. Previous research by Renato Geh and colleagues indicates that "by simply aggregating the probabilities of non canonical tokenizations, [one can] achieve improvements across a range of LLM evaluation benchmarks for a variety of architectures, including transformers and state space models." Given that non-canonical tokenizations carry some meaningful signal in autoregressive models, despite the typical assumption that the probability of a string is simply equal to the probability of its canonical tokenization, this repository seeks to extend the prior research to the domain of discrete diffusion text models.

## Functionality

The majority of the additional code beyond the original SEDD repository can be found in main.py. In addition to some helper functions, main.py contains four functions to investigate features of non-canonical tokenizations. Each of the functions calls the SEDD-medium model to generate text samples of various lengths (with 100 samples for each sequence length). The first function, check_canonicity_many(), computes the percent of samples that are fully canonical for each sequence length. The second function, check_edit_distance(), computes the average edit distance between the canonical and non-canonical tokenizations for each sequence length -- that is, the number of token additions/deletions/substitutions required to get from the canonical to the non-canonical tokenization. The third function, compare_likelihoods(), computes the average log-likelihood of both the canonical and non-canonical tokenizations using GPT-2 as the predictive language model. The motivation for computing log-likelihood using GPT-2 is explained in the "Findings" section. The fourth function, do_it_all(), computes the percent canonicity, average log-likelihood, and average log-likelihoods for the same samples, in order to determine relationships between these metrics. 

To run any of the functions, simply call the function in main.py:main(). This will generate two output files, one containing the raw data generated and the other containing a summary of the results. The names of the files can be changed within each function by editing the output_file and raw_file variables. Additionally, the number of samples can be changed within each function by editing the num_counts, batches_per_count, and batch_size variables.

## Findings

Preliminary findings can be found in both the "Real_Data" folder and the "Graphs" folder. All files in Real_Data labeled "raw" contain raw data from all samples, while the remaining files contain condensed summaries. Most of the high-level takeaways are easiest to see in the generated graphs:
1. canonicity_plot.png: As expected, longer model outputs are less likely to be canonical. The plot also illustrates the fact that tokens are not generated independently (where we would expect to see exponential decay of canonicity). This is due to the fact that discrete diffusion models update the probabilities of all tokens at each reverse diffusion step, in the context of all other tokens within the same context window.
2. edit_distance_plot.png: As expected, edit distance increases with sequence length. Unexpectedly, the edit distance decreased for sequence lengths greater than 750, although this is likely a statistical anomaly as the same result did not occur in the combined analysis.
3. log_likelihood_plot.png and log_likelihood difference.png: As expected, log-likelihood decreased with increasing sequence length, which is to say that GPT-2 predictably performs worse as predicting longer sequences. Unexpectedly, the non-canonical log-likelihood was consistently higher than the canonical log-likelihood. The motivation for using GPT-2 to compute log-likelihood is due to the fact that it is computationally intractable to compute the likelihood of a given token sequence using a discrete diffusion model. For this reason, researchers seeking to measure performance of discrete diffusion models will typically feed the model's outputs into an autoregressive model (such as GPT-2) that uses the same tokenizer as the diffusion model, and compute the log-likelihood of the sequence using the autoregressive model. We hypothesized that some researchers may be artificially boosting the performance scores of discrete diffusion models by canonically retokenizing the model's output before passing it to the autoregressive model. However, we were surprised to find the opposite -- that GPT-2 is actually better at predicting the raw outputs of the SEDD model than their canonical counterparts. This is surprising because GPT-2 was both trained on different data and has a different architecture than SEDD, and yet the result indicates that the small number of cases in which the model generates a non-canonical token (<1% of tokens) almost entirely overlap between SEDD and GPT-2. Further investigation into this phenomenon forthcoming.
4. combined_analysis_plot_2.png: As expected, edit distance and log-likelihood distance are strongly correlated, and both are anticorrelated with percent canonicity.