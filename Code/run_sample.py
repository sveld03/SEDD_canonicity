import torch
import argparse

from load_model import load_model
from transformers import GPT2TokenizerFast
import torch.nn.functional as F
import sampling

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

torch.set_printoptions(threshold=10000)

def sample_tokens(batch_size, num_tokens, steps, intermediates=False):
    
    device = torch.device('cuda:2')
    model, graph, noise = load_model("louaaron/sedd-medium", device)

    sampling_fn = sampling.get_pc_sampler(
        graph, noise, (batch_size, num_tokens), 'analytic', steps, device=device, intermediates=intermediates
    )

    return sampling_fn(model)

def main():

    # intermediate_states = sample_tokens(1, 10, 5)

    # for step, text in intermediate_states.items():
     #   print(f"Step {step}: {text}\n")

    print(tokenizer.decode([220]))

    # print(f"final: {final_state.tolist()}")
    # parser = argparse.ArgumentParser(description="Generate some samples")
    # parser.add_argument("--model_path", default="louaaron/sedd-medium", type=str)
    # parser.add_argument("--dataset", default="wikitext103", type=str)
    # parser.add_argument("--batch_size", type=int, default=1)
    # parser.add_argument("--steps", type=int, default=1024)
    # args = parser.parse_args()

    
    # device = torch.device('cuda:2')
    # model, graph, noise = load_model(args.model_path, device)
    # tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # sampling_fn = sampling.get_pc_sampler(
    #     graph, noise, (args.batch_size, 1024), 'analytic', args.steps, device=device
    # )

    # for n in range(5):
    #     tokenized_output = sampling_fn(model)

    #     print("Iteration " + str(n))
    #     print()

    #     for token_ids in tokenized_output:
    #         print(token_ids)
    #         print("=================================================")

    #     text_samples = tokenizer.batch_decode(tokenized_output)
    #     for i in text_samples:
    #         print(i)
    #         print("=================================================")
    #         print()

if __name__=="__main__":
    main()