from transformers import GPT2TokenizerFast
import torch

from run_sample import sample_tokens

torch.set_printoptions(threshold=10000)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def check_canonicity_bool(actual_tokens, canonical_tokens):
    if actual_tokens.numel() != canonical_tokens.numel():
        return False
    if (actual_tokens == canonical_tokens).all():
        return True
    else:
        return False

def check_canonicity_percent(actual_tokens, canonical_tokens):
    tokens_matched = 0
    total_tokens = 0
    atl = 0 # "actual tokens lookahead"
    ctl = 0 # "canonical tokens lookahead"
    anum = actual_tokens.numel() - 2
    cnum = canonical_tokens.numel() - 2
    for i in range(anum):
        if i+atl > anum or i+ctl > cnum:
            break
        if actual_tokens[i + atl] == canonical_tokens[i + ctl]:
            tokens_matched += 1
        elif actual_tokens[i+atl+1] == canonical_tokens[i+ctl+2]:
            ctl += 1
        elif actual_tokens[i+atl+2] == canonical_tokens[i+ctl+1]:
            atl += 1
        total_tokens += 1
    return (tokens_matched / total_tokens) * 100


def check_canonicity_many():
    output_file = "12-2-batch-check-canonicity.txt"
    raw_file = "12-2-batch-raw-check-canonicity.txt"

    device = torch.device("cuda:0")

    token_counts = [1, 50, 100, 200, 300, 500]
    step_counts = [100, 100, 100, 100, 100, 100]

    batches = 2 # change to 6
    batch_size = 3 # change to 100
    
    for i in range(batches): 
        token_count = token_counts[i]
        steps = step_counts[i]

        actual_tokens = sample_tokens(batch_size, token_count, steps)
        text = tokenizer.batch_decode(actual_tokens)
        canonical_tokens = tokenizer(
            text, 
            padding=False, 
            truncation=True,
            return_tensors='pt')["input_ids"].to(device)

        canon_count = 0

        for j in range(batch_size):
            actual_sample = actual_tokens[j]
            canonical_sample = canonical_tokens[j]
            text_sample = text[j]
            canon = check_canonicity_bool(actual_sample, canonical_sample)
            with open(raw_file, 'a') as file:
                file.write("For iteration " + str(j) + " of " + str(token_count) + " tokens and " + str(steps) + " steps, here were the results:\n\n")
                file.write("Canonical? " + ("Yes\n" if canon else "No\n"))
                file.write("Actual tokens:\n" + str(actual_sample) + "\n\n")
                file.write("Canonical tokens:\n" + str(canonical_sample) + "\n\n")
                file.write("Text:\n" + text_sample + "\n\n")
                file.write("=================================================================\n\n\n")
            if canon:
                canon_count += 1
        
        percent = (canon_count / batch_size) * 100

        with open(output_file, 'a') as file:
            file.write("For " + str(token_count) + " tokens and " + str(steps) + " steps, percent canonicity was " + str(percent) + "%\n")


    # for i in range(batches): 
    #     tokens = token_counts[i]
    #     steps = step_counts[i]
    #     canon_count = 0
    #     for j in range(batch_size):
    #         actual_tokens_raw = sample_tokens(1, tokens, steps)
    #         actual_tokens = actual_tokens_raw.squeeze(0)
    #         text = tokenizer.decode(actual_tokens)
    #         canonical_tokens_list = tokenizer.encode(text)
    #         canonical_tokens = torch.tensor(canonical_tokens_list, device='cuda:0')
    #         canon = check_canonicity_bool(actual_tokens, canonical_tokens)
    #         with open(raw_file, 'a') as file:
    #             file.write("For iteration " + str(j) + " of " + str(tokens) + " tokens and " + str(steps) + " steps, here were the results:\n\n")
    #             file.write("Actual tokens:\n" + str(actual_tokens) + "\n\n")
    #             file.write("Canonical tokens:\n" + str(canonical_tokens) + "\n\n")
    #             file.write("Text:\n" + text + "\n\n")
    #             file.write("=================================================================\n\n\n")
    #         if canon:
    #             canon_count += 1
    #     percent_canonicity = (canon_count / batch_size) * 100

    #     with open(output_file, 'a') as file:
    #         file.write("For " + str(tokens) + " tokens and " + str(steps) + " steps, percent canonicity was " + str(percent_canonicity) + "%\n")

def main():
    check_canonicity_many()

if __name__ == "__main__":
    main()





"""The following function is not to be used anytime soon... meant for long strings and to check canonicity WITHIN the string"""
def check_canonicity_long():
    output_file = "canonicity_survey_long_1.txt"

    raw_file = "canonicity_raw_long_1.txt"
    
    token_counts = [50, 100, 250, 500, 1000]
    step_counts = [200, 300, 500, 750, 1000]
    #token_counts = [100]
    #step_counts = [100]

    for i in range(2): # change to 5
        token_count = token_counts[i]
        steps = step_counts[i]
        actual_tokens = sample_tokens(3, token_count, steps) # change to 10
        # actual_tokens = actual_tokens_raw.squeeze(0)
        text = tokenizer.batch_decode(actual_tokens)
        canonical_tokens = tokenizer.batch_encode(text)
        #canonical_tokens = torch.tensor(canonical_tokens_list, device='cuda:0')
        #print(actual_tokens)
        #print(canonical_tokens)
        percent_canonicity = check_canonicity_percent(actual_tokens, canonical_tokens)

        with open(output_file, 'a') as file:
            file.write("For iteration " + str(i%10) + " of " + str(token_count) + " tokens and " + str(steps) + " steps, percent canonicity was " + str(percent_canonicity) + "%\n")

        with open(raw_file, 'a') as file:
            file.write("For " + str(token_count) + " tokens and " + str(steps) + " steps, here were the results:\n\n")
            file.write("Actual tokens:\n" + str(actual_tokens) + "\n\n")
            file.write("Canonical tokens:\n" + str(canonical_tokens) + "\n\n")
            file.write("Text:\n" + text + "\n\n")
            file.write("=================================================================\n\n\n")