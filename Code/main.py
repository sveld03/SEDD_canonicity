from transformers import GPT2TokenizerFast, AutoModelForCausalLM
import torch
import Levenshtein, tqdm, collections
import numpy as np
from datetime import datetime

from run_sample import sample_tokens
from utils import rhloglikelihood
from load_model import load_model

torch.set_printoptions(threshold=10000)

device = torch.device('cuda:2')
# model, graph, noise = load_model("louaaron/sedd-medium", device)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def check_canonicity_one(actual_tokens, canonical_tokens):
    if actual_tokens.numel() != canonical_tokens.numel():
        return False
    if (actual_tokens == canonical_tokens).all():
        return True
    else:
        return False
    
def rmst(X: list) -> list:
    "Returns a new list without special tokens."
    if np.issubdtype(type(X[0]), np.integer) or (len(X[0]) == 0):
        return [x for x in (X.numpy() if isinstance(X, torch.Tensor) else X) if x not in tokenizer.all_special_ids]
    return [[t for t in (x.numpy() if isinstance(x, torch.Tensor) else x) if t not in tokenizer.all_special_ids] for x in X]


def dist_canon(X: torch.Tensor) -> np.ndarray:
    if isinstance(X, torch.Tensor):
        X = X.tolist()
    f = tokenizer.decode if isinstance(X[0], int) else tokenizer.batch_decode
    s = f(X, skip_special_tokens=True) # s = tokenizer.decode(original_tokens, skip_special_tokens=True)
    K, O = tokenizer(s, add_special_tokens=False)["input_ids"], rmst(X)
    return np.array([Levenshtein.distance(k, o) for k, o in zip(K, O)])

def canon(X: list) -> list:
    f = tokenizer.decode if np.issubdtype(type(X[0]), np.integer) else tokenizer.batch_decode
    s = f(X, skip_special_tokens=False)
    return tokenizer(s, add_special_tokens=False)["input_ids"]

def uncanons(V: list, V_canon: list = None) -> dict:
    if isinstance(V[0], torch.Tensor): V = V.cpu().numpy()
    if V_canon is None: V_canon = canon(V)
    O, c = collections.defaultdict(list), 0
    l_u, l_v = 0, 0
    i, j, start_i, start_j = 0, 0, 0, 0
    move_i, move_j = True, True
    while (i < len(V)) and (j < len(V_canon)):
        u, v = V[i], V_canon[j]
        l_u += len(tokenizer.decode([u])) if move_i else 0
        l_v += len(tokenizer.decode([v])) if move_j else 0
        move_i, move_j = False, False
        if l_u >= l_v:
            j += 1
            move_j = True
        if l_v >= l_u:
            i += 1
            move_i = True
        if l_u != l_v:
            if c == 0: start_i, start_j = i-move_i, j-move_j
            c += 1
        elif c > 0:
            O[i-start_i].append(([tokenizer.decode([V[x]]) for x in range(start_i, i)],
                                 [tokenizer.decode([V_canon[y]]) for y in range(start_j, j)]))
            c = 0
    return O

def check_canonicity_many():
    output_file = "12-2-batch-check-canonicity.txt"
    raw_file = "12-2-batch-raw-check-canonicity.txt"

    device = torch.device("cuda:2")
    tokenizer.pad_token = tokenizer.eos_token

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
            padding=True, 
            truncation=True,
            return_tensors='pt')["input_ids"].to(device)

        canon_count = 0

        for j in range(batch_size):
            actual_sample = actual_tokens[j]
            canonical_sample = canonical_tokens[j]
            text_sample = text[j]
            canon = check_canonicity_one(actual_sample, canonical_sample)
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

def check_edit_distance():
    output_file = "REAL-12-8-edit-distance-2.txt"
    raw_file = "REAL-12-8-raw-edit-distance-2.txt"

    device = torch.device("cuda:2")
    tokenizer.pad_token = tokenizer.eos_token
    
    token_counts = [100, 250, 400, 500, 750, 900, 1000]
    step_counts = [250, 400, 500, 600, 750, 900, 1000]

    num_counts = 7
    batches_per_count = 20
    batch_size = 5

    for i in range(num_counts): 
        token_count = token_counts[i]
        steps = step_counts[i]

        with open(raw_file, 'a') as file:
            file.write("RESULTS FOR TOKEN COUNT " + str(token_count) + " AND STEP COUNT " + str(steps) + "\n\n\n" )

        all_distances = []

        for j in range(batches_per_count):

            actual_tokens = sample_tokens(batch_size, token_count, steps)
            text = tokenizer.batch_decode(actual_tokens)
            canonical_tokens = tokenizer(
                text, 
                padding=True, 
                truncation=True,
                return_tensors='pt')["input_ids"].to(device)
            distances = dist_canon(actual_tokens)

            for k in range(batch_size):

                actual_sample = actual_tokens[k]
                canonical_sample = canonical_tokens[k]

                uncanons_output = uncanons(actual_sample, canonical_sample)

                with open(raw_file, 'a') as file:
                    file.write("=================================================================\n")
                    file.write("the edit distance for sequence " + str(batch_size*j+k+1) + " with " + str(token_count) + " tokens and " + str(steps) + " steps is " + str(distances[k]) + "\n\n")
                    file.write("Here were the words that were tokenized non-canonically, along with their canonical vs non-canonical tokenizations: \n")
                    for position, token_pairs in uncanons_output.items():
                        for non_canon, canon in token_pairs:
                            file.write("Non-canonical: " + str(non_canon) + ", canonical: " + str(canon) + "\n")
                    file.write("=================================================================\n\n\n")

                all_distances.append(distances[k])

        avg_edit_distance = np.mean(all_distances)

        with open(raw_file, 'a') as file:
            file.write("=================================================================")
            file.write("=================================================================\n\n\n\n")

        with open(output_file, 'a') as file:
            file.write("The average edit distance for " + str(token_count) + " tokens and " + str(steps) + " steps is " + str(avg_edit_distance) + "\n")

def compare_likelihoods():

    output_file = "TEST-12-25-likelihood-1.txt"
    raw_file = "TEST-12-25-raw-likelihood-1.txt"

    model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")

    device = torch.device("cuda:2")
    tokenizer.pad_token = tokenizer.eos_token
    
    token_counts = [100, 250, 400, 500, 750, 900, 1000]
    step_counts = [250, 400, 500, 600, 750, 900, 1000]

    num_counts = 7
    batches_per_count = 1
    batch_size = 1

    for i in range(num_counts): 
        token_count = token_counts[i]
        steps = step_counts[i]

        with open(raw_file, 'a') as file:
            file.write("RESULTS FOR TOKEN COUNT " + str(token_count) + " AND STEP COUNT " + str(steps) + "\n\n\n" )
        
        all_non_canonical_likelihoods = []
        all_canonical_likelihoods = []

        for j in range(batches_per_count):

            actual_tokens = sample_tokens(batch_size, token_count, steps)
            non_canonical_text = tokenizer.batch_decode(actual_tokens)
            canonical_tokens = tokenizer(
                non_canonical_text, 
                padding=True, 
                truncation=True,
                return_tensors='pt')["input_ids"].to(device)

            for k in range(batch_size):

                non_canonical_likelihood = rhloglikelihood(model, tokenizer, [actual_tokens[k]]).item()
                canonical_likelihood = rhloglikelihood(model, tokenizer, [canonical_tokens[k]]).item()

                uncanons_output = uncanons(actual_tokens[k], canonical_tokens[k])

                with open(raw_file, 'a') as file:
                    file.write("=================================================================\n")
                    file.write("the non-canonical log-likelihood for sequence " + str(batch_size*j+k+1) + " with " + str(token_count) + " tokens and " + str(steps) + " steps is " + str(non_canonical_likelihood) + ", and the canonical log-likeliood is " + str(canonical_likelihood) + "\n\n")
                    for position, token_pairs in uncanons_output.items():
                        for non_canon, canon in token_pairs:
                            file.write("Non-canonical: " + str(non_canon) + ", canonical: " + str(canon) + "\n")
                    file.write("Actual tokens:\n" + str(actual_tokens[k]) + "\n\n")
                    file.write("Canonical tokens:\n" + str(canonical_tokens[k]) + "\n\n")
                    file.write("=================================================================\n\n\n")

                all_non_canonical_likelihoods.append(non_canonical_likelihood)
                all_canonical_likelihoods.append(canonical_likelihood)

        avg_non_canonical_likelihood = np.mean(all_non_canonical_likelihoods)
        avg_canonical_likelihood = np.mean(all_canonical_likelihoods)
        
        with open(raw_file, 'a') as file:
            file.write("=================================================================")
            file.write("=================================================================\n\n\n\n")

        with open(output_file, 'a') as file:
            file.write("the average non-canonical log-likelihood for " + str(token_count) + " tokens and " + str(steps) + " steps is " + str(avg_non_canonical_likelihood) + ", and the average canonical log-likeliood is " + str(avg_canonical_likelihood) + "\n\n")

def do_it_all():
    output_file = "TEST-1-30-doitall-1.txt"
    raw_file = "TEST-1-30-raw-doitall-1.txt"

    model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")

    device = torch.device("cuda:2")
    tokenizer.pad_token = tokenizer.eos_token
    
    token_count = 1000
    step_counts = [10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    num_counts = 4
    batches_per_count = 2
    batch_size = 3

    for i in range(num_counts): 
        steps = step_counts[i]

        with open(raw_file, 'a') as file:
            file.write("RESULTS FOR TOKEN COUNT " + str(token_count) + " AND STEP COUNT " + str(steps) + "\n\n\n" )

        canon_count = 0
        
        all_distances = []

        all_non_canonical_likelihoods = []
        all_canonical_likelihoods = []

        for j in range(batches_per_count):

            actual_tokens = sample_tokens(batch_size, token_count, steps)
            non_canonical_text = tokenizer.batch_decode(actual_tokens)
            canonical_tokens = tokenizer(
                non_canonical_text, 
                padding=True, 
                truncation=True,
                return_tensors='pt')["input_ids"].to(device)
            
            distances = dist_canon(actual_tokens)

            for k in range(batch_size):

                non_canonical_likelihood = rhloglikelihood(model, tokenizer, [actual_tokens[k]]).item()
                canonical_likelihood = rhloglikelihood(model, tokenizer, [canonical_tokens[k]]).item()

                canon_bool = check_canonicity_one(actual_tokens[k], canonical_tokens[k])

                uncanons_output = uncanons(actual_tokens[k], canonical_tokens[k])

                with open(raw_file, 'a') as file:
                    file.write("=================================================================\n")
                    file.write("Canonical? " + ("Yes\n" if canon_bool else "No\n"))
                    file.write("the edit distance for sequence " + str(batch_size*j+k+1) + " with " + str(token_count) + " tokens and " + str(steps) + " steps is " + str(distances[k]) + "\n\n")
                    file.write("the non-canonical log-likelihood for sequence " + str(batch_size*j+k+1) + " with " + str(token_count) + " tokens and " + str(steps) + " steps is " + str(non_canonical_likelihood) + ", and the canonical log-likeliood is " + str(canonical_likelihood) + "\n\n")
                    for position, token_pairs in uncanons_output.items():
                        for non_canon, canon in token_pairs:
                            file.write("Non-canonical: " + str(non_canon) + ", canonical: " + str(canon) + "\n")
                    file.write("=================================================================\n\n\n")

                if canon_bool:
                    canon_count += 1
                
                all_distances.append(distances[k])

                all_non_canonical_likelihoods.append(non_canonical_likelihood)
                all_canonical_likelihoods.append(canonical_likelihood)

        percent = (canon_count / (batch_size * batches_per_count)) * 100
        avg_edit_distance = np.mean(all_distances)

        avg_non_canonical_likelihood = np.mean(all_non_canonical_likelihoods)
        avg_canonical_likelihood = np.mean(all_canonical_likelihoods)
        
        with open(raw_file, 'a') as file:
            file.write("=================================================================")
            file.write("=================================================================\n\n\n\n")

        with open(output_file, 'a') as file:
            file.write("Here are the results for " + str(token_count) + " tokens and " + str(steps) + " steps: \n\n")
            file.write("Percent canonicity was " + str(percent) + "%\n")
            file.write("The average edit distance is " + str(avg_edit_distance) + "\n")
            file.write("the average non-canonical log-likelihood is " + str(avg_non_canonical_likelihood) + ", and the average canonical log-likeliood is " + str(avg_canonical_likelihood) + "\n")
            file.write("=================================================================\n\n\n")

def find_non_canonicals():
    output_file = "noncanons-1-15-try3.txt"

    model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")

    device = torch.device("cuda:2")
    tokenizer.pad_token = tokenizer.eos_token

    for i in range(5):
        actual_tokens = sample_tokens(1, 500, 500)
        non_canonical_text = tokenizer.batch_decode(actual_tokens)
        canonical_tokens = tokenizer(
            non_canonical_text, 
            padding=True, 
            truncation=True,
            return_tensors='pt')["input_ids"].to(device)
        
        for k in range(5):
            canon_bool = check_canonicity_one(actual_tokens[k], canonical_tokens[k])
            if not canon_bool:
                uncanons_output = uncanons(actual_tokens[k], canonical_tokens[k])

                with open(output_file, 'a') as file:
                    for position, token_pairs in uncanons_output.items():
                        for non_canon, canon in token_pairs:
                            file.write("Non-canonical: " + str(non_canon) + ", canonical: " + str(canon) + "\n")
                    file.write("=================================================================\n\n")
                    file.write("Text:\n" + str(non_canonical_text[k]) + "\n\n")
                    file.write("=================================================================\n\n\n")
                    file.write("=================================================================\n\n\n")


def main():
    start_time = datetime.now() 

    import re

    # Paste the provided text here
    text1 = """"<|endoftext|> Written  by  Junk o;;  items  below . 
 
;;  fortunately ,  however ,  error;;  in  this  game .  Both;; su â€” V ido -;;  der az umi 
 
;;  ï¿½ ï¿½ Sky ran ï¿½;; ï¿½ ï¿½ 
 
 Total;; ï¿½ ! 
 
 Total;;  SY R ,  CV ,;;  X DF K  rockets ,;;  Steel  Diver ,  Gun  Squad;;  Mar iza ,  G +;; :  Gat  Gat  Gun ner;;  Not  revived 
 
 Total;;  snow  through  winter  and  spring;;  connected  by  narrow  networks  of;;  railways  and  human - powered;;  of  Earth - X ï¿½;; ï¿½ s  satellites  would  know;;  reality .  And  they  would;;  Sa ikawa;;  Also ,;;  Hong su;; !  info;;  can  be;; :  Steam;;  of  the;;  place  during;;  and  view;; 
 Edited;;  the  Original;;  rate  network;; , 0;;  on  Youtube;; rob aki;; / r;; ers _-_;; t og;; un _-_;; rec ap;; _-_ the;; obby _;; ,  you;;  one  another;; 
 A;;  if  you;;  want .;; ED  7;;  c Development;; :  Tempest;; 
 
;; ï¿½ 
;;  list  level;; :  Sentinel;; list  level;;  Assault ï¿½;; list  level;; ing  Up;;  Archer ,;;  Level  2;; :  X;;  Level  3;;  Master 
;;  Region 
;;  on  preserved;;  of  protected;;  for  isolated;;  fre eways;; ."" 
;;  but  also;; 
 
 The  Command;;  found  on  the  various;;  K Å hei  Nan;; No - ji ,;;  Fighters  can  be  streamed;; me ets / t;; 69 / written _;; In cluding  the  three;;  most  likely ,  see;;  uses  a  grid ,;;  In  practice ,  you;; 
 VO :  Nintendo;; 's  ghost  from  the;;  Check  out  it  to;; Version  L OD s;; 
 
 COL LECT;;  Music  for  ï¿½ ï¿½;;  3 :  Space  for;; i 
 
 Total;; b 
 
 Total;;  list  Level  5 :;; 
 ï¿½ ï¿½ ï¿½;;  News  Article :  "";;  mountains  of  forest  that;;  would  absorb  rain  and;;  the  would - be;;  the  region ï¿½ ï¿½;; s  vast  bas ins;; hab ited ,  save;;  its  seas  and  rivers;;  of  Size 
;;  but  numerous  maps;; ,  providing  a;; Image  Sources :;; 
 
 NOTE;;  it 's  the;;  covering  the  current;;  the  process .;;  on  speed  and;;  version  2 ,;; : 
 
;; by _ read;; Viol ent _;; ! 
 
;;  many  players  achieve;;  your  friends  against;; ,  or  ï¿½;; ï¿½ ï¿½  T;; aito ,  you;;  shape  it  up;;  Few  Maps 
;;  and  Tsu  will;;  find  over  at;;  the  Temple !;;  End  Release 
;; Jose B  Area;; Jose B  Area;;  2  Van  Tun;;  groundwater .  Along;;  border  towns ,;;  land  are  far;; .  Despite  all;; 
 The  dwell;; 
 Not  the  full  game  itself  has  been  released ,;;  the  two - only  mode ,  and  others  can  note;;  run  straight  into  strategy  issues .  The  rules  are  very;;  Z ens iem ancer  has  a  new  inventory  system .;;  would  become  de celer ated  residents ,  since  most  of;;  general;;  organization;;  overview;;  by;;  to;;  are;;  Hong;;  Hom;; http;; ://;; com;; aya;;  simple;;  (;;  exploring;;  for;;  1;;  1;;  Generation;; Alpha;;  Combat;;  list;;  list;;  Frank;;  been;;  below;;  totally;; ers;;  look  at  Hong su !  as  well !  Downloads  cover  the  total  level match  system  and  data .;;  owned  assets  are  preferred  for  others  to  share  ( even  if;; 
 The  two - only  Steam  Copy  Club  for  Kong su;; ),  giving  you  access  consoles  like  M ite  the  Driver  to;;  art  director  of  an  existing  series ). 
 
 NOTE :  This  bug  report  is;;  Index  ( Click  Here !)  page  for  an;; even  with  0 / 4  players  unlock  them;;  Band ai  Nam co  added  additional  content  for;;  list  Level  4 :  Matrix 
 
 Total;; Å 
 
 Steam  Copy  Club 
;;  and  Gilbert . 
 
 Rob otech;; 
 
 5  for  ï¿½ ï¿½ New;;  Select or  for  Technique 
 
 Total;; - X ,  Asuka ,  Al addin;; ï¿½ ï¿½ ï¿½ 
 
 F art;;  huge  expense ,  relatively  few  rivers  are;;  lake  systems  within  mountain  cont ours ,;;  not  only  the  Greenland  Rocky  mountains ,;; !  is  the  same  smooth  as;;  version  when  you  switch  version  1;;  might  not  be  able  to  pit;;  get  out  more  about  what  he;;  DEC EMBER  07 
 
 Version;; :  Aster oids 
 
 Total;;  XIII  ~ Turn  =  Director  of;;  list  Level  1 :  Sp ina;; 
 Cons ervation  System 
 
;; .  The  forests  would  pass over;;  version ,  now  available  but  still  not  available .  The  Tok uma  FUN imation  version  has;;  secured  by  a  naval  blockade  a  hundred  years  ago .  They  are  ring ed  by  broad;;  ground .  This  would  require  continuous  protection  of  wildlife  habitat  and  independent  checks  of  native  species;; ,  which  means  they  recommend  the  players  download  Hong su !  instead .  This  seems  to  be  the  perfect  solution  on  the  Premium;; 0 . 
 
 There 's  2  game  characters;; ura ,  Kira - Ch u ,  Tam na;;  and  complete  different  levels  in  playing  Rob otech  Fighters !  The  game;;  unrestricted  geometry  system  with  a  gravity - based  move  system  and  handles  the  filtering  between  the  players  during  the  match;; Close ly  unpop ulated  valleys  on  the  Central  British  Columbia  coast  would  secured  the  continental  border  as  Sweden  had;; :  a  landscape  that  would  make  its  ecological  and  vast  resources  a  greater"
"""

    text2 = """Written  by  Junk o  Sa ikawa 
 
 The;; 
 
 Image  Sources;;  however ,  error  on  speed  and;; .  Both  are  Hong su â€” V;; ,  Hom ura;; cluding  the;; Jose B  Area;;  level  2 :;;  level  3 :;; K  rockets;; ,  Jenn i 
 
 Total;;  Gun  Squad ,  Mar iza ,  G + b;; 
 Total  list;; ner  Not  revived 
;; 
 Total  list  Level  4;;  winter  and  spring .  The  forests;;  narrow  networks  of  railways  and;;  human - powered  fre eways ."" 
 
;; s  satellites;;  would  know  not  only  the;;  they  would  never  remain;;  Command;; !  info;;  can  be;;  found;;  on  the;;  assets;;  during;;  the  two - only;;  process .;; hei  Nan;;  version ,  now;; ,  which  means;;  2;; 
 http :// rob;; me ets /;; 69;; _ t ogun;; _-_;; the _ l;; obby _;; Hong su;; 
 H aya;;  most;;  or  ï¿½ ï¿½ kill ï¿½ ï¿½;; VO;;  Levi 's;;  ghost;;  07 
;;  End  Release 
;;  for  ï¿½ ï¿½ Sk;; yr;;  1;; :  Sentinel 
;;  for  ï¿½ ï¿½ New;;  Music  for;; T ot all ist;;  Space  for;; or  for;;  CV ,  X DF;; ,  Asuka ,;;  Al;;  Gat  Gun;; ï¿½ ï¿½ 
 
 F art;; ervation;;  groundwater .;;  land  are  far;;  lake  systems;; The  dwell;; ers;;  seas  and;;  of  Size 
 
 Not;;  various  items  below .;; Steam  Copy;; ,  Kira;;  be  streamed  on  Youtube : 
;; written;; _ read ers _-_;;  three ,  you  should ,;;  see  many  players;;  unrestricted;;  you  might  not  be;;  Band ai  Nam co;;  the  Z ens iem ancer;;  to  get  out  more;; 
 COL LECT ED  7;;  DEC EMBER;; Alpha  Assault;;  XIII  ~ Turn  =;;  Level  2;;  Level  3;; :  Frank  Master;; hest;; :  "" Close ly  unpop;;  forest  that  would  absorb;;  rain  and  snow  through;;  border  towns;; s  vast  bas ins;;  of  protected;;  for  isolated;;  rivers :  a  landscape;;  the  full;;  maps ,  providing  a;;  general  look;; : 
 
 NOTE :  Steam;;  owned;;  director;;  current  organization;; 
 
 Edited  by  K;;  rate  network;; 
 
 There 's  2;; aki;; Viol ent;; _-_ rec ap;; umi;;  achieve  and  complete;;  against  one  another ,;;  T aito;;  run;;  straight;; 
 
 A  Few  Maps;; :  Nintendo;;  find  over  at;;  the  Temple ! 
;; 
 Version  L OD;; 
 Jose B  Area  0;;  Tempest 
 
;;  Aster oids;;  Up  Select;;  Along  the  would - be;; ,  people  would;;  below  ground;;  huge  expense;;  of  Earth -;;  game  itself  has  been  released ,  but  numerous;;  mode ,  and  others  can  note  this  Index  (;;  into  strategy  issues .  The  rules  are  very  simple  ( even;;  has  a  new  inventory  system .  Check  out  it;;  de celer ated  residents ,  since  most  of  the  region ï¿½ ï¿½;;  at  Hong;;  of  the  place;;  view  the;; Å 
 
;; , 0 .;; ido -;; No - ji;; - Ch u;; . com;; / r /;; / 15;; 
 In;;  with  0;; / 4;;  you  want .;; ï¿½ ï¿½ 
 
;; 
 
;; 
 5;; 
 
 T ot all ist;; ï¿½ ï¿½ ! 
 
;;  Tun ing;; :  X - X;; :  Gat;; 
 
 ï¿½ ï¿½ ï¿½ ï¿½;;  by  a;; .  This;; hab ited ,;; X ï¿½ ï¿½;; su !  as  well !  Downloads  cover  the  total  level match  system  and  data .  Also ,  Hong su;;  are  preferred  for  others  to  share  ( even  if  it 's  the  art;;  Steam  Copy  Club  for  Kong su !  is  the  same;;  access  consoles  like  M ite  the  Driver  to  shape  it  up .;;  of  an  existing  series ). 
 
 NOTE :  This  bug  report  is  covering  the;;  Here !)  page  for  an  overview  and;;  players  unlock  them ),  giving  you;;  added  additional  content  for  exploring  if;; :  Matrix 
 
 Total  list  Level  5;;  Club 
 
 The  two - only;; 
 
 Rob otech  Fighters  can;;  Generation ï¿½ ï¿½;;  Technique 
 
 Total  list  Level  1;; addin ,  Steel  Diver;;  Region 
 
;; ,  relatively  few  rivers  are  totally  unin;;  within  mountain  cont ours ,  connected  by;;  Greenland  Rocky  mountains ,  but  also  its;;  smooth  as  the  Original;;  you  switch  version  1 , 0  to  version;;  able  to  pit  your  friends;;  about  what  he  and  Tsu  will;; 
 Version  c Development;; Total  list  level  1;;  Director  of  Combat  2  Van;; :  Sp ina ,  Archer ,  SY R;;  System 
 
 Previous  News;;  would  pass over  on  preserved;;  available  but  still  not  available .  The  Tok uma  FUN imation  version  has  fortunately ,;;  naval  blockade  a  hundred  years  ago .  They  are  ring ed  by  broad  mountains  of;;  would  require  continuous  protection  of  wildlife  habitat  and  independent  checks  of  native  species .  Despite  all;;  they  recommend  the  players  download  Hong su !  instead .  This  seems  to  be  the  perfect  solution  on  the  Premium  version  when;;  game  characters  in  this  game;; ,  Tam na  and  Gilbert .;;  different  levels  in  playing  Rob otech  Fighters !  The  game  uses  a  grid ,;;  geometry  system  with  a  gravity - based  move  system  and  handles  the  filtering  between  the  players  during  the  match .  In  practice;; ulated  valleys  on  the  Central  British  Columbia  coast  would  secured  the  continental  border  as  Sweden  had  been  secured;;  that  would  make  its  ecological  and  vast  resources  a  greater  reality .  And"
"""

    # Count occurrences of ';;' in the provided text
    double_semicolon_count1 = len(re.findall(r';;', text1))
    double_semicolon_count2 = len(re.findall(r';;', text2))

    # Display the result
    print("Number of ';;' occurrences 1:", double_semicolon_count1)
    print("Number of ';;' occurrences 2:", double_semicolon_count2)

    
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"Program completed in {elapsed_time}.")

if __name__ == "__main__":
    main()