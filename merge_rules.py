import collections, multiprocessing, json, pathlib, math
import bitarray, transformers, torch, numpy as np

class BitMatrix:
    def __init__(self, n: int, m: int, init_val: int = 0):
        self.data = bitarray.bitarray(n*m)
        self.n = n
        self.m = m
        self.data.setall(init_val)

    def __getitem__(self, I: tuple):
        return self.data[I[0]*self.n+I[1]]

    def __setitem__(self, I: tuple, v: int):
        if isinstance(I[1], slice):
            k = I[0]*self.n
            start = 0 if I[1].start is None else I[1].start
            stop = self.m if I[1].stop is None else I[1].stop
            self.data[k+start:k+stop:I[1].step] = v
        else: self.data[I[0]*self.n+I[1]] = v

def _empty_set(): return set()
def _inf_f(): return math.inf

class MergeRules:
    def __init__(self, mdl: str):
        tokdir = f"{pathlib.Path(__file__).parent.resolve()}/tokenizers/{mdl.lower()}"
        with open(f"{tokdir}/tokenizer.json", "r") as f:
            tok_json = json.load(f)
        K = [tuple(x.split(sep=' ')) for x in tok_json["model"]["merges"]]
        self.K = collections.defaultdict(_empty_set)
        for a, b in K: self.K[a].add(b)
        self.tok = transformers.AutoTokenizer.from_pretrained(tokdir)
        self.special_tokens_set = set(self.tok.all_special_ids)
        self.whitespace_token = self.tok.convert_ids_to_tokens(self.tok.encode(" ", add_special_tokens=False)[0])[0]
        self.V = self.tok.vocab
        self.U = {self.V[v]: v for v in self.V}
        self.K_ids = {(u := self.V[y]): {self.V[x] for x in self.K[y]} for y in self.K}
        self.pK_merged = collections.defaultdict(_inf_f)
        self.pK = {a: {} for a, _ in K}
        self.pair_K = {}
        #for v in self.V: self.pK_merged[v] = 0
        for i, (a, b) in enumerate(reversed(K)):
            j = len(K)-1-i
            self.pK_merged[a+b] = j
            self.pK[a][b] = j
            self.pair_K[a+b] = [a, b]

        # self.M = BitMatrix(len(self.V), len(self.V), 1)
        # for i, v in enumerate(tqdm.tqdm(self.V, desc="Constructing mask")):
            # if v not in self.K: continue
            # for u in self.K[v]:
                # self.M[i,self.U[u]] = False

    def mask(self, T: torch.LongTensor, X: torch.Tensor, stream: list) -> np.ndarray:
        X[:] = 0
        for i in range(X.shape[0]):
            if len(stream[i]) == 0: continue
            t = T[i,-1].item()
            for v in self.K[self.U[t]]:
                token_id = self.V[v]
                X[i,token_id] = 1e-16
            for v in self.V:
                skip = False
                for j in range(len(stream[i])):
                    for l in range(len(v), 0, -1):
                        if (stream[i][j:] + v[:l]) in self.V:
                            X[i,self.V[v]] = 1e-16
                            skip = True
                            break
                    if skip: break
        return X

    def mask_parallel(self, T: torch.LongTensor, X: torch.Tensor, stream: list):
        X[:] = 0
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            P = [pool.apply_async(MergeRules.mask_parallel_f, (T[i,-1].item(), stream[i], self)) for i in range(T.shape[0])]
            R = [p.get() for p in P]
            for i, r in enumerate(R): X[i,r] = 1e-16
        return X

    @staticmethod
    def mask_parallel_f(token_id: int, stream: str, MR) -> set:
        if len(stream) == 0: return slice(None)
        I = set()
        for v in MR.K[token_id]: I.add(MR.V[v])
        for v in MR.V:
            skip = False
            for j in range(len(stream)):
                for l in range(len(v), 0, -1):
                    if (stream[j:] + v[:l]) in MR.V:
                        I.add(MR.V[v])
                        skip = True
                        break
                if skip: break
        return list(I)