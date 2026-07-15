import collections
import json
import os

class BPETokenizer:
    def __init__(self, merges=None):
        self.merges = merges if merges is not None else {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        # Reconstruct vocab from merges
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        self.vocab_size = 256 + len(self.merges)

    def train(self, text, num_merges, sample_size=300000):
        tokens = list(text.encode('utf-8'))[:sample_size]
        for i in range(num_merges):
            stats = collections.Counter(zip(tokens, tokens[1:]))
            if not stats: 
                break
            best_pair = max(stats, key=stats.get)
            new_idx = 256 + i
            self.merges[best_pair] = new_idx
            self.vocab[new_idx] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            # Apply merge to current token stream
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and (tokens[j], tokens[j+1]) == best_pair:
                    new_tokens.append(new_idx)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens

    def encode(self, text):
        # Chunk text to avoid O(N^2) behavior on large corpus
        chunk_size = 10000
        raw_bytes = list(text.encode('utf-8'))
        
        if not self.merges:
            return raw_bytes
            
        final_tokens = []
        for offset in range(0, len(raw_bytes), chunk_size):
            tokens = raw_bytes[offset:offset+chunk_size]
            while len(tokens) >= 2:
                stats = collections.Counter(zip(tokens, tokens[1:]))
                min_idx = float('inf')
                best_pair = None
                
                for pair in stats:
                    if pair in self.merges and self.merges[pair] < min_idx:
                        min_idx = self.merges[pair]
                        best_pair = pair
                        
                if best_pair is None:
                    break
                    
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                        new_tokens.append(min_idx)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
            final_tokens.extend(tokens)
        return final_tokens

    def decode(self, tokens):
        b = bytearray()
        for t in tokens:
            b.extend(self.vocab.get(t, bytes([0])))
        return b.decode('utf-8', errors='replace')
        
    def save(self, path):
        serializable_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        with open(path, 'w') as f:
            json.dump(serializable_merges, f)

def load(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'bpe_merges.json')
    merges = {}
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        merges = {tuple(map(int, k.split(','))): v for k, v in data.items()}
    return BPETokenizer(merges)
