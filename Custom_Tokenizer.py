class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.tokens = []
        self.unk_token = "[UNK]"
        
    def train(self, texts):
        self.vocab[self.unk_token] = 0
        self.tokens.append(self.unk_token)
        
        for text in texts:
            words = self._tokenize(text)
            
            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = len(self.tokens)
                    self.tokens.append(word)
    
    def _tokenize(self, text):
        for punct in ".,!?;:()[]{}\"'":
            text = text.replace(punct, f" {punct} ")
            
        return [word for word in text.split() if word]
    
    def encode(self, text):
        words = self._tokenize(text)
        return [self.vocab.get(word, self.vocab[self.unk_token]) for word in words]
    
    def decode(self, ids):
        tokens = [self.tokens[id] if id < len(self.tokens) else self.unk_token for id in ids]
        
        text = " ".join(tokens)
        
        for punct in ".,!?;:":
            text = text.replace(f" {punct}", punct)
            
        return text
    
    def vocab_size(self):
        return len(self.tokens)


if __name__ == "__main__":
    training_texts = [
        "Hello world!",
        "This is a simple example.",
        "Tokenizers split text into smaller pieces."
    ]
    
    tokenizer = SimpleTokenizer()
    tokenizer.train(training_texts)
    
    print(f"Vocabulary size: {tokenizer.vocab_size()}")
    print("Vocabulary items:")
    for token, id in sorted(tokenizer.vocab.items(), key=lambda x: x[1]):
        print(f"  {id}: {token}")
    
    test_text = "Hello world, this is an example!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nOriginal: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    new_text = "This contains unseen words like giraffe and elephant!"
    encoded_new = tokenizer.encode(new_text)
    decoded_new = tokenizer.decode(encoded_new)
    
    print(f"\nText with new words: {new_text}")
    print(f"Encoded: {encoded_new}")
    print(f"Decoded: {decoded_new}")