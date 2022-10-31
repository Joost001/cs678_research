import sentencepiece as spm

class CovostTokenizer(object):
    '''
    Create a tokenizer from the text corpus given.
    '''
    def __init__(self, corpus_root:str, src_lan: str, tgt_lan: str, vocab_size: int):
        self.corpus_root = corpus_root
        self.src_lan = src_lan
        self.tgt_lan = tgt_lan
        self.vocab_size = vocab_size

        self.model_prefix = 'm'+self.src_lan+'_'+self.tgt_lan
        spm.SentencePieceTrainer.train(input=self.corpus_root, vocab_size=self.vocab_size, model_prefix=self.model_prefix,
                                        pad_id=0, unk_id=3, bos_id=1, eos_id=2, pad_piece='<pad>', 
                                        unk_piece='<unk>', bos_piece='<s>', eos_piece='</s>')
        
        sp = spm.SentencePieceProcessor()
        self.sp = sp
        
    def tokenize(self, tgt_texts: list):
        '''
        Tokenize the translations given at tgt_texts, and return
        a list with the translations tokenized and padded.
        '''
        self.sp.load(self.model_prefix+'.model')
        max_len = 0
        tokens = []
        for tgt in tgt_texts:
            tokens.append(self.sp.encode_as_ids(tgt))
            if max_len<len(tgt):
                max_len = len(tgt)

        final_tokens = []
        for t in tokens:
            final_tokens.append([self.sp.bos_id()] + t + [self.sp.eos_id()] + [self.sp.pad_id()]*max(0,max_len-len(t)))

        return final_tokens