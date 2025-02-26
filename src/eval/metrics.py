from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
import nltk
import jieba
from transformers import AutoTokenizer
import numpy as np
from collections import Counter
import logging
import re

jieba.setLogLevel(logging.INFO)

class UnifiedEvaluator:
    def __init__(self, max_seq_length=510):
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        self.bleu_weights = [
            (1.0, 0, 0, 0),  
            (0.5, 0.5, 0, 0), 
            (0.33, 0.33, 0.33, 0), 
            (0.25, 0.25, 0.25, 0.25)  
        ]

        self.smoothie = SmoothingFunction().method1
        self.tokenizer = AutoTokenizer.from_pretrained(
            "hfl/chinese-bert-wwm",
            use_fast=True  
        )
        self.max_seq_length = max_seq_length
        self._init_jieba()

    def _init_jieba(self):
        jieba.initialize()

    def calculate_all_metrics(self, preds, refs, keyword_lists):

        return {
            **self._get_rouge(preds, refs),
            **self._get_bert_score(preds, refs),
            **self._get_bleu(preds, refs),
            **self._get_keyword_accuracy(keyword_lists, preds),
            **self._get_char_f1(preds, refs),
            **self._get_meteor(preds, refs)
        }
    
    def _remove_punctuation(self, text):
        return re.sub(r'[\W]+', '', text)

    def _get_rouge(self, preds, refs):
        f_scores = []
        for p, r in zip(preds, refs):
            if not p.strip() or not r.strip():
                continue
            
            p = self._preprocess_text(p)
            r = self._preprocess_text(r)

            scores = self.scorer.score(r, p)
            f_scores.append(scores["rougeL"].fmeasure)
        
        return {"rouge-l": round(np.mean(f_scores).item(), 4)} if f_scores else {"rouge-l": 0.0}

    def _safe_truncate(self, text):
        if not isinstance(text, str) or not text.strip():
            return ""
        
        try:
            tokens = self.tokenizer.encode(
                text,
                truncation=True,
                max_length=self.max_seq_length,
                add_special_tokens=False
            )
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        except Exception as e:
            print(f"truncation error: {str(e)}")
            return text[:self.max_seq_length]

    def _get_bert_score(self, preds, refs):
        try:
            truncated_preds = [self._safe_truncate(p) for p in preds]
            truncated_refs = [self._safe_truncate(r) for r in refs]
            
            valid_pairs = [
                (p, r) for p, r in zip(truncated_preds, truncated_refs)
                if p.strip() and r.strip()
            ]
            if not valid_pairs:
                return {"bert-precision": 0.0, "bert-recall": 0.0, "bert-f1": 0.0}
            
            valid_preds, valid_refs = zip(*valid_pairs)

            P, R, F1 = bert_score(
                cands=valid_preds,
                refs=valid_refs,
                lang="zh",
                model_type="hfl/chinese-bert-wwm",
                num_layers=8,          
                batch_size=32,         
                nthreads=4,            
                rescale_with_baseline=True,
                verbose=True           
            )
            
            return {
                "bert-precision": round(P.mean().item(), 4),
                "bert-recall": round(R.mean().item(), 4),
                "bert-f1": round(F1.mean().item(), 4)
            }
        except Exception as e:
            print(f"BERTScore Error: {str(e)}")
            return {"bert-precision": 0.0, "bert-recall": 0.0, "bert-f1": 0.0}

    def _preprocess_text(self, text):
        '''preprocess'''
        text = re.sub(r'\*\*.*?\*\*', '', text)
        text = re.sub(r'\*\*|^\d+\.\s+|-\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n+|\s+', ' ', text)
        return text.strip()
    
    def _tokenize_text(self, text):
        tokens = list(jieba.cut(text))
        tokens = [t for t in tokens if not re.match(r'[^\w\s]', t)]

        return tokens

    def _get_bleu(self, preds, refs):
        bleu_scores = {f'bleu-{i+1}': [] for i in range(4)}
        
        for p, r in zip(preds, refs):
            if not p.strip() or not r.strip():
                continue
            
            p = self._preprocess_text(p)
            r = self._preprocess_text(r)

            p_tokens = self._tokenize_text(p)
            r_tokens = self._tokenize_text(r)

            try:
                for i, weights in enumerate(self.bleu_weights):
                    score = sentence_bleu(
                        [r_tokens], 
                        p_tokens, 
                        weights=weights,
                        smoothing_function=self.smoothie
                    )

                    bleu_scores[f'bleu-{i+1}'].append(score)
            except Exception as e:
                print(f"ERROR: {str(e)}")
                continue

        return {k: round(np.mean(v).item(), 4) if v else 0.0 
                for k, v in bleu_scores.items()}

    def _get_keyword_accuracy(self, keyword_lists, preds):
        accuracies = []
        for keywords, pred in zip(keyword_lists, preds):
            try:
                if not keywords or not isinstance(keywords, list):
                    continue
                
                hits = sum(1 for kw in keywords if kw in pred)
                
                if len(keywords) > 0:
                    accuracies.append(hits / len(keywords))
            except Exception as e:
                print(f"Error processing keywords {keywords} with prediction {pred}: {e}")
                accuracies.append(0.0)
        return {"keyword_accuracy": round(np.mean(accuracies).item(), 4) if accuracies else 0.0}

    def _get_char_f1(self, preds, refs):
        precision_scores, recall_scores, f1_scores = [], [], []
        
        for p, r in zip(preds, refs):
            try:

                p = self._preprocess_text(p)
                r = self._preprocess_text(r)
                p_tokens = self._tokenize_text(p)
                r_tokens = self._tokenize_text(r)

                p_tokens = list(p_tokens)
                r_tokens = list(r_tokens)               
                
                common = Counter(p_tokens) & Counter(r_tokens)
                num_same = sum(common.values())
                
                if num_same == 0:
                    continue
                    
                precision = 1.0 * num_same / len(p_tokens) if p_tokens else 0
                recall = 1.0 * num_same / len(r_tokens) if r_tokens else 0
                f1 = self._safe_f1(precision, recall)
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
            except Exception as e:
                logging.error(f"Char F1 Error: {str(e)}")
                precision_scores.append(0.0)
                recall_scores.append(0.0)
                f1_scores.append(0.0)
        return {
            "char_precision": round(np.mean(precision_scores).item(), 4) if precision_scores else 0.0,
            "char_recall": round(np.mean(recall_scores).item(), 4) if recall_scores else 0.0,
            "char_f1": round(np.mean(f1_scores).item(), 4) if f1_scores else 0.0
        }

    def _normalize(self, text):
        return text.lower().strip().replace(" ", "").replace("\n", "").replace("\t", "")


    def _safe_f1(self, p, r):
        denominator = p + r
        return 2 * p * r / denominator if denominator > 1e-9 else 0.0
    
    def _clean_text(self, text):
        text = re.sub(r'[\*\n\r]', '', text) 
        text = re.sub(r'\s+', ' ', text) 
        text = text.strip() 
        return text

    def _preprocess_meteor(self, text):
        return ' '.join(jieba.cut(text))

    def _get_meteor(self, preds, refs):
        nltk.download('wordnet')
        scores = []
        for p, r in zip(preds, refs):
            p=self._clean_text(p)
            r=self._clean_text(r)
            p_processed = self._preprocess_meteor(p).split()
            r_processed = self._preprocess_meteor(r).split()

            try:
                score = meteor_score(
                    [r_processed],
                    p_processed
                )
                scores.append(score)
            except Exception as e:
                print(f"METEOR ERROR: {str(e)}")
                scores.append(0.0)
        
        return {"meteor": round(np.mean(scores).item(), 4)} if scores else {"meteor": 0.0}