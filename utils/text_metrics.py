"""
utils/text_metrics.py — Self-contained text-generation metrics.

BLEU-1..4, ROUGE-L, and a simplified CIDEr-D, implemented with no external
dependencies (no nltk / pycocoevalcap) so they run anywhere.

IMPORTANT — honest scope note for this project:
    The trained pipeline uses `llm_backend='cross_attn_only'`, which does NOT
    free-generate text. It performs cross-attention fusion + 6-class
    classification. The "reasoning response" sentence is templated from the
    PREDICTED class (see CLASS_DESCRIPTIONS in the demo). Therefore, when these
    metrics are computed with the templated response as the hypothesis and the
    dataset annotation as the reference, they measure how well the
    predicted-class description matches the ground-truth description — i.e. a
    text-space reflection of classification quality, NOT free caption
    generation. Report them as such in any paper.
"""

import math
from collections import Counter
from typing import List, Sequence


def _tokenize(text: str) -> List[str]:
    """Lowercase whitespace tokenizer, stripping basic punctuation."""
    out = []
    for tok in text.lower().split():
        tok = tok.strip(".,;:!?'\"()[]{}")
        if tok:
            out.append(tok)
    return out


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    return Counter(
        tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)
    )


# ─── BLEU ─────────────────────────────────────────────────────────────────────

def sentence_bleu(hypothesis: str, reference: str, max_n: int = 4) -> float:
    """Single-sentence BLEU with +1 (Laplace) smoothing on n-gram precisions.

    Returns BLEU-`max_n` in [0, 1].
    """
    hyp = _tokenize(hypothesis)
    ref = _tokenize(reference)
    if len(hyp) == 0:
        return 0.0

    log_prec_sum = 0.0
    for n in range(1, max_n + 1):
        h = _ngram_counts(hyp, n)
        r = _ngram_counts(ref, n)
        overlap = sum((h & r).values())
        total = max(sum(h.values()), 1)
        # add-1 smoothing keeps log finite for short sentences / missing n-grams
        precision = (overlap + 1.0) / (total + 1.0)
        log_prec_sum += math.log(precision)

    geo_mean = math.exp(log_prec_sum / max_n)

    # Brevity penalty
    h_len, r_len = len(hyp), len(ref)
    if h_len > r_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - r_len / max(h_len, 1))

    return bp * geo_mean


def corpus_bleu(hypotheses: List[str], references: List[str],
                max_n: int = 4) -> float:
    """Mean sentence-BLEU over a corpus (simple and stable for short texts)."""
    if not hypotheses:
        return 0.0
    scores = [sentence_bleu(h, r, max_n)
              for h, r in zip(hypotheses, references)]
    return sum(scores) / len(scores)


def bleu_n_breakdown(hypotheses: List[str], references: List[str]) -> dict:
    """Return BLEU-1, BLEU-2, BLEU-3, BLEU-4 over the corpus."""
    return {
        f"BLEU-{n}": corpus_bleu(hypotheses, references, max_n=n)
        for n in (1, 2, 3, 4)
    }


# ─── ROUGE-L ──────────────────────────────────────────────────────────────────

def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Longest common subsequence length (DP, O(|a||b|))."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[len(b)]


def sentence_rouge_l(hypothesis: str, reference: str, beta: float = 1.2) -> float:
    """ROUGE-L F-measure for a single sentence pair."""
    hyp = _tokenize(hypothesis)
    ref = _tokenize(reference)
    if not hyp or not ref:
        return 0.0
    lcs = _lcs_length(hyp, ref)
    if lcs == 0:
        return 0.0
    prec = lcs / len(hyp)
    rec = lcs / len(ref)
    if prec + rec == 0:
        return 0.0
    beta2 = beta * beta
    return ((1 + beta2) * prec * rec) / (rec + beta2 * prec)


def corpus_rouge_l(hypotheses: List[str], references: List[str]) -> float:
    if not hypotheses:
        return 0.0
    return sum(sentence_rouge_l(h, r)
               for h, r in zip(hypotheses, references)) / len(hypotheses)


# ─── Simplified CIDEr-D ─────────────────────────────────────────────────────--

def corpus_cider(hypotheses: List[str], references: List[str],
                 max_n: int = 4) -> float:
    """A simplified single-reference CIDEr-D.

    TF-IDF-weighted n-gram cosine similarity, averaged over n=1..4 and over the
    corpus. IDF is computed from the reference set. With a single reference per
    image this is an approximation of the full multi-reference CIDEr, but it is
    deterministic and dependency-free. Reported on the same 0..10-ish scale as
    CIDEr (n-gram average × 10 is the convention; here we keep the raw [0,1]
    cosine mean and note it explicitly).
    """
    if not hypotheses:
        return 0.0

    refs_tok = [_tokenize(r) for r in references]

    # Document frequency of each n-gram across the reference corpus
    total = []
    for n in range(1, max_n + 1):
        df = Counter()
        for rt in refs_tok:
            df.update(set(_ngram_counts(rt, n).keys()))
        num_docs = max(len(refs_tok), 1)

        def _tfidf(counts: Counter) -> dict:
            vec = {}
            length = max(sum(counts.values()), 1)
            for g, c in counts.items():
                tf = c / length
                idf = math.log((num_docs + 1.0) / (df.get(g, 0) + 1.0))
                vec[g] = tf * idf
            return vec

        sims = []
        for h, rt in zip(hypotheses, refs_tok):
            hv = _tfidf(_ngram_counts(_tokenize(h), n))
            rv = _tfidf(_ngram_counts(rt, n))
            keys = set(hv) | set(rv)
            dot = sum(hv.get(k, 0.0) * rv.get(k, 0.0) for k in keys)
            hn = math.sqrt(sum(v * v for v in hv.values()))
            rn = math.sqrt(sum(v * v for v in rv.values()))
            sims.append(dot / (hn * rn) if hn > 0 and rn > 0 else 0.0)
        total.append(sum(sims) / len(sims))

    return sum(total) / len(total)


# ─── Convenience: all text metrics at once ────────────────────────────────────

def compute_all_text_metrics(hypotheses: List[str],
                             references: List[str]) -> dict:
    """Return BLEU-1..4, ROUGE-L and simplified CIDEr for a hyp/ref corpus."""
    out = bleu_n_breakdown(hypotheses, references)
    out["ROUGE-L"] = corpus_rouge_l(hypotheses, references)
    out["CIDEr"] = corpus_cider(hypotheses, references)
    return {k: round(float(v), 4) for k, v in out.items()}
