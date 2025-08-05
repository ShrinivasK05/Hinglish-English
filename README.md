# Hinglish-to-English Code-Mixed Translation using mBART

This project implements a **two-stage curriculum learning approach** using the **facebook/mbart-large-cc25** model to translate **code-mixed Hinglish** (Hindi-English) sentences into standard English. It sets a new state-of-the-art performance on the PHINC dataset with a **BLEU score of 31.6**, surpassing previous benchmarks.

## 🚀 Project Highlights

- **Model**: mBART (Multilingual BART) - `facebook/mbart-large-cc25`
- **Approach**: Two-stage curriculum learning
  - Stage 1: Auxiliary training (English → Hinglish)
  - Stage 2: Main training (Hinglish → English)
- **Dataset**: PHINC (13,789 Hinglish-English sentence pairs)
- **BLEU Score**: 31.6 (7.1% improvement over prior SOTA)

## 📂 Dataset

- **PHINC Dataset**: Contains 13,789 parallel Hinglish-English sentences.
- Preprocessing includes:
  - Punctuation normalization
  - Transliteration inconsistency cleanup
  - Sentence length filtering (3–50 tokens)

## 🛠️ Tools and Libraries

- Python, PyTorch
- Hugging Face Transformers
- SentencePiece Tokenizer (mBART)
- sacreBLEU for evaluation

## 📌 Methodology

1. **Preprocessing**
   - Noise removal, punctuation normalization
   - Sentence length filtering
2. **Tokenization**
   - mBART tokenizer (sentencepiece)
   - Padding/truncating to 128 tokens
3. **Model Configuration**
   - Encoder-decoder transformer (610M parameters)
   - Language code set to `en_XX` for both source and target
4. **Training**
   - Optimizer: AdamW with weight decay
   - Scheduler: Linear warm-up
   - Gradient clipping for stability
   - Stage 1: 2–4 epochs at 5e-5 LR (English → Hinglish)
   - Stage 2: 3–6 epochs at 3e-5 LR (Hinglish → English)
5. **Evaluation**
   - sacreBLEU (BLEU-4) on test set (2,789 samples)
   - Real-time interactive inference module

## 📈 Results

| Model                            | BLEU Score |
|----------------------------------|------------|
| Srivastava & Singh (2020)        | 15.3       |
| Gupta et al. (2021, mT5)         | 29.5       |
| **Our Model (2-Stage mBART)**    | **31.6**   |

Sample Translation:

| Hinglish Input                                | English Output                                   |
|-----------------------------------------------|--------------------------------------------------|
| `Main surprised hu ye itna low hai`           | `I'm surprised how low it is.`                   |
| `Kal party mein aana hai kya?`                | `Are you coming to the party tomorrow?`         |

## 💡 Future Work

- Extend to other code-mixed languages (e.g., Tamlish, Benglish)
- Integrate language identification as an auxiliary task
- Use transformer variants (BigBird, Longformer) for longer sequences
- Deploy real-time inference in production environments (e.g., WhatsApp/Twitter)

## 🙏 Acknowledgements

Special thanks to:
- PES University (CCBD & CDSAML)
- Hugging Face, PyTorch, and the open-source community
- PHINC dataset authors for enabling research on code-mixed translation

---

