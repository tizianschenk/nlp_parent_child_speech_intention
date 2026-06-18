# nlp_parent_child_speech_intention
The goal of this project is to automatically classify what young children say during playtime into four speech categories: Prosocial Talk, Question, Command, and Negative Talk. This is based on a standard clinical system used by therapists to evaluate parent-child interactions. Because child speech data is rare and hard to label, we started by simulating a low-resource scenario using a tiny training set of just 32 examples (8 per class). Under these strict conditions, standard models overfit immediately. To solve this, we tested several data-lean techniques using a lightweight sentence transformer model (`all-MiniLM-L6-v2`), comparing traditional keyword rules, zero-shot models, basic text edits (EDA), and AI-generated synthetic data. 

We compared our models against the benchmarks from the original *Playlogue* paper, which used GPT-4 and a large, fully trained RoBERTa model. When data was limited, generating synthetic training data with an LLM worked best, matching our manual keyword rules. When we scaled up and trained our SetFit model on the full dataset, it easily beat GPT-4's zero-shot scores and matched the state-of-the-art RoBERTa model's  accuracy, despite our model being much smaller and faster.

| Model / Approach | Data Used | Accuracy | Macro F1 |
| :--- | :---: | :---: | :---: |
| **Low-Resource Baseline** (Random) | 0 examples | 61.00% | 25.00% |
| **Rule-Based** (Keyword Matching) | 0 examples | 83.00% | 61.00% |
| **SetFit + LLM Synthetic Data** | 32 real + 80 AI examples | 82.00% | 62.00% |
| **SetFit (Full Dataset)** | Full Dataset (~3,000 examples) | **94.24%** | **61.70%** |
| **SetFit + LLM (Full Dataset)** | Full Dataset + AI examples | **93.42%** | **64.59%** |
| *SOTA Benchmark: GPT-4 (Zero-Shot)* | 0 examples | 87.35% | 52.87% |
| *SOTA Benchmark: RoBERTa-Large (Full Data)* | Full Dataset (~3,000 examples) | 92.02% | 72.46% |

Our results show that deep learning models excel at learning broad semantic meaning from full datasets, eliminating the need for complex, manual keyword rules. Our full-data model naturally learned the real-world imbalances of clinical practice. It correctly defaulted to Prosocial Talk when uncertain, which significantly cut down on false alarms for Negative Talk. However, a major hurdle remains: child sentences are incredibly short (averaging just 3 words), ungrammatical, and context-dependent. For example, a child saying a random noun during play can look like a Command to a model, and sentences containing the word "no" are often mislabeled as Negative Talk even when they aren't. Ultimately, text-only models hit a natural ceiling here. To truly bridge the gap on minority classes, future models must look beyond single sentences and take the surrounding conversational history into account more efficiently.

Given these constraints, we also explored whether the best-performing model could be made lighter for deployment. In order to achieve this, we distilled the best model into an MLP "student", which matched the "teacher's" F1: for a loss of only 0.01 points in this metric, a 14% reduction was achieved. Moreover, we also applied quantization, which still showed a modest loss in F1. However, it was way slower than both the "teacher" and the distilled ("student") models, as it was forced to run on CPU, something we acknowledge as a limitation. Lastly, the Negative Talk class remained the weakest point across all the compression setups, leading us to the conclusion that a data problem cannot be fixed by any of these techniques.

***Note:*** The notebook main.ipynb should be mainly ran in Google Colab. To run it locally, access the uv files in the Github. Additionally, due to computational constraints, the notebook is dependent on external folders generated during the training process. These are currently not on the github due to size constraints. These are available upon request but we recommend not running the notebook without them in order to not overwrite the outputs.

Made by: Daniel Campos, Joshua Castillo, Corneel Moons, Tizian Schenk