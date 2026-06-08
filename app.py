import os
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import streamlit as st
import torch
import math
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer as rs
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass
st.set_page_config(page_title="Transformer Hub", layout="wide", page_icon="🤖")

with st.sidebar:
    st.title("🤖 Transformer Hub")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📚 Comparative Analysis",
        "✍️ Generator",
        "🔬 Architectures"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption("MSc AI/ML · Aditya Sharma")

# ══════════════════════════════════════════════════════════════
# PAGE 1 — ANALYSIS
# ══════════════════════════════════════════════════════════════
if page == "📚Comparative Analysis":
    st.title("📚 Comparative Analysis of Transformer Models")
    st.write("Page loaded successfully")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Technical Comparison", "Quantitative", "Strengths & Weaknesses", "Applications & Conclusion"
    ])

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🔷 Encoder-Only")
            st.markdown("**Models:** BERT, RoBERTa, DistilBERT")
            st.info("""
**How it works:**
- Full **bidirectional** self-attention
- Trained with **MLM** — predict masked tokens
- `[CLS]` token used for classification

**Strengths:**
- Best NLU, NER, classification, extractive QA
- Low hallucination risk
- Fast inference

**Limitations:**
- Cannot generate text
- Fixed 512-token context
- Weak at translation/summarization
            """)
        with col2:
            st.markdown("### 🔶 Decoder-Only")
            st.markdown("**Models:** GPT-2, GPT-3, GPT-Neo")
            st.warning("""
**How it works:**
- **Causal (unidirectional)** attention
- Trained with **CLM** — predict next token
- GPT-3 introduced **in-context learning (ICL)**

**Strengths:**
- Best open-ended text generation
- Few-shot / zero-shot via prompting
- Scales to emergent capabilities

**Limitations:**
- Weaker NLU than bidirectional
- High hallucination risk
- GPT-3 (175B) very expensive
            """)
        with col3:
            st.markdown("### 🟣 Encoder-Decoder")
            st.markdown("**Models:** T5, BART, FLAN-T5")
            st.success("""
**How it works:**
- Encoder: bidirectional attention on input
- Decoder: autoregressive + **cross-attention**
- T5: text-to-text; BART: denoising; FLAN-T5: instruction-tuned

**Strengths:**
- Best seq2seq: translation, summarization
- FLAN-T5 strong zero-shot generalization
- Lower hallucination than pure decoders

**Limitations:**
- Higher inference cost (2 forward passes)
- More complex to train and fine-tune
            """)

    with tab2:
        st.markdown("### Technical Comparison")
        st.markdown("""
| Parameter | Encoder-Only | Decoder-Only | Encoder-Decoder |
|---|---|---|---|
| Attention Type | Bidirectional (full) | Causal (masked) | Cross-attention + causal |
| Training Objective | MLM | CLM | Seq2Seq / Span Masking |
| Context Understanding | 5/5 | 3/5 | 4/5 |
| Text Generation | 2/5 | 5/5 | 4/5 |
| Classification | 5/5 | 3/5 | 4/5 |
| Translation | 2/5 | 3/5 | 5/5 |
| Summarization | 3/5 | 4/5 | 5/5 |
| Few-shot Prompting | 2/5 | 5/5 | 4/5 |
        """)

    with tab3:
        st.markdown("### Quantitative Comparison")
        st.markdown("""
| Metric | BERT-Base | GPT-2 (M) | GPT-3 (175B) | T5-Base | BART-Large | FLAN-T5-Base |
|---|---|---|---|---|---|---|
| Parameters | 110M | 345M | 175B | 250M | 400M | 250M |
| Context Length | 512 | 1,024 | 4,096 | 512 | 1,024 | 512 |
| SQuAD F1 | 88.5 | ~65 | ~85 | ~81 | ~77 | ~86 |
| ROUGE-L (CNN/DM) | ~40 | ~35 | ~39 | ~41.5 | ~44.2 | ~43.1 |
| Perplexity (LM) | N/A | ~35 | ~20.1 | N/A | N/A | N/A |
| Min GPU VRAM | 4GB | 8GB | API only | 8GB | 12GB | 8GB |
| Hallucination Risk | Very Low | Moderate | High | Low | Low | Low |
| Open Source | ✔ | ✔ | ✘ | ✔ | ✔ | ✔ |
        """)

    with tab4:
        st.markdown("### Strengths & Weaknesses")
        st.markdown("""
| Model | Strengths | Weaknesses |
|---|---|---|
| BERT | Deep NLU, fast inference, low hallucination | No generation, short context |
| GPT-2/Neo | Fluent generation, few-shot learning | Hallucination, weak NLU |
| GPT-3 | Best few-shot, emergent reasoning | Proprietary, expensive, hallucination |
| T5 | Versatile seq2seq, open-source | Slower inference than encoders |
| BART | Best summarization (ROUGE-L 44.2) | Heavier than T5-Base |
| FLAN-T5 | Strong zero-shot, instruction-following | Lags GPT-3 on open-ended generation |
        """)

    with tab5:
        st.markdown("### Application-Based Recommendations")
        st.markdown("""
| Task | Best Model | Reason |
|---|---|---|
| Text Classification | BERT / RoBERTa | Bidirectional attention + [CLS] fine-tuning |
| Named Entity Recognition | BERT | Rich per-token representations |
| Machine Translation | T5 / FLAN-T5 | Cross-attention maps source → target optimally |
| Abstractive Summarization | BART / FLAN-T5 | Denoising pretraining; highest ROUGE-L |
| Open-ended Generation | GPT-2 / GPT-Neo | CLM objective trains generation directly |
| Extractive QA | BERT | Span extraction is natural for encoders |
| Conversational AI | GPT-3 / GPT-Neo | Autoregressive + in-context conversation history |
| Zero/Few-Shot Tasks | FLAN-T5 / GPT-3 | Instruction tuning vs. scale-driven ICL |
| Resource-Constrained | DistilBERT / T5-Base | <300M params, deployable on 4–8GB VRAM |
        """)
        st.markdown("---")
        st.markdown("### Conclusion")
        st.markdown("""
| Use Case | Recommended Model | Why |
|---|---|---|
| NLU / Understanding | BERT / RoBERTa | Bidirectional attention = best representations |
| Text Generation | GPT-2 / GPT-Neo | Open-source autoregressive generation |
| Translation / Summarization | FLAN-T5 / BART | Seq2seq architecture + instruction tuning |
| Zero-shot Versatility | FLAN-T5 | Trained on 1,800+ tasks; open-source |
| **Best Overall** | **FLAN-T5** | Strong across all task types, open-source, efficient |
        """)
        st.markdown("### Future Trends")
        trends = [
            ("⚡ Efficient Transformers", "FlashAttention, Sparse Attention push context to 128K+ tokens"),
            ("📏 Long-context Models", "GPT-4 (128K), Gemini 1.5 (1M tokens) via RoPE, ALiBi encodings"),
            ("🖼️ Multimodal Transformers", "CLIP, GPT-4o, Gemini unify vision + language + audio"),
            ("🧩 Mixture-of-Experts", "Mixtral 8×7B activates only 12.9B of 46.7B params — same performance, fraction of cost"),
            ("🤖 Agentic AI", "LLMs as reasoning engines in tool-using pipelines (LangGraph, AutoGen)"),
        ]
        for title, desc in trends:
            st.markdown(f"**{title}** — {desc}")


# ══════════════════════════════════════════════════════════════
# PAGE 2 — GENERATOR
# ══════════════════════════════════════════════════════════════
elif page == "✍️ Generator":
    st.title("✍️ Story & Poem Generator")

    MODELS = {
        "GPT-2 Small (117M)":   "gpt2",
        "GPT-2 Medium (345M)":  "gpt2-medium",
        "GPT-Neo 125M":         "EleutherAI/gpt-neo-125m",
        "T5-Small":             "t5-small",
        "BART-Base":            "facebook/bart-base",
        "FLAN-T5-Small":        "google/flan-t5-small",
    }

    SEQ2SEQ = {"T5-Small", "BART-Base", "FLAN-T5-Small"}

    @st.cache_resource
    def load_pipe(model_id, is_s2s):
        task = "text2text-generation" if is_s2s else "text-generation"
        return pipeline(task, model=model_id,
                        device=-1 if torch.cuda.is_available() else -1)

    def run_generate(pipe, is_s2s, prompt, max_length, temperature, top_k, top_p, num_seq):
        if is_s2s:
            results = pipe(prompt, max_new_tokens=max_length,
                           num_return_sequences=num_seq,
                           do_sample=True, temperature=temperature)
            return [r["generated_text"] for r in results]
        else:
            results = pipe(prompt, max_length=max_length,
                           temperature=temperature, top_k=top_k, top_p=top_p,
                           num_return_sequences=num_seq, do_sample=True)
            return [r["generated_text"] for r in results]

    with st.sidebar:
        st.markdown("---")
        st.markdown("**⚙️ Parameters**")
        max_length  = st.slider("Max Length",    100, 600, 300)
        temperature = st.slider("Temperature",   0.1, 2.0, 0.9, 0.05)
        top_k       = st.slider("Top-k",         0,   100, 50)
        top_p       = st.slider("Top-p",         0.1, 1.0, 0.95, 0.05)
        num_seq     = st.slider("Num Sequences", 1,   3,   1)

    gen_tab1, gen_tab2 = st.tabs(["🖊️ Single Model", "🔀 Compare Models"])

    with gen_tab1:
        selected = st.selectbox("Choose model:", list(MODELS.keys()))
        is_s2s   = selected in SEQ2SEQ

        if is_s2s:
            st.info("💡 Seq2seq model — phrase your prompt as an instruction, e.g: *'Write a short poem about: the lighthouse at the edge of the world'*")

        prompt = st.text_area("Enter prompt:", height=100, key="single_p",
                              placeholder="The old lighthouse stood alone at the edge of the world,")

        if st.button("🚀 Generate", key="btn_single"):
            if not prompt.strip():
                st.warning("Enter a prompt first.")
            else:
                with st.spinner(f"Generating with {selected}..."):
                    pipe    = load_pipe(MODELS[selected], is_s2s)
                    outputs = run_generate(pipe, is_s2s, prompt,
                                           max_length, temperature, top_k, top_p, num_seq)
                for i, text in enumerate(outputs):
                    st.subheader(f"Output {i+1}" if num_seq > 1 else "Generated Output")
                    st.write(text)
                    st.download_button("⬇ Download", text,
                                       file_name=f"output_{i+1}.txt", key=f"dl_s_{i}")
                    st.divider()

                with st.expander("📊 Evaluation Metrics"):
                    tok = GPT2Tokenizer.from_pretrained("gpt2")
                    mdl = GPT2LMHeadModel.from_pretrained("gpt2")
                    mdl.eval()
                    tok.pad_token = tok.eos_token
                    for i, text in enumerate(outputs):
                        ref   = nltk.word_tokenize(prompt.lower())
                        hyp   = nltk.word_tokenize(text.lower())
                        bleu  = sentence_bleu([ref], hyp,
                                              smoothing_function=SmoothingFunction().method4)
                        scorer = rs.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
                        rouge  = scorer.score(prompt, text)
                        enc    = tok(text, return_tensors="pt", truncation=True, max_length=512)
                        with torch.no_grad():
                            loss = mdl(enc.input_ids, labels=enc.input_ids).loss
                        ppl = round(math.exp(loss.item()), 2)
                        st.markdown(f"**Output {i+1}**")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("BLEU",      round(bleu, 4))
                        m2.metric("ROUGE-1",   round(rouge['rouge1'].fmeasure, 4))
                        m3.metric("ROUGE-L",   round(rouge['rougeL'].fmeasure, 4))
                        m4.metric("Perplexity", ppl)

    with gen_tab2:
        st.markdown("Same prompt — every selected model generates side by side.")
        selected_compare = st.multiselect(
            "Models to compare:",
            list(MODELS.keys()),
            default=["GPT-2 Small (117M)", "GPT-Neo 125M", "FLAN-T5-Small"]
        )
        compare_prompt = st.text_area("Enter prompt:", height=100, key="compare_p",
                                      placeholder="In a city where no one remembered their dreams,")

        if st.button("🔀 Compare", key="btn_compare"):
            if not compare_prompt.strip():
                st.warning("Enter a prompt first.")
            elif len(selected_compare) < 2:
                st.warning("Select at least 2 models.")
            else:
                cols = st.columns(len(selected_compare))
                for col, name in zip(cols, selected_compare):
                    with col:
                        st.markdown(f"**{name}**")
                        with st.spinner(f"{name}..."):
                            s2s  = name in SEQ2SEQ
                            pipe = load_pipe(MODELS[name], s2s)
                            out  = run_generate(pipe, s2s, compare_prompt,
                                                max_length, temperature, top_k, top_p, 1)
                        st.write(out[0])
                        st.download_button("⬇ Download", out[0],
                                           file_name=f"{name}.txt",
                                           key=f"dl_c_{name}")


# ══════════════════════════════════════════════════════════════
# PAGE 3 — ARCHITECTURES
# ══════════════════════════════════════════════════════════════
elif page == "🔬 Architectures":
    st.title("🔬 Architectures & Mathematical Intuition")

    model_choice = st.selectbox("Select model:", [
        "BERT", "GPT-2 / GPT-3 / GPT-Neo", "T5", "BART", "FLAN-T5"
    ])

    # ── BERT ─────────────────────────────────────────────────
    if model_choice == "BERT":
        st.header("BERT — Bidirectional Encoder Representations from Transformers")
        st.caption("Devlin et al., Google 2018")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Architecture")
            st.markdown("""
<div style="font-family:monospace; background:#1e1e2e; padding:20px; border-radius:10px; line-height:2;">
  <div style="text-align:center; color:#cdd6f4;">Input: [CLS] tok₁ tok₂ ... tokₙ [SEP]</div>
  <div style="text-align:center; color:#6c7086;">↓</div>
  <div style="background:#313244; border-radius:8px; padding:10px; margin:5px 0; text-align:center; color:#89b4fa;">
    🔷 Embedding Layer<br><small style="color:#a6adc8;">Token + Position + Segment</small>
  </div>
  <div style="text-align:center; color:#6c7086;">↓</div>
  <div style="background:#313244; border-radius:8px; padding:10px; margin:5px 0; text-align:center; color:#89b4fa;">
    🔄 Transformer Encoder × 12
    <div style="background:#45475a; border-radius:6px; padding:8px; margin:6px 0; font-size:0.85em; color:#cdd6f4;">
      Multi-Head Self-Attention (Bidirectional)<br>
      Add &amp; LayerNorm<br>
      Feed-Forward Network<br>
      Add &amp; LayerNorm
    </div>
  </div>
  <div style="text-align:center; color:#6c7086;">↓</div>
  <div style="background:#313244; border-radius:8px; padding:10px; margin:5px 0; text-align:center; color:#a6e3a1;">
    [CLS] Representation
  </div>
  <div style="text-align:center; color:#6c7086;">↓</div>
  <div style="background:#313244; border-radius:8px; padding:10px; margin:5px 0; text-align:center; color:#f38ba8;">
    Task Head (Classifier / QA / NER)
  </div>
</div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("#### Mathematical Intuition")
            st.markdown("**Self-Attention (Scaled Dot-Product):**")
            st.latex(r"\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V")
            st.markdown("- Q, K, V are linear projections of the input\n- $\\sqrt{d_k}$ prevents vanishing gradients in softmax")
            st.markdown("**Multi-Head Attention:**")
            st.latex(r"\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1,...,\text{head}_h)W^O")
            st.latex(r"\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)")
            st.markdown("**Feed-Forward Network:**")
            st.latex(r"\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2")
            st.markdown("**MLM Training Objective:**")
            st.latex(r"\mathcal{L}_{MLM} = -\sum_{i \in \mathcal{M}} \log P(x_i \mid x_{\backslash \mathcal{M}})")
            st.markdown("- $\\mathcal{M}$ = set of masked positions\n- Predicts masked token from full bidirectional context")
            st.markdown("**Key Config (BERT-Base):** Layers: 12 · Heads: 12 · Hidden: 768 · Params: 110M")

    # ── GPT ──────────────────────────────────────────────────
    elif model_choice == "GPT-2 / GPT-3 / GPT-Neo":
        st.header("GPT-2 / GPT-3 / GPT-Neo — Decoder-Only Autoregressive LMs")
        st.caption("Radford et al. 2019 · Brown et al. 2020 · EleutherAI 2021")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Architecture")
            st.markdown("""
<div style="font-family:monospace; background:#1e1e2e; padding:20px; border-radius:10px; line-height:2;">
  <div style="text-align:center; color:#cdd6f4;">Input tokens (left to right only →)</div>
  <div style="text-align:center; color:#6c7086;">↓</div>
  <div style="background:#313244; border-radius:8px; padding:10px; margin:5px 0; text-align:center; color:#89b4fa;">
    🔶 Token + Positional Embedding
  </div>
  <div style="text-align:center; color:#6c7086;">↓</div>
  <div style="background:#313244; border-radius:8px; padding:10px; margin:5px 0; text-align:center; color:#89b4fa;">
    🔄 Transformer Decoder × L
    <div style="background:#45475a; border-radius:6px; padding:8px; margin:6px 0; font-size:0.85em; color:#cdd6f4;">
      Masked Self-Attention (Causal ◀)<br>
      <span style="color:#f9e2af;">No future tokens visible</span><br>
      Add &amp; LayerNorm<br>
      Feed-Forward Network<br>
      Add &amp; LayerNorm
    </div>
    <small style="color:#a6adc8;">GPT-2: L=12 · GPT-3: L=96 · GPT-Neo: L=24</small>
  </div>
  <div style="text-align:center; color:#6c7086;">↓</div>
  <div style="background:#313244; border-radius:8px; padding:10px; margin:5px 0; text-align:center; color:#a6e3a1;">
    Linear + Softmax
  </div>
  <div style="text-align:center; color:#6c7086;">↓</div>
  <div style="background:#313244; border-radius:8px; padding:10px; margin:5px 0; text-align:center; color:#f38ba8;">
    P(next token) → autoregressive generation
  </div>
</div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("#### Mathematical Intuition")
            st.markdown("**Causal (Masked) Self-Attention:**")
            st.latex(r"\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V")
            st.markdown("- $M_{ij} = -\\infty$ if $j > i$, else $0$ — blocks future tokens")
            st.markdown("**CLM Training Objective:**")
            st.latex(r"\mathcal{L}_{CLM} = -\sum_{t=1}^{T} \log P(x_t \mid x_1, x_2, ..., x_{t-1})")
            st.markdown("**In-Context Learning (GPT-3):**")
            st.latex(r"P(y \mid \text{prompt}) = \prod_{t} P(y_t \mid \text{prompt}, y_{<t})")
            st.markdown("- No gradient update — examples in prompt shift the conditional distribution")
            st.markdown("**Scaling Law (Kaplan et al.):**")
            st.latex(r"\mathcal{L}(N) \propto N^{-\alpha}, \quad \alpha \approx 0.076")
            st.markdown("- Loss decreases as a power law with parameter count N")
            st.markdown("**Key Configs:**\n- GPT-2: 12L · 12H · 768d · 117M\n- GPT-3: 96L · 96H · 12288d · 175B\n- GPT-Neo: 24L · 16H · 2048d · 1.3B")

    # ── T5 ───────────────────────────────────────────────────
    elif model_choice == "T5":
        st.header("T5 — Text-to-Text Transfer Transformer")
        st.caption("Raffel et al., Google 2020")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Architecture")
            st.markdown("""
<div style="font-family:monospace; background:#1e1e2e; padding:20px; border-radius:10px; line-height:2;">
  <div style="text-align:center; color:#cdd6f4;">"translate English to French: Hello"</div>
  <div style="text-align:center; color:#6c7086;">↓</div>
  <div style="display:flex; gap:10px;">
    <div style="flex:1; background:#1e3a5f; border-radius:8px; padding:10px; text-align:center;">
      <div style="color:#89b4fa; font-weight:bold;">🔷 ENCODER × 12</div>
      <div style="background:#2a4a6f; border-radius:6px; padding:8px; margin:6px 0; font-size:0.85em; color:#cdd6f4;">
        Bidirectional Self-Attn<br>Add &amp; LayerNorm<br>FFN (ReLU)<br>Add &amp; LayerNorm
      </div>
    </div>
    <div style="flex:1; background:#3a1e3a; border-radius:8px; padding:10px; text-align:center;">
      <div style="color:#cba6f7; font-weight:bold;">🔶 DECODER × 12</div>
      <div style="background:#4a2a4a; border-radius:6px; padding:8px; margin:6px 0; font-size:0.85em; color:#cdd6f4;">
        Masked Self-Attn<br>
        <span style="color:#f9e2af;">Cross-Attention ←──</span><br>
        FFN<br>Add &amp; LayerNorm
      </div>
    </div>
  </div>
  <div style="text-align:center; color:#6c7086; margin-top:8px;">↓</div>
  <div style="background:#313244; border-radius:8px; padding:10px; margin:5px 0; text-align:center; color:#a6e3a1;">
    Output text (token by token)
  </div>
  <div style="margin-top:10px; color:#a6adc8; font-size:0.85em; background:#313244; padding:8px; border-radius:6px;">
    ✦ Sentinel tokens: &lt;extra_id_0&gt; replace masked spans<br>
    ✦ All tasks unified as text-to-text format
  </div>
</div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("#### Mathematical Intuition")
            st.markdown("**Cross-Attention (key decoder operation):**")
            st.latex(r"\text{CrossAttn}(Q_{dec}, K_{enc}, V_{enc}) = \text{softmax}\left(\frac{Q_{dec}K_{enc}^T}{\sqrt{d_k}}\right)V_{enc}")
            st.markdown("- Q from decoder states · K, V from encoder output")
            st.markdown("**Span Corruption Objective:**")
            st.latex(r"\mathcal{L}_{T5} = -\sum_{s \in \mathcal{S}} \log P(\text{span}_s \mid \tilde{x})")
            st.markdown("- Random spans replaced with sentinel tokens\n- Model learns to reconstruct original spans")
            st.markdown("**Relative Position Bias:**")
            st.latex(r"a_{ij} = \frac{(q_i + b_{ij})k_j^T}{\sqrt{d_k}}")
            st.markdown("- $b_{ij}$ = learned scalar per relative distance — more flexible than sinusoidal PE")
            st.markdown("**Key Config (T5-Base):** 12E+12D layers · 12 heads · 768d · 250M params")

    # ── BART ─────────────────────────────────────────────────
    elif model_choice == "BART":
        st.header("BART — Denoising Sequence-to-Sequence Model")
        st.caption("Lewis et al., Facebook AI 2019")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Architecture")
            st.markdown("""
<div style="font-family:monospace; background:#1e1e2e; padding:20px; border-radius:10px; line-height:2;">
  <div style="background:#3a2a1e; border-radius:8px; padding:10px; margin:5px 0; text-align:center; color:#fab387;">
    Corrupted Input
    <div style="font-size:0.8em; color:#a6adc8; margin-top:4px;">
      token masking · deletion · infilling · sentence permutation
    </div>
  </div>
  <div style="text-align:center; color:#6c7086;">↓</div>
  <div style="display:flex; gap:10px;">
    <div style="flex:1; background:#1e3a5f; border-radius:8px; padding:10px; text-align:center;">
      <div style="color:#89b4fa; font-weight:bold;">ENCODER × 6</div>
      <div style="color:#a6adc8; font-size:0.85em; margin-top:4px;">BERT-style<br>Bidirectional</div>
    </div>
    <div style="flex:1; background:#3a1e3a; border-radius:8px; padding:10px; text-align:center;">
      <div style="color:#cba6f7; font-weight:bold;">DECODER × 6</div>
      <div style="color:#a6adc8; font-size:0.85em; margin-top:4px;">GPT-style<br>Autoregressive<br>+ Cross-Attention</div>
    </div>
  </div>
  <div style="text-align:center; color:#6c7086;">↓</div>
  <div style="background:#313244; border-radius:8px; padding:10px; margin:5px 0; text-align:center; color:#a6e3a1;">
    Reconstructed original text
  </div>
  <div style="margin-top:10px; color:#a6adc8; font-size:0.85em; background:#313244; padding:8px; border-radius:6px;">
    💡 Encoder = robust understanding of noisy/long input<br>
    💡 Decoder = fluent generation of clean output<br>
    💡 Directly matches the summarization task setting
  </div>
</div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("#### Mathematical Intuition")
            st.markdown("**Denoising Objective:**")
            st.latex(r"\mathcal{L}_{BART} = -\sum_{t} \log P(x_t \mid \tilde{x}, x_{<t};\theta)")
            st.markdown("- $\\tilde{x}$ = corrupted input · $x$ = original document\n- Model learns to reconstruct x from noisy version")
            st.markdown("**Cross-Attention:**")
            st.latex(r"C = \text{softmax}\left(\frac{Q_{dec}K_{enc}^T}{\sqrt{d_k}}\right)V_{enc}")
            st.markdown("**Why denoising helps generation:**\n- Encoder learns robust representations from noise\n- Decoder learns to complete/fluently generate from those\n- Directly mirrors summarization: long article → clean summary")
            st.markdown("**Key Config (BART-Large):** 12E+12D layers · 16 heads · 1024d · 400M params")

    # ── FLAN-T5 ──────────────────────────────────────────────
    elif model_choice == "FLAN-T5":
        st.header("FLAN-T5 — Instruction-Tuned T5")
        st.caption("Chung et al., Google 2022")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Architecture")
            st.markdown("""
<div style="font-family:monospace; background:#1e1e2e; padding:20px; border-radius:10px; line-height:2;">
  <div style="background:#313244; border-radius:8px; padding:10px; margin:5px 0; text-align:center; color:#89b4fa;">
    T5 Encoder-Decoder (identical architecture)
  </div>
  <div style="text-align:center; color:#6c7086;">↓ fine-tuned on ↓</div>
  <div style="background:#1e3a2a; border-radius:8px; padding:12px; margin:5px 0;">
    <div style="color:#a6e3a1; font-weight:bold; text-align:center;">FLAN Instruction Collection</div>
    <div style="display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; justify-content:center;">
      <span style="background:#2a4a3a; padding:4px 8px; border-radius:4px; color:#cdd6f4; font-size:0.8em;">"Translate to French: Hello"</span>
      <span style="background:#2a4a3a; padding:4px 8px; border-radius:4px; color:#cdd6f4; font-size:0.8em;">"Summarize: &lt;article&gt;"</span>
      <span style="background:#2a4a3a; padding:4px 8px; border-radius:4px; color:#cdd6f4; font-size:0.8em;">"Classify sentiment: &lt;text&gt;"</span>
      <span style="background:#2a4a3a; padding:4px 8px; border-radius:4px; color:#cdd6f4; font-size:0.8em;">"Answer: &lt;question&gt;"</span>
      <span style="background:#2a4a3a; padding:4px 8px; border-radius:4px; color:#cdd6f4; font-size:0.8em;">+ 1,800 more tasks · 473 datasets</span>
    </div>
  </div>
  <div style="text-align:center; color:#6c7086;">↓</div>
  <div style="background:#313244; border-radius:8px; padding:10px; margin:5px 0; text-align:center; color:#f38ba8;">
    Strong zero-shot &amp; few-shot on unseen tasks
  </div>
  <div style="margin-top:8px; color:#a6adc8; font-size:0.85em; background:#313244; padding:8px; border-radius:6px;">
    💡 Seeing enough task <i>formats</i> lets the model generalize<br>
    to new tasks from instruction alone — no fine-tuning needed
  </div>
</div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("#### Mathematical Intuition")
            st.markdown("**Base architecture identical to T5** — same cross-attention, span corruption pretraining, relative position bias.")
            st.markdown("**Instruction Tuning Objective:**")
            st.latex(r"\mathcal{L}_{FLAN} = -\sum_{(x,y) \in \mathcal{D}_{inst}} \log P_\theta(y \mid \text{inst}(x))")
            st.markdown("- $\\mathcal{D}_{inst}$ = instruction-formatted dataset across 1,800+ tasks\n- $\\text{inst}(x)$ = natural language task description prepended to input")
            st.markdown("**Zero-shot generalization:**")
            st.latex(r"P(y_{new} \mid x_{new}) \approx P_\theta(y \mid \text{inst}(x_{new}))")
            st.markdown("- No gradient update at inference — just a well-crafted instruction prompt")
            st.markdown("**FLAN-T5 vs T5 at same size:**")
            st.markdown("""
| Benchmark | T5-XL (3B) | FLAN-T5-Base (250M) |
|---|---|---|
| MMLU | 46.7 | 49.4 |
| Zero-shot avg | 52.1 | 55.3 |

FLAN-T5-Base beats T5-XL (12× larger) on zero-shot benchmarks.
            """)
            st.markdown("**Key Config:** Same as T5-Base · 250M params · fine-tuned on FLAN collection")