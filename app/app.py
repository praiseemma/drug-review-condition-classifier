# Web app and deployment by Arshi Nisar
"""Streamlit front end for the Drug Review Condition Classifier.

Privacy: visitor text is only passed to predict(). It is never printed, logged or saved.
No external fonts, scripts or images are loaded (system fonts only).
All numbers shown here come from the project README (Praise Emma). If the project is
updated, edit METRICS and WEAK_CLASSES_TEST_F1 below.
"""
import html

import streamlit as st

from inference import MAX_CHARS, MIN_WORDS, AppError, load_artifacts, predict

st.set_page_config(
    page_title="Drug Review Condition Classifier",
    page_icon="🩺",
    layout="centered",
)

# ---------- Numbers from the README (single place to edit) ----------
METRICS = {
    "n_classes": 28,
    "val_f1_both": 0.854,   # text + drug, validation
    "test_f1_both": 0.855,  # text + drug, final test
    "val_f1_text": 0.737,   # text only, validation
    "val_f1_drug": 0.741,   # drug only, validation (not used in this app)
    "n_train": 59634,
    "n_val": 14907,
    "n_test": 10236,
}
# Weakest classes on the final test set (README)
WEAK_CLASSES_TEST_F1 = {
    "Generalized Anxiety Disorder": 0.449,
    "Panic Disorder": 0.455,
    "Anxiety and Stress": 0.526,
    "Abnormal Uterine Bleeding": 0.555,
}
LOW_CONFIDENCE = 0.40   # top score below this -> "model is unsure"
CLOSE_MARGIN = 0.10     # top-1 minus top-2 below this -> "close call"
NOT_SPECIFIED = "Not specified"

# ---------- Styling: all colours are defined once, in :root ----------
st.markdown(
    """<style>
:root{--bg:#EFEADF;--card:#F7F3EA;--ink:#14120F;--muted:#5E564A;--accent:#C2410C;--accent-dark:#9A3412;--grey:#7A7062;--line:#C9C1B0;
--sans:"Helvetica Neue",Arial,"Segoe UI",system-ui,sans-serif;--serif:Georgia,"Times New Roman",serif;--mono:ui-monospace,"SFMono-Regular",Consolas,"Courier New",monospace;}
.eyebrow{font-family:var(--sans);font-size:.75rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--accent-dark);display:flex;align-items:center;gap:.7rem;margin-top:.5rem}
.eyebrow .rule{flex:0 0 2.5rem;height:1px;background:var(--grey)}
.eyebrow .dim{color:var(--muted);font-family:var(--mono);font-weight:400}
.h1{font-family:var(--sans);font-weight:800;letter-spacing:-.03em;line-height:1.1;font-size:clamp(1.6rem,6vw,2.4rem);color:var(--ink);margin:.5rem 0 .6rem 0}
.h1 .acc{display:block;color:var(--accent);font-size:.72em;font-weight:700;letter-spacing:-.02em;margin-top:.15rem}
.lede{font-family:var(--serif);font-style:italic;color:var(--muted);font-size:clamp(.95rem,3vw,1.05rem);line-height:1.5;margin:0 0 .4rem 0}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:.7rem;margin:1.1rem 0 1.4rem 0}
.stat{background:var(--card);border:1px solid var(--line);border-top:3px solid var(--accent);border-radius:6px;padding:.85rem .95rem;box-shadow:0 1px 3px rgba(20,18,15,0.10)}
.sv{font-family:var(--mono);font-size:clamp(1.3rem,4.5vw,1.9rem);color:var(--ink);line-height:1.1}
.sl{font-family:var(--sans);font-size:.72rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--ink);margin-top:.35rem}
.ss{font-family:var(--mono);font-size:.72rem;color:var(--muted);margin-top:.2rem}
.sec{display:flex;align-items:baseline;gap:.7rem;border-top:1px solid var(--ink);padding-top:.7rem;margin:1.6rem 0 .6rem 0}
.sec .n{font-family:var(--mono);color:var(--accent-dark);font-size:.85rem}
.sec .t{font-family:var(--sans);font-size:.78rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.disc{background:var(--card);border:1px solid var(--ink);border-left:8px solid var(--accent);border-radius:2px;padding:.8rem 1rem;margin:.4rem 0 1rem 0;color:var(--ink);font-size:.97rem;line-height:1.5}
.note{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--grey);border-radius:2px;padding:.65rem 1rem;margin:.5rem 0;color:var(--ink);font-size:.93rem;line-height:1.5}
.note.warn{border-left-color:var(--accent)}
.meta{font-family:var(--mono);font-size:.78rem;color:var(--muted);margin:.3rem 0 .7rem 0}
.res{background:var(--card);border:1px solid var(--line);border-radius:2px;padding:.75rem 1rem;margin-bottom:.55rem}
.res.top{border:2px solid var(--accent)}
.res-head{display:flex;justify-content:space-between;gap:.6rem;font-family:var(--sans);font-weight:700;color:var(--ink);margin-bottom:.45rem}
.res-head .pct{font-family:var(--mono);font-weight:400}
.bar{background:#E0D9C9;border-radius:2px;height:10px;overflow:hidden}
.fill{background:var(--grey);height:100%}
.res.top .fill{background:var(--accent)}
.foot{font-family:var(--mono);font-size:.75rem;color:var(--muted);line-height:1.6;border-top:1px solid var(--ink);padding-top:.8rem;margin-top:1.6rem}
.foot a{color:var(--accent-dark)}
.stButton>button{border-radius:2px;font-family:var(--sans);font-weight:600}
@media (max-width:640px){.stats{grid-template-columns:repeat(2,1fr)}.stat{border-bottom:1px solid var(--line)}.eyebrow .rule{display:none}}
</style>""",
    unsafe_allow_html=True,
)


# ---------- Small HTML helpers (dynamic text is always escaped) ----------
def section(num, title):
    st.markdown(
        f'<div class="sec"><span class="n">{num}</span><span class="t">{html.escape(title)}</span></div>',
        unsafe_allow_html=True,
    )


def note(text_html, kind="info"):
    """text_html must already be safe HTML (escape any dynamic part first)."""
    st.markdown(f'<div class="note {kind}">{text_html}</div>', unsafe_allow_html=True)


def result_card(rank, name, prob):
    pct = prob * 100
    label = "&lt;1%" if pct < 1 else f"{pct:.0f}%"
    css = "res top" if rank == 0 else "res"
    return (
        f'<div class="{css}"><div class="res-head"><span>{html.escape(name)}</span>'
        f'<span class="pct">{label}</span></div>'
        f'<div class="bar"><div class="fill" style="width:{max(pct, 2):.1f}%"></div></div></div>'
    )


def stat_strip(items):
    cells = "".join(
        f'<div class="stat"><div class="sv">{value}</div>'
        f'<div class="sl">{label}</div><div class="ss">{sub}</div></div>'
        for value, label, sub in items
    )
    st.markdown(f'<div class="stats">{cells}</div>', unsafe_allow_html=True)


# ---------- Header ----------
st.markdown(
    f'<div class="eyebrow">Condition classifier<span class="rule"></span>'
    f'<span class="dim">{METRICS["n_classes"]} conditions</span></div>'
    '<div class="h1">What is this review about?'
    '<span class="acc">Add the drug for a sharper answer.</span></div>'
    '<p class="lede">Paste a patient review, optionally choose the drug, '
    "and see the three most likely conditions.</p>",
    unsafe_allow_html=True,
)


# ---------- Load models (once per server, cached) ----------
@st.cache_resource(show_spinner="Loading models...")
def get_artifacts():
    return load_artifacts()


try:
    art = get_artifacts()
except AppError as err:
    st.error(str(err))
    st.stop()
except Exception as err:  # never show a raw crash page
    st.error(f"The app could not start ({type(err).__name__}). Please try again later.")
    st.stop()

stat_strip(
    [
        (f"{METRICS['test_f1_both']}", "Macro F1", f"text + drug, final test, {METRICS['n_test']:,} reviews"),
        (f"{METRICS['n_classes']}", "Conditions", "after two merges"),
        (f"{METRICS['n_train']:,}", "Training reviews", f"{METRICS['n_val']:,} validation"),
        (f"{len(art['drugs']):,}", "Drugs", "known to the model"),
    ]
)

# Example texts written for this demo (not taken from the dataset)
EXAMPLES = {
    "Acne example": (
        NOT_SPECIFIED,
        "My skin was full of breakouts for years. After about three months of "
        "using this every night my face is much clearer and the redness is gone.",
    ),
    "Overlap example": (
        "Sertraline",
        "I started this a few weeks ago. I feel calmer and the constant worry has "
        "eased, but I still have low days and trouble sleeping.",
    ),
    "Short example": (NOT_SPECIFIED, "This medicine saved my life."),
}


def fill(drug, review):
    """Button callback: set both fields (runs before the widgets are drawn)."""
    st.session_state["drug"] = drug if drug in art["drug_set"] else NOT_SPECIFIED
    st.session_state["review"] = review


def show_results(result):
    top = result["top"]
    names = [n for n, _ in top]

    section("02", "Result")
    st.markdown(
        f'<div class="meta">Model used: <b>{html.escape(result["model_used"])}</b></div>',
        unsafe_allow_html=True,
    )
    for rank, (name, prob) in enumerate(top):
        st.markdown(result_card(rank, name, prob), unsafe_allow_html=True)
    st.markdown(
        '<div class="meta">Scores are the model\'s internal probabilities. They are not '
        "calibrated, so use them as a rough ranking, not as certainty.</div>",
        unsafe_allow_html=True,
    )

    for n in result["notes"]:
        note(html.escape(n))

    if top[0][1] < LOW_CONFIDENCE:
        note("The top score is low, so the model is unsure about this review.", "warn")
    elif len(top) > 1 and top[0][1] - top[1][1] < CLOSE_MARGIN:
        note("The top two scores are close, so treat this as a close call.", "warn")

    if result["model_used"] == "text only":
        note(
            "No drug was used. The text-only model is much weaker (validation macro F1 "
            f"{METRICS['val_f1_text']} vs {METRICS['val_f1_both']} with a drug), "
            "especially for short reviews."
        )
    else:
        note(
            "The selected drug strongly influences this result: the drug name alone "
            f"reaches validation macro F1 {METRICS['val_f1_drug']}."
        )

    weak = [n for n in names if n in WEAK_CLASSES_TEST_F1]
    if weak:
        detail = ", ".join(
            f"{html.escape(n)} (test F1 {WEAK_CLASSES_TEST_F1[n]:.3f})" for n in weak
        )
        note(
            "Some of these labels overlap with related ones in patient language and were "
            f"among the weakest in the project's final test: {detail}. "
            "Read the whole ranking, not only the top answer.",
            "warn",
        )
    elif "Anxiety" in names and "Depression" in names:
        note(
            "Anxiety and Depression reviews overlap in patient language, so the model can "
            "confuse them. Read the whole ranking, not only the top answer."
        )
    if "Abnormal Uterine Bleeding" in names and "Birth Control" in names:
        note(
            "Abnormal Uterine Bleeding and Birth Control likely overlap, because hormonal "
            "contraceptives are also used for bleeding problems."
        )


# ---------- Disclaimer and privacy ----------
st.markdown(
    '<div class="disc"><b>Educational demo, not medical advice.</b> It predicts which '
    "condition a patient review is about. It does not diagnose, judge side effects or "
    "recommend treatment. <b>This app does not store or log what you type.</b> "
    "Please do not enter personal details.</div>",
    unsafe_allow_html=True,
)

# ---------- Inputs ----------
section("01", "Review")
st.write("**Try an example:**")
cols = st.columns(len(EXAMPLES) + 1)
for col, (label, (ex_drug, ex_review)) in zip(cols, EXAMPLES.items()):
    col.button(label, on_click=fill, args=(ex_drug, ex_review), width="stretch")
cols[-1].button("Clear", on_click=fill, args=(NOT_SPECIFIED, ""), width="stretch")

drug = st.selectbox(
    "Drug (optional, you can type to search)",
    [NOT_SPECIFIED] + art["drugs"],
    key="drug",
    help=(
        "Choosing a drug makes the prediction much stronger. Brand and generic names "
        "are separate entries, so pick the exact name as listed."
    ),
)
review = st.text_area(
    "Patient review",
    height=180,
    max_chars=MAX_CHARS,
    key="review",
    placeholder="Paste or type a review in English...",
)
st.markdown(
    f'<div class="meta">{len(review.split())} words · {len(review)}/{MAX_CHARS} characters · '
    f"at least {MIN_WORDS} words needed</div>",
    unsafe_allow_html=True,
)

# ---------- Prediction ----------
if st.button("Predict condition", type="primary", width="stretch"):
    chosen = None if drug == NOT_SPECIFIED else drug
    try:
        with st.spinner("Thinking..."):
            result = predict(review, chosen, art)
    except AppError as err:
        st.error(str(err))
    except Exception as err:  # never show a raw crash page; never print the review
        st.error(f"Something went wrong ({type(err).__name__}). Please try again.")
    else:
        show_results(result)

# ---------- About the model ----------
section("03", "About")
with st.expander("About this model and its limits"):
    weak_lines = "\n".join(
        f"- {name}: {score:.3f}" for name, score in WEAK_CLASSES_TEST_F1.items()
    )
    st.markdown(
        f"""
**What it is.** A logistic regression on TF-IDF word features, trained on
{METRICS['n_train']:,} cleaned Drugs.com reviews across {METRICS['n_classes']} conditions.
When a known drug is chosen, the drug name is added as a feature. It predicts the
condition a review is about, not side effects, effectiveness or treatment.

| Model | Used here when | Validation macro F1 |
|---|---|---|
| Text only | no drug, or a drug not in the training data | {METRICS['val_f1_text']} |
| Drug name only | not used in this app | {METRICS['val_f1_drug']} |
| Text + drug | a known drug is chosen | {METRICS['val_f1_both']} (final test {METRICS['test_f1_both']}) |

The final test was run once on {METRICS['n_test']:,} reviews that share no text with
training or validation. Only the final model was run on it; the other two rows are
validation only.

**Weakest classes in the final test (F1):**

{weak_lines}

**Limits**
- Several labels overlap (the anxiety family, and Abnormal Uterine Bleeding vs Birth Control).
  Part of the label may be a choice among the conditions a drug is listed for, which
  text alone cannot recover.
- The drug carries much of the signal, so a good score does not mean the model understands language.
- For very short reviews (1 to 4 words) text-only accuracy was 0.28 versus 0.81 with the drug.
  This app asks for at least {MIN_WORDS} words, as the model was trained on reviews of that length.
- One validation split and one test run. The dataset was published in 2018, and its
  patient-written reviews are older than that.
"""
    )

# ---------- Footer ----------
st.markdown(
    '<div class="foot">Data: Kallumadi &amp; Gräßer (2018), Drug Reviews (Drugs.com), '
    "UCI Machine Learning Repository, "
    '<a href="https://doi.org/10.24432/C5SK5S" target="_blank" rel="noopener noreferrer">'
    "https://doi.org/10.24432/C5SK5S</a>. Non-commercial research use.<br>"
    "Model and project by Praise Emma. Web app by "
    '<a href="https://www.linkedin.com/in/arshi-n/" target="_blank" rel="noopener noreferrer">Arshi Nisar</a>. '
    'GitHub: <a href="https://github.com/praiseemma/drug-review-condition-classifier" '
    'target="_blank" rel="noopener noreferrer">'
    "https://github.com/praiseemma/drug-review-condition-classifier</a>.</div>",
    unsafe_allow_html=True,
)