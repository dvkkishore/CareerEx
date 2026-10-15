# CareerEx: Combining classification and recommendation in a career-exploration prototype

## Problem

A career decision spans both personal interests and practical options. This academic project explores presenting career-field predictions together with related university and job information in one web experience. The technical challenge is combining data sources and algorithms with a comprehensible user flow.

## System components

```text
User questionnaire -> Flask -> vocabulary/vectorization -> MultinomialNB -> career label

Job keywords -> CountVectorizer -> cosine similarity -> FinalJob.csv rows

University filters -> courses/city/state -> description similarity -> uni.csv rows

Employment view -> job_data.csv -> Matplotlib chart
```

The application stores account data in MySQL, renders pages through Jinja templates, and loads serialized classification artifacts from the repository.

## Methods and rationale

**Bag-of-words classifier:** Multinomial Naive Bayes and CountVectorizer offer a simple, explainable baseline for textual questionnaire responses. They have known limitations: term frequency can obscure meaning, labels may reflect dataset biases and the classifier is not a substitute for a counselor.

**Content similarity for suggestions:** vectorized skills/description matching is straightforward to inspect but cannot establish that a user meets requirements for a given job, course or university.

**Separation of experimental and interface code:** `career_field.py` describes training; `app.py` performs inference and renders results. The split makes the two roles easy to discuss but does not guarantee that the serialized models are current.

## Evidence vs. claims

| Observable artifact | What it supports |
| --- | --- |
| `career_field.py` + `career_fields.csv` | Training workflow and labeled data |
| `app.py` | Questionnaire / filtering / visualization routes |
| `FinalJob.csv`, `uni.csv`, `job_data.csv` | Examples of inputs used by the prototype |
| Jinja templates | UI flow across prediction, jobs, universities |
| Offline test suite | Limited source/data contract checks |

The existing documentation makes performance claims without a recorded, independently reproduced benchmark. Therefore this case study reports **no model accuracy, placement outcome, or user adoption figures**.

## Engineering and safety considerations

The current app still compares **plaintext stored passwords**. Database credentials were previously committed publicly and must be rotated externally. A secure modern redesign would use password hashing with explicit migration/reset procedures, CSRF controls and deployment tests, rather than claiming this old Flask project is production-ready.

## Authorship and contributions

**Academic collaborators credited:** Dannana Venkata Kishore, Kuncha Supriya and Moldireddy Gari Swetha. Specific individual modules are not attributed in the source, so this report does not invent personal ownership. Confirm roles and original design decisions with the team before making an individual CV claim.

**Primary sources:** [README](../README.md) · [Flask app](../app.py) · [classifier](../career_field.py) · [security review](../AUTH_REVIEW.md).
