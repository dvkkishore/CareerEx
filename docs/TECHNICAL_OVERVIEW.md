# CareerEx: Technical overview

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

## Models and data processing

**Bag-of-words classifier:** Multinomial Naive Bayes and CountVectorizer offer a simple, explainable baseline for textual questionnaire responses. They have known limitations: term frequency can obscure meaning, labels may reflect dataset biases and the classifier is not a substitute for a counselor.

**Content similarity for suggestions:** vectorized skills/description matching is straightforward to inspect but cannot establish that a user meets requirements for a given job, course or university.

**Separation of experimental and interface code:** `career_field.py` describes training; `app.py` performs inference and renders results. The split makes the two roles easy to discuss but does not guarantee that the serialized models are current.

## Verification status

| Observable artifact | What it supports |
| --- | --- |
| `career_field.py` + `career_fields.csv` | Training workflow and labeled data |
| `app.py` | Questionnaire / filtering / visualization routes |
| `FinalJob.csv`, `uni.csv`, `job_data.csv` | Examples of inputs used by the prototype |
| Jinja templates | UI flow across prediction, jobs, universities |
| Offline test suite | Limited source/data contract checks |

The existing documentation makes performance claims without a recorded, independently reproduced benchmark. Therefore these technical notes report **no model accuracy, placement outcome, or user adoption figures**.

## Engineering and safety considerations

The current app still compares **plaintext stored passwords**. Database credentials were previously committed publicly and must be rotated externally. A secure modern redesign would use password hashing with explicit migration/reset procedures, CSRF controls and deployment tests, rather than claiming this old Flask project is production-ready.

## Team and source provenance

**Academic collaborators credited:** Dannana Venkata Kishore, Kuncha Supriya and Moldireddy Gari Swetha. Specific individual modules are not attributed in the source, so this report does not invent personal ownership. For maintenance and attribution, confirm roles and design decisions with project records or collaborators.

**Primary sources:** [README](../README.md) · [Flask app](../app.py) · [classifier](../career_field.py) · [security review](../AUTH_REVIEW.md).


## Repository map

| File or directory | Responsibility |
| --- | --- |
| `app.py` | Flask routes, form handling, model and dataset loading |
| `career_field.py` | Naive Bayes and CountVectorizer model training |
| `templates/` and `static/` | Jinja UI, CSS, and generated chart assets |
| `career_fields.csv`, `FinalJob.csv`, `uni.csv`, `job_data.csv` | Tracked input datasets |
| `career_field_prediction_model.pkl`, `vocabulary.pkl` | Legacy model artifacts loaded by the app |
| `tests/` | Offline source/dataset checks |
| `SECURITY.md`, `AUTH_REVIEW.md` | Known authentication and credential exposure risks |

## Reproducing behavior

See the [README](../README.md) for the historical dependency install and environment variables. Model artifacts are deserialized from pickle/joblib; load only sources you trust. The existing offline tests do not connect to MySQL. Running `python app.py` also depends on a separately managed MySQL instance and its schema.

The original training script prints training and holdout accuracy, but no independently reproduced score is asserted in this document. Do not infer job-placement outcomes or validate career guidance solely from a classifier label.

## Known deployment blockers

Stored plaintext passwords are still compared during authentication, and database credentials were previously exposed in public Git history. They require a deliberate credential-rotation and account password migration plan outside these documentation changes. This project is not approved for public deployment.
