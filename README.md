# CareerEx

**Documentation:** [Features](#what-the-repository-implements) · [Data flow](#how-it-works) · [Running locally](#local-setup-and-security) · [Technical overview](docs/TECHNICAL_OVERVIEW.md) · [Security notes](AUTH_REVIEW.md)

**An academic career-exploration web prototype combining questionnaire-based classification with course, university and job search.**

## The problem

Career guidance often combines personal interests, skills and practical education and employment options. CareerEx brings several exploratory tools into one web interface so a user can compare possible paths. The application is a **learning project**, not a validated career adviser.

## What the repository implements

- **Career-field suggestion:** a questionnaire whose text answers are converted into token-count features and classified with a serialized Multinomial Naive Bayes model.
- **Job exploration:** keyword search against skills in `FinalJob.csv`, ranked using vectorized text and cosine similarity.
- **University exploration:** filters on courses, city or state, with description-based cosine similarity as an additional ranking heuristic.
- **Employment view:** a visualization of the tracked `job_data.csv` sample.
- **Flask pages:** templates for the questionnaire, prediction output, universities, jobs and login.

## How it works

```text
Questionnaire answers -> text cleaning -> CountVectorizer vocabulary
                                             |
                                        Naive Bayes model
                                             |
                                      suggested career field

Skill keywords -> CountVectorizer -> cosine similarity -> matching CSV jobs
Course/location input -> university filter -> similarity -> matching universities
```

The system uses local tabular datasets and serialized scikit-learn artifacts. `career_field.py` contains the training approach; `app.py` connects the saved model and recommendations to the web interface.

## ML and recommendation methods

| Component | Evidence in repository | Limitation |
| --- | --- | --- |
| Career classifier | `career_field.py`, `career_fields.csv`, model/vocabulary pickles | No independently reproduced held-out accuracy or fairness study |
| Job matching | `FinalJob.csv` and cosine similarity over the `Skills` field | Keyword similarity is not verified job-fit quality |
| University suggestions | `uni.csv` and description/attribute matching | Heuristic, not personalized admissions advice |
| Employment chart | `job_data.csv` | Small sample; not verified as authoritative labor statistics |

See the [technical overview](docs/TECHNICAL_OVERVIEW.md) for system boundaries and trade-offs.

## Local setup and security

The repository uses a historical Python dependency export and a MySQL database. A compatible environment, trusted model files and an appropriately configured local database are prerequisites.

```bash
git clone https://github.com/dvkkishore/CareerEx.git
cd CareerEx
python -m venv .venv
# Activate the environment for your shell.
pip install -r requirements.txt
# Set MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB and FLASK_SECRET_KEY.
python app.py
```

**Important:** This is **not safe for public deployment as-is**. The current app still uses legacy plaintext-password storage and checking. Previously committed database credentials must be revoked or rotated at the database provider. See [SECURITY.md](SECURITY.md) and [AUTH_REVIEW.md](AUTH_REVIEW.md) before any deployment. Serialized pickles must only be loaded from trusted sources.

For an offline source/dataset contract check:

```bash
python -m unittest discover -s tests -v
```

Passing these checks does not exercise the live MySQL application.

## Team

Academic team credited in the original README: **Dannana Venkata Kishore, Kuncha Supriya, and Moldireddy Gari Swetha**, at Vel Tech Rangarajan Dr. Sagunthala R&D Institute of Science and Technology. This repository does not itemize individual responsibilities for the implemented components; refer to project records and collaborators for detailed attribution.

## Status

Educational proof of concept. The project demonstrates an integration of classic NLP/ML techniques into a multi-page web interface; model accuracy, recommendation relevance, end-to-end installation and production security are **not independently verified**. No specific usage, impact or employment outcomes are claimed.
