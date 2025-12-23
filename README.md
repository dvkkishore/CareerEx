
# CareerEx   
**Career Exploration and Recommendation Platform**

CareerEx is a full-stack web application designed to help students and professionals make informed career decisions. It leverages **Machine Learning**, **Natural Language Processing**, and **content-based recommendation systems** to predict suitable career fields and recommend relevant universities, courses, and job opportunities based on user input.

---

## Project Overview

Choosing the right career path is a complex process involving interests, skills, goals, and market trends. CareerEx simplifies this process by:

- Predicting the most suitable **career field** using a trained ML model  
- Recommending **universities** based on course, city, and state  
- Filtering and recommending **job roles** based on skills and keywords  
- Visualizing **employment statistics** for better decision-making  

This project is built using **Flask** and follows a modular, scalable architecture.

---

## Key Features

- User Authentication (Login / Logout)
- Career Field Prediction using **Multinomial Naive Bayes**
- University Recommendation using **Content-Based Filtering**
- Job Recommendation using **NLP + Cosine Similarity**
- Employment Data Visualization
- Clean modular project structure
- Tested with multiple datasets

---

## System Architecture

CareerEx follows a **3-layer architecture**:

1. **Presentation Layer**  
   - HTML, CSS, Jinja Templates  
   - User interaction and visualization  

2. **Application Layer**  
   - Flask backend  
   - ML models and recommendation logic  

3. **Data Layer**  
   - CSV-based datasets  
   - Serialized ML models (`.pkl`)  

---

## Technologies Used

### Backend
- Python 3
- Flask
- Jinja2

### Machine Learning & NLP
- scikit-learn
- Multinomial Naive Bayes
- CountVectorizer
- Cosine Similarity
- pandas, numpy

### Frontend
- HTML5
- CSS3
- Bootstrap

### Tools
- Git & GitHub
- Virtual Environment (`venv`)
- Jupyter Notebook

---

## Project Structure

```
CareerEx/
│
├── app.py                         # Main Flask application
├── career_field.py                # Career prediction logic
├── career_field_prediction_model.pkl
├── vocabulary.pkl
│
├── templates/                     # HTML templates
├── static/                        # CSS & assets
│
├── data/
│   ├── career_fields.csv
│   ├── universities.csv
│   ├── jobs.csv
│   ├── FinalJob.csv
│   └── uni.csv
│
├── universities.ipynb             # Data analysis notebook
├── requirements.txt               # Python dependencies
├── README.md
└── .gitignore
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/CareerEx.git
cd CareerEx
```

### 2. Create Virtual Environment
```bash
python -m venv env
source env/bin/activate   # Linux/Mac
env\Scripts\activate    # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

### 5. Access the App
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## Machine Learning Details

- **Algorithm**: Multinomial Naive Bayes  
- **Input**: User responses to career questionnaire  
- **Output**: Predicted Career Field  
- **Text Processing**: CountVectorizer  
- **Recommendation Logic**: Content-based filtering using cosine similarity  

---

## Use Cases

- Students choosing a career path
- Graduates selecting higher education options
- Job seekers exploring suitable roles
- Institutions analyzing career trends

---

## Author

**Dannana Venkata Kishore, Kuncha Supriya, Moldireddy Gari Swetha**  
Vel Tech Rangarajan Dr.Sagunthala R&D Institute of Science and Technology  

---

## License

This project is for academic and learning purposes.
