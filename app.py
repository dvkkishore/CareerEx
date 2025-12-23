from flask import Flask, redirect, render_template, request, url_for, session
from flask_mysqldb import MySQL
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('career_field_prediction_model.pkl')
vocabulary = joblib.load('vocabulary.pkl')

logout_success = False

# Load jobs data from CSV file
jobs_df = pd.read_csv('jobs.csv')

# Load universities data from CSV file
universities_df = pd.read_csv('uni.csv')
universities_df.dropna(subset=['description'], inplace=True)


# Create a CountVectorizer to convert job descriptions into a matrix of token counts
vectorizer = CountVectorizer(stop_words='english')
job_descriptions_matrix = vectorizer.fit_transform(jobs_df['Description'])

# Compute cosine similarity between job descriptions
jobs_cosine_similarities = cosine_similarity(job_descriptions_matrix)

# Set similarity threshold
similarity_threshold = 0.1  # Update this value to your desired threshold

# Set a threshold for cosine similarity
threshold = 0.7

uni_vectorizer = CountVectorizer(stop_words='english')
university_descriptions_matrix = uni_vectorizer.fit_transform(universities_df['description'])

# Compute cosine similarity between university descriptions
uni_cosine_similarities = cosine_similarity(university_descriptions_matrix)

# MySQL configurations
app.config['MYSQL_HOST'] = 'bqfnfor0rb1750evuwe9-mysql.services.clever-cloud.com'  # Replace with your MySQL host
app.config['MYSQL_USER'] = 'uhohj6k8z1a2ofmr'       # Replace with your MySQL username
app.config['MYSQL_PASSWORD'] = 'gnhesf9e5zIRiqaPLTn3'  # Replace with your MySQL password
app.config['MYSQL_DB'] = 'bqfnfor0rb1750evuwe9'     # Replace with your MySQL database name
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

# Secret key for session
app.secret_key = 'your_secret_key'

logged_in = False
# Initialize MySQL
mysql = MySQL(app)

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html', logged_in = logged_in, logout_success = logout_success)

# Route for the courses page
@app.route('/courses')
def courses():
    # Add your logic here for retrieving courses data and rendering the courses template
    # You can use the render_template function to render the courses.html template
    return redirect(url_for('index'))

# Route for careers.html
@app.route('/careers', methods=['GET', 'POST'])
def careers():
    if logged_in:
        if request.method == 'POST':
        # Get keyword input from the form
            keyword = request.form['keyword']
            
            # Convert the keyword input into a matrix of token counts
            keyword_matrix = vectorizer.transform([keyword])
            
            # Compute cosine similarity between the keyword and job descriptions
            keyword_similarity = cosine_similarity(keyword_matrix, job_descriptions_matrix).flatten()
            
            # Sort the jobs by similarity in descending order
            similar_jobs_indices = keyword_similarity.argsort()[::-1]
            
            # Filter jobs by similarity threshold
            similar_jobs_indices = similar_jobs_indices[keyword_similarity[similar_jobs_indices] > similarity_threshold]
            
            # Get the top 10 most similar jobs
            similar_jobs = jobs_df.iloc[similar_jobs_indices][:9]
            
            if similar_jobs.empty:
                # Render the careers.html template without any similar jobs
                return render_template('careers.html', jobs=None)
            else:
                # Convert similar jobs dataframe to dictionary
                similar_jobs_dict = similar_jobs.to_dict(orient='records')
                
                # Render the careers.html template with the similar jobs
                return render_template('careers.html', jobs=similar_jobs_dict, logged_in = logged_in)    
        
        # Render the careers.html template without any job recommendations
        return render_template('careers.html', logged_in = logged_in)    
    else:
        return redirect(url_for('login'))

# Route for the employment page
@app.route('/employment')
def employment():
    if logged_in:
        # Read the CSV data into a Pandas DataFrame
        data = pd.read_csv('job_data.csv')

        # Extract the required columns
        job_industries = data['Job Industries']
        growth_rates = data['Employment Rate']

        # Create a bar plot of job categories vs growth rates
        plt.figure(figsize=(12, 6))
        plt.bar(job_industries, growth_rates)
        plt.xlabel('Job Industry')
        plt.ylabel('Employment Rate')
        plt.title('Employment Rate in Job Industries')
        plt.xticks(rotation=90)
        plt.tight_layout()

        # Save the plot to a static image file
        plt.savefig('static/job_growth.png')

        # Render the plot in the HTML template
        return render_template('employment.html', plot_path='static/job_growth.png', logged_in = logged_in)
    else:
        return redirect(url_for('login'))

# Sign up route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Get form data
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Form validation
        if not username or not email or not password or not confirm_password:
            error = 'Please fill out all the fields.'
            return render_template('signup.html', error=error)
        elif password != confirm_password:
            error = 'Password and Confirm Password do not match.'
            return render_template('signup.html', error=error)
        else:
            # Register new user to MySQL database
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                        (username, email, password))
            mysql.connection.commit()
            cur.close()
            return redirect(url_for('login'))

    return render_template('signup.html')


# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        global logged_in, logout_success
        # Get form data
        username = request.form['username']
        password = request.form['password']

        # Authenticate user in MySQL database
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        cur.close()
        if user and user['password'] == password:
            # Store user data in session for authentication
            session['user_id'] = user['id']
            session['username'] = user['username']
            logged_in = True
            logout_success = False
            return redirect(url_for('index'))
        else:
            error = 'Invalid username or password.'
            return render_template('login.html', error=error)

    return render_template('login.html')

# Route for the logout page
@app.route('/logout')
def logout():
    global logged_in, logout_success  # Access the global variable
    logged_in = False  # Set logged_in status to False
    logout_success = True
    return redirect(url_for('index'))

# Route for the universities page
@app.route('/universities', methods=['GET', 'POST'])
def universities():
    if logged_in:
        if request.method == 'POST':
            # Get user input from the form
            input_type = request.form['input_type']
            input_value = request.form['input_value']
            
            if input_type == 'courses':
                # Filter universities based on courses
                universities_filtered = universities_df[universities_df['Courses'].str.contains(input_value, case=False)]
            elif input_type == 'city':
                # Filter universities based on city
                universities_filtered = universities_df[universities_df['City'].str.contains(input_value, case=False)]
            elif input_type == 'state':
                # Filter universities based on state
                universities_filtered = universities_df[universities_df['State'].str.contains(input_value, case=False)]
            else:
                # Render the universities.html template without any recommendations
                return render_template('universities.html', universities=None)
            
            if universities_filtered.empty:
                # Render the universities.html template without any recommendations
                return render_template('universities.html', universities=None)
            
            # Get the cosine similarity between input value and universities
            university_similarity = uni_cosine_similarities[universities_filtered.index].mean(axis=0)
            
            # Filter universities based on cosine similarity threshold
            similar_universities_indices = [i for i, similarity in enumerate(university_similarity) if similarity > threshold]
            similar_universities = universities_df.iloc[similar_universities_indices][:10]
            
            if similar_universities.empty:
                # Render the universities.html template without any recommendations
                return render_template('universities.html', universities=None)
            
            # Render the universities.html template with the similar universities
            return render_template('universities.html', universities=similar_universities, logged_in = logged_in)
        
        # Render the universities.html template without any recommendations
        return render_template('universities.html', universities=None, logged_in = logged_in)
    else:
        return redirect(url_for('login'))

@app.route('/forms')
def forms():
    if logged_in:
        return render_template('forms.html', logged_in = logged_in)
    else:
        return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if logged_in:
        # Get user input from form
        interests_hobbies = request.form['interests_hobbies']
        subjects_passionate = request.form['subjects_passionate']
        skills_talents = request.form['skills_talents']
        career_goals = request.form['career_goals']
        work_environment = request.form['work_environment']
        values_beliefs = request.form['values_beliefs']
        strengths_weaknesses = request.form['strengths_weaknesses']
        industry_field_thriving = request.form['industry_field_thriving']
        work_life_balance = request.form['work_life_balance']

        # Concatenate user input into a single string
        input_data = interests_hobbies + ' ' + subjects_passionate + ' ' + skills_talents + ' ' + career_goals + ' ' + work_environment + ' ' + values_beliefs + ' ' + strengths_weaknesses + ' ' + industry_field_thriving + ' ' + work_life_balance

        # Preprocess the input data
        input_data = ' '.join([word.lower() for word in input_data.split()])  # Convert to lowercase
        input_data = ' '.join([word for word in input_data.split() if word.isalpha()])  # Remove non-alphabetic characters

        # Vectorize the input data using the vocabulary
        vectorizer = CountVectorizer(vocabulary=vocabulary)
        input_data_vectorized = vectorizer.transform([input_data])

        # Predict the job industry
        predicted_career = model.predict(input_data_vectorized)[0]
        print(predicted_career)

        return render_template('result.html', predicted_career = predicted_career, logged_in = logged_in)
    else:
        return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)
