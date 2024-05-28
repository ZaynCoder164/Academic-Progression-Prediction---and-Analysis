
from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from flask import redirect, url_for
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor 
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
import sqlite3
import csv
import matplotlib.pyplot as plt
import seaborn as sns

# Create a Flask application
app = Flask(__name__,static_folder='static')
app.secret_key = 'your_secret_key'

# Initialize the SQLite3 database
def init_db():
    conn = sqlite3.connect('userpass.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
    conn.close()

init_db()
@app.after_request
def add_cache_control(response):
    response.cache_control.no_cache = True
    return response

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["Email"]
        password = request.form["Password"]
        code = request.form["Code"]
        if code != "AMCEC@123":
            flash("Code not verified. Please try again.")
            return redirect("/")

        # Check if email is in the correct format
        # Check if email is in the correct format
        if not email.endswith("@amcec.edu"):
            error_messages = ["Invalid email address. Please use an email address from @amcec.edu domain."]
            return render_template("login.html", error_messages=error_messages)


        # Store the email and password in the database
        conn = sqlite3.connect('userpass.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (email, password))
            conn.commit()
            flash("You have successfully signed up!")
            return redirect("/")
        except sqlite3.IntegrityError:
            flash("Email already exists.")
        finally:
            conn.close()

    return render_template("login.html")


# Define a login page
def valid_user(email, password):
    conn = sqlite3.connect('userpass.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (email, password))
    user = c.fetchone()
    conn.close()
    return user is not None

def is_logged_in():
    return 'email' in session

@app.route("/", methods=["GET", "POST"])
def login():
    error_message = ""
    if request.method == "POST":
        email = request.form["Email"]
        password = request.form["Password"]

        if not email.endswith("@amcec.edu"):
            error_message = "Invalid email address. Please use an email address from @amcec.edu domain."
            return render_template("login.html", error_message=error_message)

        if not valid_user(email, password):
            error_message = "Invalid email or password"
        else:
            session['email'] = email
            return redirect('/dashboard')

    return render_template("login.html", error_message=error_message)

import json

@app.route('/save-note', methods=['POST'])
def save_note():
    note_data = request.get_json()
    note_id = note_data['noteId']
    content = note_data['content']
    user_email = session['email']

    # Load the existing data from the JSON file
    with open('emps.json', 'r') as file:
        data = json.load(file)

    # Update the data for the specific user and note
    if user_email in data:
        data[user_email][note_id] = content
    else:
        data[user_email] = {note_id: content}

    # Save the updated data back to the JSON file
    with open('emps.json', 'w') as file:
        json.dump(data, file)

    # No return statement



@app.route("/dashboard")
def dashboard():
    df = pd.read_csv('data.csv')
    if not is_logged_in():
        return redirect(url_for('login'))
    return render_template('dashboard.html',students=df.to_dict('records'))

# Load the student data from a CSV file
df = pd.read_csv('data.csv')

# Define a predictor page
from flask import request
import math
@app.route("/students")
def students():
    df = pd.read_csv('data.csv')

    return render_template('students.html', students=df.to_dict('records'))

@app.route('/update-marks', methods=['POST'])
def update_marks():
    usn = request.form['usn']
    sem1 = request.form['sem1']
    sem2 = request.form['sem2']
    sem3 = request.form['sem3']
    sem4 = request.form['sem4']
    sem5 = request.form['sem5']
    
    # Update the marks in the CSV file
    with open('data.csv', 'r') as file:
        students = list(csv.DictReader(file))

    for student in students:
        if student['USN'] == usn:
            student['First sem'] = sem1
            student['Second sem'] = sem2
            student['Third sem'] = sem3
            student['Fourth sem'] = sem4
            student['Fifth sem'] = sem5

    with open('data.csv', 'w', newline='') as file:
        fieldnames = ['Name', 'USN', 'First sem', 'Second sem', 'Third sem', 'Fourth sem', 'Fifth sem', 'Sixth sem']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(students)

    # Reload the DataFrame with the updated data
    df = pd.read_csv('data.csv')

    return render_template('dashboard.html', students=df.to_dict('records'))

@app.route("/predictor")
def predictor():
    # Load the student data from a CSV file
    df = pd.read_csv('data.csv')
    # Render the predictor.html template with the student data
    return render_template('predictor.html', students=df.to_dict('records'))

# Define a view page for each student

# Define a check_marks page
def load_data():
    data = []
    with open('data.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

# Retrieve the marks for a specific student
def get_marks(usn):
    data = load_data()
    for row in data:
        if row['USN'] == usn:
            return row
    return None

@app.route("/view/<int:usn>")
def view(usn):
    df = pd.read_csv('data.csv', dtype={'USN': int})
    student = df[df['USN'] == usn].to_dict('records')
    if not student:
        return "Student not found", 404
    print(student[0])  # Debug: Print the student data
    return render_template('view.html', student=student[0])

# Define your routes and functions...
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from flask import jsonify

@app.route('/results/<int:usn>')
def results(usn):
    df = pd.read_csv('data.csv')
    student = df.loc[df['USN'] == usn]
    student_name = student['Name'].values[0]  # Get the student's name
    student_usn = student['USN'].values[0]  # Get the student's USN

    sem1 = student['First sem'].values[0]
    sem2 = student['Second sem'].values[0]
    sem3 = student['Third sem'].values[0]
    sem4 = student['Fourth sem'].values[0]
    sem5 = student['Fifth sem'].values[0]

    X = df.iloc[:, 2:-1].values
    y = df.iloc[:, -1].values

    input_data = [[sem1, sem2, sem3, sem4, sem5]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     # Initialize and train the XGBoost model
    model = XGBRegressor()
    model.fit(X_train, y_train)

    # Make predictions
    input_data = np.array([[sem1, sem2, sem3, sem4, sem5]])
    predicted_marks = model.predict(input_data)

    # Calculate R2 score
    y_pred_train = model.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)

    y_pred_test = model.predict(X_test)
    r2_test = r2_score(y_test, y_pred_test)

    mse = mean_squared_error(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = mean_squared_error(y_test, y_pred_test, squared=False)

    first_semester_marks = student.iloc[:, 2].values
    last_semester_marks = student.iloc[:, 7].values
    percentage_change1 = ((last_semester_marks - first_semester_marks) / first_semester_marks) * 100
    percentage_change = "{:.2f}%".format(percentage_change1[0])

    #Using violin plot (the violin plot compares the distribution of actual marks (blue) with the distribution of predicted marks (orange). By observing the shape and spread of the violins, we can gain insights into how closely the predicted marks align with the actual marks.
    #If the two violins exhibit similar shapes and spreads, it suggests that the model's predictions closely match the actual data.)
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=[y_test, y_pred_test], palette='pastel') #muted, deep
    plt.xticks([0, 1], ['Actual Marks', 'Predicted Marks'])
    plt.ylabel('Marks')
    plt.title('Distribution of Actual vs. Predicted Marks')
    graph_filename = 'violin_plot.png'
    plt.savefig(os.path.join('static', graph_filename))

    return render_template('results.html', student=student, student_name=student_name, student_usn=student_usn, usn=usn, sem1=sem1, sem2=sem2, sem3=sem3, sem4=sem4, sem5=sem5,predicted_marks=predicted_marks[0],r2_train=r2_train, r2_test=r2_test,
                               mse=mse, mae=mae, rmse=rmse, percentage_change=percentage_change, graph_filename=graph_filename)


@app.route("/logout",methods=['GET', 'POST'])
def logout():
    session.pop('email', None)
    return redirect(url_for('login'))
if __name__ == '__main__':
    app.secret_key = 'your_secret_key'
    app.run(debug=True)
