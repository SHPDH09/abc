# 🧠 AI-Based Workforce Productivity & Burnout Analyzer

An AI-powered web application built with **Flask** that helps analyze an employee's productivity and burnout risk level based on working hours, tasks completed, breaks taken, and satisfaction level.  

---

## 🚀 Features
- Predict employee burnout risk using a trained ML model (`burnout_model.pkl`)
- Store and view prediction history in an SQLite database
- Modern, responsive UI with gold-gray theme
- Secure and scalable structure
- View results in **Low, Medium, and High Burnout Risk**
- Easy to deploy (PythonAnywhere, Render, or Heroku)

---

## 📂 Project Structure
AI-Based-Workforce-Productivity-Burnout-Analyzer/
│
├── app.py # Main Flask application
├── burnout_model.pkl # Trained Machine Learning Model
├── employee_data.csv # Sample dataset for training/testing
├── employee_results.db # SQLite database (auto-generated)
├── requirements.txt # Required dependencies
│
├── templates/ # HTML templates
│ ├── index.html
│ ├── result.html
│ └── history.html
│
├── static/
│ └── css/
│ └── style.css # Styling for UI
│
└── README.md

yaml
Copy
Edit

---

## ⚙️ Installation & Setup
### 1️⃣ Clone the repository
```bash
git clone https://github.com/SHPDH09/AI-Based-Workforce-Productivity-Burnout-Analyzer.git
cd AI-Based-Workforce-Productivity-Burnout-Analyzer
2️⃣ Create virtual environment
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate     # On Windows
source venv/bin/activate  # On Mac/Linux
3️⃣ Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Run the application
bash
Copy
Edit
python app.py
The application will run at:

cpp
Copy
Edit
http://127.0.0.1:5000
🛠 Technologies Used
Python 3.x

Flask – Web Framework

SQLite – Database

HTML, CSS (Gold-Gray Theme) – Frontend

Scikit-learn, Joblib, NumPy – Machine Learning

📊 Burnout Risk Levels
Risk Level	Description
Low	Healthy work-life balance
Medium	Needs improvement in breaks or hours
High	Immediate attention required

🌐 Deployment
You can host this project on:

PythonAnywhere – Free, beginner-friendly

Render – Free tier available

Heroku – Git-based deployment

✨ Author
Name: Raunak Kumar

Email: rk331159@gmail.com

Portfolio: https://portfolioraunakprasad.netlify.app/

📜 License
This project is licensed under the MIT License.
You are free to use, modify, and distribute this project.