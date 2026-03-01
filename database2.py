import psycopg2 # NEW: Use psycopg2 instead of sqlite3
from psycopg2.extras import RealDictCursor
import pandas as pd
import hashlib
import json
import re
import datetime
import google.generativeai as genai
import streamlit as st # <--- ADD THIS LINE HERE
import time
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

# 1. REPLACE THIS WITH YOUR CLOUD URL (from Supabase/Neon)
DB_URL = st.secrets["NEON_DB_URL"]

def get_connection():
    # Helper to get a cloud connection instead of local file
    return psycopg2.connect(DB_URL)

def init_db():
    conn = get_connection()
    
    c = conn.cursor()
    # SQL syntax changes slightly for PostgreSQL (SERIAL instead of AUTOINCREMENT)
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS expenses (
                    id SERIAL PRIMARY KEY,
                    username TEXT,
                    date TEXT,
                    amount REAL,
                    description TEXT,
                    category TEXT)''')
    c.execute("""
        CREATE TABLE IF NOT EXISTS budget_limits (
            username TEXT PRIMARY KEY,
            limits JSONB
        )
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS subscriptions (
        id SERIAL PRIMARY KEY,
        username TEXT,
        service_name TEXT,
        amount FLOAT,
        renewal_date DATE,
        category TEXT)""")
    # --- Update inside your init_db function ---
    c.execute("""
    ALTER TABLE subscriptions 
        ADD COLUMN IF NOT EXISTS last_notified DATE
        """)
    c.execute("""
        ALTER TABLE users 
        ADD COLUMN IF NOT EXISTS telegram_chat_id TEXT
    """)
    # NEW: Add GST support to the expenses table
    c.execute("""
        ALTER TABLE expenses 
        ADD COLUMN IF NOT EXISTS gst_rate REAL DEFAULT 0,
        ADD COLUMN IF NOT EXISTS gst_amount REAL DEFAULT 0,
        ADD COLUMN IF NOT EXISTS is_business BOOLEAN DEFAULT FALSE
    """)
    c.execute("""
        ALTER TABLE users 
        ADD COLUMN IF NOT EXISTS salary REAL DEFAULT 60000
    """)
    c.execute("""
        ALTER TABLE users 
        ADD COLUMN IF NOT EXISTS is_premium BOOLEAN DEFAULT FALSE
    """)
    conn.commit()
    conn.close()

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

def create_user(username, password):
    conn = get_connection()
    c = conn.cursor()
    # Clean the input before hashing and saving
    c.execute('INSERT INTO users(username,password) VALUES (%s,%s)', 
              (username.strip(), make_hashes(password.strip())))
    conn.commit()
    conn.close()

def login_user(username, password):
    conn = get_connection()
    c = conn.cursor()
    # Using .strip() ensures accidental spaces don't break the login
    hashed_pw = make_hashes(password.strip())
    c.execute('SELECT * FROM users WHERE username = %s AND password = %s', (username.strip(), hashed_pw))
    data = c.fetchall()
    conn.close() # Always close your connection
    return data # Returns a list; an empty list evaluates to False in Streamlit


def get_all_expenses(username): # Added username
    conn = get_connection()
    # CRITICAL: Filter data so User A cannot see User B's vault
    query = "SELECT * FROM expenses WHERE username = %s ORDER BY date DESC"
    df = pd.read_sql_query(query, conn, params=(username,))
    conn.close()
    return df.to_dict('records')

def add_expense_to_db(username, date, amount, description, category, gst_rate=0, is_business=False):
    """
    Records a new expense in the user's vault.
    Args:
        username (str): The logged-in user's name.
        date (str): Date of expense in 'YYYY-MM-DD' format.
        amount (float): The total money spent.
        description (str): Brief detail of the expense.
        category (str): Choose from Food, Transport, Shopping, Bills, or Other.
    """
    """
    Records a new expense. Includes optional GST and business tagging.
    """
    # Calculate tax component based on the total amount
    # GST = Total - (Total / (1 + Rate/100))
    gst_amount = amount - (amount / (1 + gst_rate / 100)) if gst_rate > 0 else 0
    conn = get_connection()
    c = conn.cursor()
    c.execute("INSERT INTO expenses (username, date, amount, description, category, gst_rate, gst_amount, is_business) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", 
              (username, date, amount, description, category, gst_rate, gst_amount, is_business))
    conn.commit()
    conn.close()
    return f"Successfully added {description} for ₹{amount}."

def edit_expense_in_db(expense_id, amount=None, description=None, category=None):
    """
    Updates an existing expense record by its ID.
    Args:
        expense_id (int): The unique ID of the record to change.
        amount (float): New amount (optional).
        description (str): New description (optional).
        category (str): New category (optional).
    """
    conn = get_connection()
    c = conn.cursor()
    updates = []
    params = []
    if amount: updates.append("amount = %s"); params.append(amount)
    if description: updates.append("description = %s"); params.append(description)
    if category: updates.append("category = %s"); params.append(category)

    if updates:
        params.append(expense_id)
        c.execute(f"UPDATE expenses SET {', '.join(updates)} WHERE id = %s", params)
        conn.commit()
    conn.close()
    return f"Record {expense_id} has been updated."

def delete_expense_from_db(expense_id):
    """
    Permanently removes an expense record from the vault using its ID.
    Args:
        expense_id (int): The ID of the record to delete.
    """
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM expenses WHERE id = %s", (expense_id,))
    conn.commit()
    conn.close()
    return f"Record {expense_id} deleted."

# --- Add these new functions to database.py ---


def save_user_limits(username, limits_dict):
    """Saves or updates budget limits for a specific user."""
    conn = get_connection()
    cur = conn.cursor()
    # Using 'UPSERT' logic to insert or update the record
    cur.execute("""
        INSERT INTO budget_limits (username, limits)
        VALUES (%s, %s)
        ON CONFLICT (username) DO UPDATE SET limits = EXCLUDED.limits
    """, (username, json.dumps(limits_dict)))
    conn.commit()
    cur.close()
    conn.close()

def get_user_limits(username):
    """Fetches saved limits for a user from the cloud vault."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT limits FROM budget_limits WHERE username = %s", (username,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0] if row else None

def add_subscription_to_db(username, service, amount, renewal, category):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO subscriptions (username, service_name, amount, renewal_date, category)
        VALUES (%s, %s, %s, %s, %s)
    """, (username, service, amount, renewal, category))
    conn.commit()
    cur.close()
    conn.close()
    
def get_user_subscriptions(username):
    """Fetches all manually registered subscriptions for a user."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, service_name, amount, CAST(renewal_date AS TEXT),  category, last_notified FROM subscriptions WHERE username = %s", (username,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows # Returns a list of tuples

def delete_subscription_from_db(sub_id):
    """Permanently removes a subscription record from the cloud vault."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM subscriptions WHERE id = %s", (sub_id,))
    conn.commit()
    cur.close()
    conn.close()
    
def update_notified_status(sub_id, notified_date):
    """Marks a subscription as notified for a specific date."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE subscriptions SET last_notified = %s WHERE id = %s", (notified_date, sub_id))
    conn.commit()
    cur.close()
    conn.close()
    
# --- ADD TO database.py ---

def update_user_telegram_id(username, chat_id):
    """Saves or updates a user's Telegram Chat ID in the cloud."""
    conn = get_connection()
    cur = conn.cursor()
    # Using the existing users table
    cur.execute("UPDATE users SET telegram_chat_id = %s WHERE username = %s", (chat_id, username))
    conn.commit()
    cur.close()
    conn.close()

def get_telegram_id(username):
    """Retrieves the Telegram ID for a specific user."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT telegram_chat_id FROM users WHERE username = %s", (username,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result[0] if result else None

def get_user_data(username):
    """Fetches all profile information for a specific user."""
    conn = get_connection()
    cur = conn.cursor()
    # Using real_dict_cursor or returning a dictionary is best for Streamlit
    cur.execute("SELECT * FROM users WHERE username = %s", (username,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    
    # Map result to a dictionary so sanchay.py can read it
    if result:
        return {
            "username": result[0],
            "password": result[1],
            "telegram_chat_id": result[2],
            "salary": result[3] if len(result) > 3 else 60000,
            "is_premium": result[4] if len(result) > 4 else False # <-- ADD THIS
        }
    return None


# REMOVED @st.cache_data completely so it has to think fresh every time!
def predict_category(description):
    """Uses Gemini to predict the best category for a manual or SMS entry."""
    model = genai.GenerativeModel('gemini-2.5-flash') 
    
    prompt = f"""
    Classify the following expense into exactly ONE of these categories: 
    [Food, Transport, Shopping, Bills, Other]
    
    Description: "{description}"
    
    Return ONLY the category name. No punctuation or extra words.
    """
    try:
        response = model.generate_content(prompt)
        raw_text = str(response.text).lower()
        
        # Smart Cleaner
        for c in ["Food", "Transport", "Shopping", "Bills", "Other"]:
            if c.lower() in raw_text:
                return c
                
        return "Other"
        
    except Exception as e:
        # THE FIX: Stop silently hiding API errors! Show it on the screen.
        st.toast(f"⚠️ AI API Error: {e}", icon="🚨")
        time.sleep(1) # Let the user read the error message
        return "Other"

# --- Add this to the bottom of database.py ---
def update_user_salary(username, salary):
    """Permanently saves the user's monthly salary in the cloud vault."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("UPDATE users SET salary = %s WHERE username = %s", (salary, username))
        conn.commit()
    finally:
        cur.close()
        conn.close()
        
def upgrade_user_to_premium(username):
    """Permanently upgrades a user to the Sanchay++ premium tier in the cloud database."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        # PostgreSQL uses TRUE for boolean values
        cur.execute("UPDATE users SET is_premium = TRUE WHERE username = %s", (username,))
        conn.commit()
    except Exception as e:
        print(f"Error upgrading user: {e}")
    finally:
        cur.close()
        conn.close()