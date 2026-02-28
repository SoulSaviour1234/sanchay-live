
import os
import io
import streamlit as st
import datetime
import time
import requests
import json
import pandas as pd
import plotly.express as px
import google.generativeai as old_genai
# Add your new functions to the import list
# Update your import line (around Line 15) to include the new function
from database2 import (
    get_telegram_id, get_user_subscriptions, init_db, add_expense_to_db, 
    get_all_expenses, create_user, login_user, edit_expense_in_db, 
    delete_expense_from_db, save_user_limits, get_user_limits, 
    add_subscription_to_db, delete_subscription_from_db, get_connection, 
    update_notified_status, get_user_data, update_user_telegram_id, 
    predict_category, update_user_salary, upgrade_user_to_premium  # <--- ADD THIS
)

# --- CLOUD TIMEZONE FIX (Forces IST) ---
os.environ['TZ'] = 'Asia/Kolkata'
if hasattr(time, 'tzset'):
    time.tzset()

# --- NEW: AI TOOL FOR UPDATING BUDGETS ---
def update_category_budget_ai(username, category, new_limit):
    """Updates the monthly budget limit for a single category (Food, Transport, Shopping, Bills, Other)."""
    current_limits = get_user_limits(username)
    if not current_limits:
        current_limits = {"Food": 5000.0, "Transport": 2000.0, "Shopping": 3000.0, "Bills": 10000.0, "Other": 1500.0}
        
    # Ensure category matches exact formatting
    category = category.capitalize()
    if category in current_limits:
        current_limits[category] = float(new_limit)
        save_user_limits(username, current_limits)
        st.session_state.category_limits = current_limits # Updates the live UI
        return f"Successfully updated {category} budget envelope to ₹{new_limit}."
    return f"Error: Category must be exactly Food, Transport, Shopping, Bills, or Other."


def generate_gst_report(df):
    """Calculates GST components and returns a business-ready DataFrame."""
    if df.empty:
        return df
    
    # Assume 18% if not specified, or use the value from the DB
    df['GST Rate (%)'] = 18.0 
    # Formula: GST = Total - (Total / (1 + Rate/100))
    df['Taxable Value'] = df['amount'] / (1 + df['GST Rate (%)'] / 100)
    df['GST Amount'] = df['amount'] - df['Taxable Value']
    
    # Split into CGST and SGST (9% each for 18% total)
    df['CGST (9%)'] = df['GST Amount'] / 2
    df['SGST (9%)'] = df['GST Amount'] / 2
    
    return df

def decode_upi_qr(image_file):
    """Decodes a UPI QR image and returns the payment URI."""
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Initialize the QR code detector
    detector = cv2.QRCodeDetector()
    data, bbox, straight_qrcode = detector.detectAndDecode(img)
    
    if data:
        # UPI URIs look like: upi://pay?pa=address@bank&pn=Name...
        return data
    return None

# --- NEW: Tool for the AI to check affordability ---
def get_financial_health_stats(username):
    """Calculates savings trends and disposable income for the AI."""
    df = pd.DataFrame(get_all_expenses(username))
    limits = get_user_limits(username) or {}
    
    if df.empty:
        return "No data available yet."

    # Calculate average daily spend over the last 30 days
    df['date'] = pd.to_datetime(df['date'])
    last_30_days = df[df['date'] > (datetime.datetime.now() - datetime.timedelta(days=30))]
    avg_daily_spend = last_30_days['amount'].sum() / 30
    
    # Calculate total monthly envelope limits
    total_monthly_limit = sum(limits.values())
    
    return {
        "avg_daily_spend": round(avg_daily_spend, 2),
        "monthly_limit_set": total_monthly_limit,
        "recent_spending_trend": "High" if avg_daily_spend > 1000 else "Stable"
    }

def send_telegram_notification(target_id, message):
    """Sends a free push notification to your phone via Telegram."""
    # REPLACE THESE with your real data
    token = st.secrets["TELEGRAM_BOT_TOKEN"]
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    headers = {"Content-Type": "application/json"}
    payload = {"chat_id": target_id, "text": message}
    
    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        # Force the error to appear on your Sanchay dashboard
        if response.status_code != 200:
            st.error(f"❌ Telegram Error: {response.text}")
            return False
        return True
    except Exception as e:
        st.error(f"🌐 Network Error: {e}")
    return False


def get_envelope_status(df, current_user, category_limits):
    """Calculates how much of the monthly envelope is remaining for each category."""
    user_df = df[df['username'] == current_user].copy()
    if user_df.empty:
        return {cat: 0.0 for cat in category_limits}

    # Filter for the current month
    user_df['date_dt'] = pd.to_datetime(user_df['date'])
    current_month = datetime.datetime.now().month
    month_df = user_df[user_df['date_dt'].dt.month == current_month]

    # Calculate spending per category
    category_spending = month_df.groupby('category')['amount'].sum().to_dict()
    
    status = {}
    for cat, limit in category_limits.items():
        spent = category_spending.get(cat, 0.0)
        status[cat] = {"spent": spent, "limit": limit, "remaining": limit - spent}
    return status

def get_garden_state(streak, remaining_ratio):
    """Determines the growth stage and health of the user's virtual tree."""
    # Growth Stage based on Streak
    if streak >= 30: stage = "🌳 Ancient Oak"
    elif streak >= 15: stage = "🌳 Mature Tree"
    elif streak >= 7: stage = "🌿 Growing Sapling"
    elif streak >= 3: stage = "🌱 Sprout"
    else: stage = "🌰 Seed"

    # Health status based on daily budget remaining
    if remaining_ratio > 0.5: health = "🌟 Radiant"
    elif remaining_ratio > 0: health = "🍃 Healthy"
    else: health = "🍂 Wilting (Overspent)"
    
    return stage, health

def check_quests(df, current_user):
    """Checks for completed challenges for the specific user."""
    trophies = []
    # Filter to ensure we only analyze the current user's data
    user_df = df[df['username'] == current_user].copy()
    
    if user_df.empty:
        return trophies

    # Ensure date is in datetime format for calculation
    user_df['date_dt'] = pd.to_datetime(user_df['date'])
    daily_sums = user_df.groupby('date')['amount'].sum()
    
    # NEW: Find the most recent date where spending was < ₹300
    frugal_days = daily_sums[daily_sums < 300]
    if not frugal_days.empty:
        # Get the latest date from that list
        latest_frugal_date = frugal_days.index[-1] 
        trophies.append(f"👑 Frugal King (Unlocked on {latest_frugal_date})")
        
    # 3. UPDATED Quest: The Minimalist (Flex Mode)
    if len(daily_sums) >= 3:
        # Get data from the last 3 active days
        last_3_days = daily_sums.tail(3).index
        mask = user_df['date'].isin(last_3_days)
        window_df = user_df[mask]
        
        # Identify all unique categories used across these 3 days
        unique_cats_in_window = window_df['category'].unique().tolist()
        
        # SUCCESS: Total unique categories for the whole 3 days is 1 or 2
        if 0 < len(unique_cats_in_window) <= 5:
            cat_str = " & ".join(unique_cats_in_window)
            trophies.append(f"🧘 The Minimalist (3-day focus on {cat_str}!)")
    
    # QUEST 1: The No-Spend Weekend (No entries on Sat/Sun)
    # Find the most recent Saturday and Sunday dates
    today = datetime.date.today()
    # Subtracting days based on weekday to find the previous weekend
    days_since_sunday = (today.weekday() + 1) % 7
    last_sunday = today - datetime.timedelta(days=days_since_sunday)
    last_saturday = last_sunday - datetime.timedelta(days=1)
    
    # Check for spending on those specific dates
    recent_weekend_spending = user_df[user_df['date_dt'].dt.date.isin([last_saturday, last_sunday])]
    
    if recent_weekend_spending.empty:
        trophies.append(f"🛡️ Weekend Guardian ({last_saturday} to {last_sunday})")

    return trophies

def get_quest_progress(df, current_user):
    """Calculates progress percentage for each active quest."""
    progress = {"Frugal King": 0.0, "Weekend Guardian": 0.0, "The Minimalist": 0.0}
    user_df = df[df['username'] == current_user].copy()
    
    if user_df.empty:
        return progress

    user_df['date_dt'] = pd.to_datetime(user_df['date'])
    daily_sums = user_df.groupby('date')['amount'].sum()
    today = datetime.date.today()

    # 1. Frugal King Tracker
    today_str = today.strftime("%Y-%m-%d")
    today_spend = daily_sums.get(today_str, 0)
    progress["Frugal King"] = max(0.0, min(1.0, (300 - today_spend) / 300)) if today_spend < 300 else 0.0

    # 2. Weekend Guardian Tracker
    if today.weekday() == 5: # Saturday
        progress["Weekend Guardian"] = 0.5 if today_str not in daily_sums or daily_sums[today_str] == 0 else 0.0
    elif today.weekday() == 6: # Sunday
        progress["Weekend Guardian"] = 1.0 
    
    # 3. UPDATED: The Minimalist Tracker (Rewards 2+ categories for 3 days)
    streak = 0
    if len(daily_sums) >= 1:
        for d in reversed(daily_sums.tail(3).index):
            # Checking for your NEW multi-category (2+) requirement
            if user_df[user_df['date'] == d]['category'].nunique() >= 2:
                streak += 1
            else: break
    progress["The Minimalist"] = min(1.0, streak / 3)

    return progress

def calculate_streak(df):
    """Calculates the number of consecutive days with recorded expenses."""
    if df.empty or 'date' not in df.columns:
        return 0
    
    # Get unique dates and sort them descending
    unique_dates = pd.to_datetime(df['date']).dt.date.unique()
    unique_dates = sorted(unique_dates, reverse=True)
    
    streak = 0
    today = datetime.date.today()
    current_check = today
    
    # Check back from today to see how far the streak goes
    for date in unique_dates:
        if date == current_check:
            streak += 1
            current_check -= datetime.timedelta(days=1)
        elif date == today - datetime.timedelta(days=1):
            # If they haven't added today yet, but did yesterday, streak is still alive
            continue 
        else:
            break
    return streak

def get_badge(streak):
    """Returns a badge name and emoji based on the streak length."""
    if streak >= 30: return "You are 💎 Wealth Legend", "Gold"
    if streak >= 15: return "You are 🔥 Savings Warrior", "Orange"
    if streak >= 7:  return "You are 🎖️ Week Winner", "Green"
    if streak >= 3:  return "You are 🌱 Budget Starter", "Blue"
    return "You are 🥚 Newbie", "Grey"

@st.cache_data(ttl=3600)
def detect_anomalies(df, user):
    """Uses Gemini to identify duplicates or price hikes in the vault."""
    if df.empty:
        return None
        
    # We only send the most recent 20 transactions to save tokens and focus on fresh data
    recent_data = df.tail(20).to_string()
    
    model = old_genai.GenerativeModel('gemini-2.5-flash') # Use the same model as in sanchay.py
    prompt = f"""
    You are a financial auditor for {user}. Review these recent transactions:
    {recent_data}
    
    Identify:
    1. Exact Duplicates: Same amount, description, and date.
    2. Price Hikes: Same description/category but a significant increase in amount compared to previous entries.
    3. Unusual Spikes: Amounts much higher than the average for that category.
    
    If you find any, list them as bullet points starting with "⚠️". 
    If none are found, return "No anomalies detected."
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error detecting anomalies: {e}")
        return None


def process_receipt(image_file):
    """Uses Gemini Vision to extract details from a receipt image."""
    model_vision = old_genai.GenerativeModel('gemini-2.5-flash') # Vision-capable model
    
    prompt = """
    Analyze this receipt carefully and classify it into ONE of these specific categories:
    - Food: Restaurants, cafes, groceries, bakeries, or delivery (e.g., Star Market, Swiggy, KFC).
    - Transport: Fuel/petrol, bus, taxi, train, or metro (e.g., Shell, Uber, Ola, Metro).
    - Shopping: Clothes, electronics, furniture, or generic retail (e.g., Zara, Sai Mart, H&M).
    - Bills: Electricity, water, internet, insurance, or rent.
    - Other: Any expense that doesn't fit the above.

    STRICT RULES:
    1. If the Merchant Name is "SAI MART", look at the items. If it's mostly clothes, it's Shopping. If it's snacks or groceries, it's Food.
    2. Extract Merchant Name, Date (YYYY-MM-DD), and Total Amount.
    
    Return the result EXACTLY in this format:
    Merchant Name | Date | Total Amount | Category (Food, Transport, Shopping, Bills, or Other)
    
    Example: Star Market | 2026-01-14 | 1250.50 | Food
    """
    
    # Convert uploaded file to bytes for Gemini
    img_data = image_file.getvalue()
    response = model_vision.generate_content([prompt, {'mime_type': 'image/jpeg', 'data': img_data}])
    return response.text.strip()

# --- ADD THIS CONFIGURATION AT THE TOP (Right after imports) ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] # Use your actual key
old_genai.configure(api_key=GEMINI_API_KEY)

# Set up Page Branding
st.set_page_config(page_title="Sanchay - Smart Expense Tracker", page_icon="💰", layout="wide")

init_db()

# --- NEW: Session State for Authentication ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = ""
    
# --- PASTE THE LIMIT INITIALIZATION HERE ---
if 'category_limits' not in st.session_state:
    # Try to fetch from Neon first
    saved_limits = get_user_limits(st.session_state.user)
    if saved_limits:
        st.session_state.category_limits = saved_limits
    else:
        st.session_state.category_limits = {
            "Food": 5000.0, 
            "Transport": 2000.0, 
            "Shopping": 3000.0, 
            "Bills": 10000.0, 
            "Other": 1500.0
        }
    
    # --- Place this after your session state initialization ---
def login_page():
    st.subheader("🔐 User Login & Registration")
    choice = st.radio("Select Action", ["Login", "Sign Up"], horizontal=True)

    if choice == "Login":
        user = st.text_input("Username")
        passwd = st.text_input("Password", type='password')
        if st.button("Login", use_container_width=True):
            # login_user must be defined in your database.py
            if login_user(user, passwd):
                st.session_state.logged_in = True
                st.session_state.user = user
            else:
                st.error("Invalid Username or Password")

    else:
        new_user = st.text_input("Choose Username")
        new_passwd = st.text_input("Choose Password", type='password')
        if st.button("Create Account", use_container_width=True):
            # create_user must be defined in your database.py
            create_user(new_user, new_passwd)
            st.success("Account created! Please switch to Login.")
            
# --- Logic Gate ---
if not st.session_state.logged_in:
    login_page()
else:
    
    # NEW: Initialize widget versioning to fix the "sticky" dropdown issue
    if 'w_version' not in st.session_state: st.session_state.w_version = 0
    
    
    
    
    # --- UPDATED: Granting the AI access to the new tools ---
    sanchay_tools = [
        add_expense_to_db, get_all_expenses, edit_expense_in_db, delete_expense_from_db, 
        get_financial_health_stats, predict_category,
        get_user_limits, get_user_subscriptions, add_subscription_to_db, update_category_budget_ai
    ]
    # ADD A LOGOUT OPTION IN THE SIDEBAR
        
    
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False

    

    with st.container():
        # --- NEW PREMIUM HEADER & UNLOCK BUTTON ---
        # 1. Fetch user data to check premium status
        user_data = get_user_data(st.session_state.user)
        st.session_state.is_premium = user_data.get('is_premium', False) if user_data else False

        # 2. Dynamic Header Layout
        header_col1, header_col2 = st.columns([4, 1])
        
        with header_col1:
            if st.session_state.is_premium:
                st.title("💰 Sanchay++")
                st.caption("Premium Vault Manager by **RAAJPAKHI**")
            else:
                st.title("💰 Sanchay")
                st.caption("A product by **RAAJPAKHI**")

        with header_col2:
            if not st.session_state.is_premium:
                # Use Streamlit's native dialog for the pop-up
                @st.dialog("🚀 Upgrade to Sanchay++")
                def show_upgrade_modal():
                    st.write("### Scan, Pay and Unlock")
                    st.write("Pay **₹5** to permanently unlock AI God Mode, Smart Receipt Scanning, Business GST Exports, and more!")
                    
                    # Fixed Amount UPI Format. (Make sure to replace YOUR_UPI_ID@BANK with your real UPI ID)
                    # Fetch UPI ID from secrets
                    my_upi = st.secrets["UPI_ID"]
                    st.image(f"https://api.qrserver.com/v1/create-qr-code/?size=250x250&data=upi://pay?pa={my_upi}%26pn=Sanchay%26am=5.00%26cu=INR", width=250)
                    
                    utr_code = st.text_input("Enter 12-Digit UTR / Reference Number after paying:")
                    if st.button("Verify Payment & Unlock", use_container_width=True):
                        if len(utr_code) == 12:
                            # Call the DB function you added to database2.py
                            upgrade_user_to_premium(st.session_state.user)
                            st.success("Payment verified! Welcome to Sanchay++ 👑")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("Invalid UTR. Please check your payment app.")

                if st.button("✨ Unlock Sanchay++", type="primary", use_container_width=True):
                    show_upgrade_modal()
        st.divider()
        # --- Step 1: Salary Input ---
        # --- Locate the Salary Input in your Sidebar (Approx Line 500) ---
        with st.sidebar:
            # 1. Fetch the existing saved salary from the database first
            user_data = get_user_data(st.session_state.user)
            saved_salary = user_data.get('salary', 60000) if user_data else 60000

            # 2. Display the input box with the saved value
            new_salary = st.number_input(
                "Enter your Monthly Salary (₹):", 
                min_value=0, 
                value=int(saved_salary), 
                step=500
            )

            # 3. Add a button to save this value permanently
            if st.button("💾 Save Salary & Budget", use_container_width=True):
                # Use the dedicated function to avoid UndefinedColumn errors
                update_user_salary(st.session_state.user, new_salary)
                st.success("Salary updated permanently!")
                time.sleep(1)
                st.rerun()

            daily_salary = new_salary / 30
            st.sidebar.metric("Daily Budget", f"₹{daily_salary:.2f}")
    
        # --- NEW: SCAN & PAY SECTION ---
        with st.expander("⚡ Scan & Pay (UPI)", expanded=False):
            st.write("Scan a QR code to pay via GPay or PhonePe")
            # NEW: Toggle between Live Camera and File Upload
            mode = st.radio("Choose Scanner Mode", ["Upload Image", "Live Camera"], horizontal=True)
            qr_data = None
            if mode == "Live Camera":
                # This opens the actual camera feed in the browser
                cam_image = st.camera_input("Point at QR Code")
                if cam_image:
                    qr_data = decode_upi_qr(cam_image)
                    # SHOW ERROR ONLY IF A PHOTO WAS TAKEN BUT NO DATA FOUND
                    if not qr_data:
                        st.error("No valid UPI data found in this photo. Try again!")
            else:
                qr_file = st.file_uploader("Upload QR Photo", type=['jpg', 'png', 'jpeg'])
                if qr_file:
                    qr_data = decode_upi_qr(qr_file)
                    # SHOW ERROR ONLY IF A FILE WAS UPLOADED BUT NO DATA FOUND
                    if not qr_data:
                        st.error("No valid UPI data found in this file.")
                
            if qr_data:
                upi_uri = qr_data
                # Parsing the UPI URI
                # pa = Address, pn = Merchant Name, am = Amount (sometimes present)
                params = {x.split('=')[0]: x.split('=')[1] for x in upi_uri.split('?')[1].split('&')}
                merchant = params.get('pn', 'Merchant').replace('%20', ' ')
                fixed_amt = params.get('am', 0.0)

                st.success(f"✅ Ready to pay {merchant}!")

                # --- THE MAGIC STEP: Pre-fill Quick Entry ---
                if st.button(f"Confirm & Pay {merchant}"):
                    st.session_state.temp_desc = f"UPI: {merchant}"
                    st.session_state.temp_amt = float(fixed_amt)
                    # Use Gemini to predict category from the merchant name
                    st.session_state.temp_cat = predict_category(merchant) 
                    st.session_state.w_version += 1 # Forces UI refresh
                    st.toast("Redirecting to Payment App...", icon="🚀")
                    # 5. TRIGGER REDIRECT (using the URI decoded from QR)
                    # We use a slight delay so you can see the toast before GPay opens
                    # STEP 2: The actual UPI Launch Button
                    # Using a raw HTML link is the ONLY way to bypass mobile security reliably
                    st.markdown(f"""
                        <a href="{upi_uri}" target="_blank" style="text-decoration: none;">
                            <div style="
                                background-color: #1D72B8;
                                color: white;
                                padding: 15px;
                                text-align: center;
                                border-radius: 10px;
                                font-weight: bold;
                                margin-top: 10px;
                                cursor: pointer;">
                                🚀 2️⃣ PAY VIA GPAY / PHONEPE
                            </div>  
                        </a>
                    """, unsafe_allow_html=True)
                    st.toast(f"Details saved for {merchant}! Log it when you return.", icon="📝")
                    st.rerun()
                
        
    
        with st.container(border=True):
            st.subheader("➕ Quick Entry")
        
            # --- PREMIUM GATE: Receipt Scanning Section ---
            if st.session_state.is_premium:
                uploaded_file = st.file_uploader("Scan Receipt (JPG/PNG)", type=["jpg", "jpeg", "png"])
            
                if uploaded_file is not None:
                    if st.button("✨ Auto-Fill from Receipt", use_container_width=True):
                        with st.spinner("Sanchay AI is reading your receipt..."):
                            result = process_receipt(uploaded_file)
                            try:
                                # Extract the data from the AI's piped output
                                merchant, r_date, r_amount, r_category = result.split(" | ")
                            
                                # Store in session state to "fill" the inputs below
                                st.session_state.temp_amt = float(r_amount) if r_amount != "Unknown" else 0.0
                                st.session_state.temp_desc = merchant
                                st.session_state.temp_cat = r_category if r_category in ["Food", "Transport", "Shopping", "Bills", "Other"] else "Other"
                                # NEW: Increment version to force the UI to unlock
                                st.session_state.w_version += 1
                                st.success(f"Details extracted: {merchant} for ₹{r_amount} (Category: {r_category})")
                            except Exception:
                                st.error("Could not read receipt clearly. Please enter manually.")
            else:
                st.info("🔒 **Sanchay++ Exclusive Feature**")
                st.markdown("""
                <div style='filter: blur(2px); opacity: 0.6; pointer-events: none; border: 1px solid #444; padding: 20px; border-radius: 10px; margin-bottom: 15px;'>
                    📸 <i>Drop a receipt image here to auto-extract Merchant, Date, Amount, and Category using AI.</i><br>
                    <button disabled style='margin-top:10px; padding: 5px 10px; border-radius: 5px; background-color: #333; color: #888; border: 1px solid #555;'>Upload Image</button>
                </div>
                """, unsafe_allow_html=True)
                
            col1, col2, col3, col4 = st.columns([1, 2, 1.2, 1.2], vertical_alignment="bottom")
        
            with col1:
                # Use .get() to catch the Auto-Fill values
                amt = st.number_input("Amount (₹)", min_value=0.0,
                                 value=st.session_state.get('temp_amt', 0.0), step=10.0,
                                 key=f"amt_input_{st.session_state.w_version}")
            with col2:
                desc = st.text_input("Description (with AI Categorization)", placeholder="e.g. Metro fare", 
                                value=st.session_state.get('temp_desc', ""), key=f"manual_desc_input_{st.session_state.w_version}")
            
                # THE FIX: Add a small delay or button for the auto-categorizer, 
                # or rely on the fact that Streamlit only triggers when the user hits 'Enter'
                if desc and desc != st.session_state.get('last_processed_desc'):
                    with st.spinner("AI Categorizing..."):
                        # Now it will only run when the description actually changes!
                        predicted = predict_category(desc)
                        st.session_state.temp_cat = predicted
                        st.session_state.last_processed_desc = desc
                        st.session_state.w_version += 1
                    
            with col3:
                # Use .index() to find the position of the predicted category
                options = ["Food", "Transport", "Shopping", "Bills", "Other"]
                default_cat = st.session_state.get('temp_cat', "Food")
            
                # Find the position (index) of the predicted category in the list
                try:
                    cat_index = options.index(default_cat)
                except ValueError:
                    cat_index = 0 # Fallback to Food if error
        
                # FIXED: Added a dynamic key based on w_version
                cat = st.selectbox("Category", options, index=cat_index, key=f"cat_dropdown_{st.session_state.w_version}")
            
            with col4:
                # THE BUSINESS TAG: This unlocks the GST Portal
                is_biz = st.checkbox("💼 Business?", key=f"biz_{st.session_state.w_version}")
        
                # NEW: Only show GST rate if it is a Business expense
                gst_rate = 0
                if is_biz:
                    gst_rate = st.selectbox("GST Rate (%)", [5, 12, 18, 28], index=2) # Default to 18%
            
            if st.button("Save to Sanchay", use_container_width=True):
                if desc:
                    date = datetime.datetime.now().strftime("%Y-%m-%d")
                    add_expense_to_db(st.session_state.user, date, amt, desc, cat, gst_rate, is_biz)
                
                    # Clear the "temp" session state after saving
                    if 'temp_amt' in st.session_state: del st.session_state.temp_amt
                    if 'temp_desc' in st.session_state: del st.session_state.temp_desc
                
                    st.success(f"Added: {desc} of Rs. {amt} to {cat}")
                    st.success(f"✅ {'Business' if is_biz else 'Personal'} expense logged!")
                else:
                    st.error("Please enter a description.")
                
        # --- Step 3: Responsive Dashboard Logic ---
        # FETCH DATA FROM DATABASE HERE
        # Pass the session state user to filter the database
        data_list = get_all_expenses(st.session_state.user)
        df = pd.DataFrame(data_list)
    
        # Initialize dashboard variables with defaults
        total_spent = 0.0
        remaining = daily_salary
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
        # FIX 2: Only process data if the vault isn't empty to avoid KeyError
        # --- Around Line 515 in sanchay.py ---
        if not df.empty and 'date' in df.columns:
            # 1. Force the database dates into YYYY-MM-DD format only
            # 'errors=coerce' handles any weird entries that might break the app
            df['date_dt'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    
            # 2. Get today's date in the EXACT same format
            today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
            # 3. Filter the dataframe
            today_df = df[df['date_dt'] == today_str]
    
            # 4. Calculate total (this is the real-time sum)
            total_spent = float(today_df['amount'].sum())
    
            remaining = daily_salary - total_spent
        else:
            total_spent = 0.0
            remaining = daily_salary
    
        # --- PREMIUM GATE: AI Security Scan (Anomaly Detection) ---
        if not df.empty:
            if st.session_state.is_premium:
                with st.expander("🔍 AI Security Scan", expanded=False):
                    anomalies = detect_anomalies(df, st.session_state.user)
                    if anomalies and "No anomalies" not in anomalies:
                        st.warning(anomalies)
                    else:
                        st.success("✅ No unusual spending detected in your recent history.")
            else:
                # --- FREE USER BLURRED VIEW ---
                with st.expander("🔍 AI Security Scan (Locked)", expanded=False):
                    st.error("🔒 **Sanchay++ Exclusive Security Feature**")
                    st.markdown("""
                    <div style='filter: blur(2.5px); opacity: 0.5; pointer-events: none; border: 1px solid #333; border-radius: 10px; padding: 15px; margin-bottom: 10px;'>
                        <p style='margin: 0 0 10px 0;'>⚠️ <b>Duplicate Detected:</b> "Metro fare" for ₹40.00 logged twice on 2026-03-01.</p>
                        <p style='margin: 0 0 10px 0;'>⚠️ <b>Price Spike:</b> "Electricity Bill" is 45% higher than your average for this category.</p>
                        <p style='margin: 0;'>✅ <b>Resolution:</b> Delete duplicate entry?</p>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("Unlock Security Scan", use_container_width=True):
                        st.warning("Click the 'Unlock Sanchay++' button at the top of the dashboard!")
                
            streak_count = calculate_streak(df)
            badge_name, badge_color = get_badge(streak_count)
        
            st.divider()
    
            # Replace your current sidebar initialization with this:
            with st.sidebar:
                # This radio button will now live permanently in the left sidebar
                page = st.radio(
                    "Select Workspace:",
                    ["📊 Live Dashboard", "📋 Manage Records", "💼 Business Hub", "🧾 Receipt Editor"],
                    label_visibility="collapsed"
                )

            with st.container():
                if page == "📊 Live Dashboard":
                    # 2. Handle the "No Daily Expenses" state
                    if today_df.empty:
                        st.info("🚀 **Welcome, SOUL SAVIOUR! Your vault is currently empty.**")
                        st.info("✨ **No expenses recorded for today yet!** Your full daily budget is available.")
                        st.markdown("Start by adding an expense manually or scanning a receipt above.")
                        st.metric("Daily Budget Available: ", f"₹{daily_salary:.2f}")
                        st.metric("Remaining Balance: ", f"₹{daily_salary:.2f}")
                    else:
                        st.write("### Today's Financial Health")
                        # Metrics with 'deltas' to show spending vs balance
                        m1, m2 = st.columns(2)
                        # Side-by-side Layout for Analysis
                        c1, c2 = st.columns([1.5, 1], vertical_alignment="top")
                        with c1:
                            st.write("#### Category Breakdown")
                            fig = px.pie(df, values='amount', names='category', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
                            st.plotly_chart(fig, use_container_width=True)
                        with c2:
                            st.write("#### Insights")
                            st.info(f"Main Expense: **{df.groupby('category')['amount'].sum().idxmax()}**")
                            m1.metric("Total Spent", f"₹{total_spent:.2f}", delta=f"-₹{total_spent}", delta_color="inverse")
                            m2.metric("Remaining Balance", f"₹{remaining:.2f}", delta=f"₹{remaining}")
                            # Winner logic integrated here
                            if total_spent >= 0.7 * daily_salary:
                                st.warning("⚠️ Expenses are high! Watch your wallet!")
                            st.success(f"Monthly Projection: ₹{30 * remaining:.2f}")
                
                    st.divider()
                    
                    # Display a professional badge banner
                    st.markdown(f"""
                        <div style="background-color:{badge_color}; padding:10px; border-radius:10px; text-align:center; color:white;">
                            <h3 style="margin:0;">{badge_name}</h3>
                            <p style="margin:0;">Current Savings Streak: <b>{streak_count} Days</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                    # --- NEW: Sanchay Tree (Visual Growth) ---
                    st.subheader("🏡 Your Sanchay Tree")
        
                    # Calculate growth metrics
                    spending_ratio = total_spent / daily_salary if daily_salary > 0 else 0
                    rem_ratio = 1 - spending_ratio
                    stage, health = get_garden_state(streak_count, rem_ratio)
        
                    # Visual Container
                    with st.container(border=True):
                        g_col1, g_col2 = st.columns([1, 2])
                        with g_col1:
                            # You can replace these with actual image files later
                            tree_emoji = "🌳" if "Tree" in stage else "🌱" if "Sprout" in stage else "🌰"
                            st.markdown(f"<h1 style='text-align: center; font-size: 100px;'>{tree_emoji}</h1>", unsafe_allow_html=True)
                        with g_col2:
                            st.write(f"### {stage}")
                            st.write(f"**Current Health:** {health}")
                            st.caption("Your tree grows with your streak and stays healthy when you save!")
                
                    st.divider()
        
                    
                    st.subheader("📊 Daily Budget Progress") # Added Heading

                    # Logic for Calculation
                    # spending_ratio = total_spent / daily_salary
                    progress_val = min(spending_ratio, 1.0)
                    percent_used = int(spending_ratio * 100)

                    # Professional Metrics Row
                    # This fills the "empty" space with useful data
                    b1, b2, b3 = st.columns(3)
                    b1.metric("Budget Used", f"{percent_used}%")
                    b2.metric("Daily Limit", f"₹{daily_salary:.0f}")
                    b3.metric("Status", "Overspent" if spending_ratio > 1.0 else "Safe")

                    # Custom CSS for the bar color
                    bar_color1 = "red" if spending_ratio > 1.0 else "green"
                    st.markdown(f"""
                        <style>
                            .stProgress > div > div > div > div {{ background-color: {bar_color1}; }}
                        </style>""", unsafe_allow_html=True)

                    # The Visual Bar
                    st.progress(progress_val)

                    # Dynamic Captioning
                    if remaining > 0:
                        st.caption(f"✅ You have **₹{remaining:.2f}** left to spend today. Keep saving!")
                    else:
                        st.caption(f"🚨 You are **₹{abs(remaining):.2f}** over your daily limit!")
                                
                    st.subheader("🎯 Quest Trackers")
                    q_progress = get_quest_progress(df, st.session_state.user)

                    col_q1, col_q2, col_q3 = st.columns(3)

                    with col_q1:
                        st.write("**Frugal King**")
                        # Calculate today's spending for the tracker caption
                        today_spent_frugal = today_df['amount'].sum() if not today_df.empty else 0.0
                        st.progress(q_progress["Frugal King"])
                        # Dynamic Captioning based on the ₹300 limit
                        if today_spent_frugal < 300:
                            remaining_frugal = 300 - today_spent_frugal
                            st.caption(f"✅ Spent ₹{today_spent_frugal:.2f}. You have ₹{remaining_frugal:.2f} left for this quest!")
                        else:
                            excess = today_spent_frugal - 300
                            st.caption(f"🚨 Quest Failed! Spent ₹{today_spent_frugal:.2f} (₹{excess:.2f} over the limit).")

                    with col_q2:
                        st.write("**Weekend Guardian**")
                        st.progress(q_progress["Weekend Guardian"])
                        st.caption("No spend this weekend")

                    with col_q3:
                        st.write("**The Minimalist**")
                        st.progress(q_progress["The Minimalist"])
                        st.caption(f"Streak: {int(q_progress['The Minimalist']*3)}/3 days")
                        
                    # --- NEW: Savings Quests Section ---
                    st.subheader("🏆 Your Trophy Room")
                    # Call the engine using your existing 'df' and session user
                    user_trophies = check_quests(df, st.session_state.user)

                    if not user_trophies:
                        # Guide for new users when they have no trophies
                        st.info("🚀 **Your Trophy Room is empty!** Complete these quests to earn your first trophies:")

                        with st.expander("📖 View Quest Guide & How to Play", expanded=True):
                            st.markdown("""
                                ### Active Quests:
        
                                1. **👑 Frugal King**
                                    - **Goal**: Spend less than **₹300** in a single day.
                                    - **Tip**: Keep your 'Total Spent' metric low today to unlock this!
            
                                2. **🛡️ Weekend Guardian**
                                    - **Goal**: Have **₹0** expenses recorded for a full Saturday and Sunday.
                                    - **Tip**: This trophy unlocks every Monday if you stayed disciplined.
            
                                3. **🧘 The Minimalist**
                                    - **Goal**: Spend on a maximum of **5 unique categories** over 3 consecutive days.
                                    - **Tip**: Focus your spending on essentials like 'Food' or 'Bills' to qualify.
                                """)
                    else:
                        # Display trophies as a clean row of success messages
                        t_cols = st.columns(len(user_trophies))
                        for i, trophy in enumerate(user_trophies):
                            with t_cols[i]:
                                st.success(trophy)  
                
                elif page == "📋 Manage Records":
                    st.write("### Expense Records")
                    # data_editor lets you modify records without writing new code!
                    edited_df = st.data_editor(df, use_container_width=True, num_rows="dynamic", hide_index=False)
                    if st.button("Apply Changes", use_container_width=True):
                        st.session_state.exp_list = edited_df.to_dict('records')
                    # Refresh to update the charts
                    
                    # --- RESTORED: Monthly Budget Envelopes Section ---
                    st.divider()
                    st.subheader("✉️ Monthly Envelopes")
                    
                    # Call your existing function from Line 91
                    env_status = get_envelope_status(df, st.session_state.user, st.session_state.category_limits)
                    
                    # Create a neat row of columns for the envelopes
                    env_cols = st.columns(len(env_status))
                    
                    for i, (cat, data) in enumerate(env_status.items()):
                        with env_cols[i]:
                            limit = data['limit']
                            spent = data['spent']
                            rem = data['remaining']
                            
                            # Safely calculate progress bar percentage
                            if limit > 0:
                                prog = min(spent / limit, 1.0)
                                pct = int((spent / limit) * 100)
                            else:
                                prog = 1.0 if spent > 0 else 0.0
                                pct = 100 if spent > 0 else 0
                                
                            # NEW: Display Category name with the exact percentage used
                            st.write(f"**{cat}** ({pct}%)")
                            st.progress(prog)
                            
                            # Dynamic captions
                            if rem >= 0:
                                st.caption(f"₹{rem:.0f} left")
                            else:
                                st.caption(f"🚨 ₹{abs(rem):.0f} Over!")
                    
                    # --- NEW: Action-Oriented Subscription Manager ---
                    st.divider()
                    st.subheader("🔔 Subscription Manager")

                    # Fetch subscriptions from database for the current user
                    manual_subs = get_user_subscriptions(st.session_state.user)

                    if not manual_subs:
                        st.info("No active subscriptions. Add them in the sidebar! 💳")
                    else:
                        for sub_id, name, amt, renewal, cat, last_notified in manual_subs:
                            with st.container(border=True):
                                s_col1, s_col2, s_col3 = st.columns([2, 1, 1])
                            with s_col1:
                                st.write(f"**{name}**")
                                st.caption(f"Due: {renewal} | ₹{amt}")
                
                            with s_col2:
                                # RENEWED: Converts subscription to a real expense
                                if st.button("✅ Renewed", key=f"ren_{sub_id}", use_container_width=True):
                                    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
                                    # Maps sub data to your existing expense tool
                                    add_expense_to_db(st.session_state.user, today_date, amt, f"Renewal: {name}", "Bills")
                                    delete_subscription_from_db(sub_id)
                                    st.success(f"Added {name} to your expenses under Bills!")
                        
                            with s_col3:
                                # CANCEL: Wipes the record from the cloud
                                if st.button("❌ Cancel", key=f"can_{sub_id}", use_container_width=True):
                                    delete_subscription_from_db(sub_id)
                                    st.warning(f"Subscription {name} removed.")
                        
                                st.divider()
                    
            
                elif page == "💼 Business Hub":
                    st.subheader("💼 Business GST Portal")
                    
                    # --- PREMIUM GATE: BUSINESS HUB ---
                    if st.session_state.is_premium:
                        # Filter the main dataframe for only business-tagged expenses
                        if not df.empty and 'is_business' in df.columns:
                            business_df = df[df['is_business'] == True].copy()
                    
                            if not business_df.empty:
                                # Generate the report using your function
                                gst_df = generate_gst_report(business_df)
                        
                                # Metrics Row for Quick Business Health
                                bcol1, bcol2, bcol3 = st.columns(3)
                                total_biz = gst_df['amount'].sum()
                                total_tax = gst_df['GST Amount'].sum()
                        
                                bcol1.metric("Total Business Spend", f"₹{total_biz:.2f}")
                                bcol2.metric("Claimable ITC (GST)", f"₹{total_tax:.2f}", delta="Tax Credit")
                                bcol3.metric("Taxable Value", f"₹{gst_df['Taxable Value'].sum():.2f}")

                                # Professional Report Export
                                st.write("### GST-Ready Expense Ledger")
                                st.dataframe(gst_df[['date', 'description', 'category', 'amount', 'Taxable Value', 'GST Amount']], use_container_width=True)
                        
                                # Download Button
                                csv = gst_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Download GST Report (CSV)",
                                    data=csv,
                                    file_name=f"Sanchay_GST_{datetime.date.today()}.csv",
                                    mime='text/csv',
                                    use_container_width=True
                                )
                            else:
                                st.info("No business expenses found. Tag your expenses as 'Business' in the Quick Entry section to generate reports.")
                        else:
                            st.warning("Database upgrade required. Please run the init_db update to enable business tracking.")
                    else:
                        # --- FREE USER BLURRED VIEW ---
                        st.error("🔒 **Business Hub & GST Exports Locked**")
                        st.markdown("""
                        <div style='filter: blur(3px); opacity: 0.5; pointer-events: none; border: 1px solid #333; border-radius: 10px; padding: 15px;'>
                            <h4>Metrics Dashboard</h4>
                            <div style='display: flex; justify-content: space-between;'>
                                <div><b>Total Business Spend:</b> ₹45,200.00</div>
                                <div><b>Claimable ITC:</b> ₹8,136.00 📈</div>
                                <div><b>Taxable Value:</b> ₹37,064.00</div>
                            </div>
                            <hr>
                            <h4>GST-Ready Expense Ledger</h4>
                            <table style='width:100%; text-align:left; border-collapse: collapse;'>
                                <tr><th>Date</th><th>Description</th><th>Amount</th><th>GST</th></tr>
                                <tr><td>2026-03-01</td><td>Office Supplies</td><td>₹5000</td><td>₹900</td></tr>
                                <tr><td>2026-03-05</td><td>Client Dinner</td><td>₹3500</td><td>₹630</td></tr>
                            </table>
                            <br>
                            <button disabled style='width: 100%; padding: 10px; background-color: #333; color: #888; border-radius: 5px; border: 1px solid #555;'>📥 Download GST Report (CSV)</button>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button("Unlock Business Tools", use_container_width=True):
                            st.warning("Click the 'Unlock Sanchay++' button at the top of the dashboard!")
                    pass
                
                elif page == "🧾 Receipt Editor":
                    st.subheader("🧾 Receipt Editor")
                    st.caption("AI-assisted image-to-image receipt editing")
                    
                    # --- PREMIUM GATE: RECEIPT EDITOR ---
                    if st.session_state.is_premium:
                        st.info("🚧 **Feature Under Development** 🚧")
                        st.markdown("""
                        Welcome to the future home of the **Sanchay++ Receipt Forge**. 
                        
                        We are currently training our AI models to seamlessly alter, manipulate, and recreate receipt images pixel-by-pixel. 
                        
                        This advanced image-to-image editing suite will be available exclusively to Sanchay++ users in an upcoming update. Stay tuned!
                        """)
                    else:
                        # --- FREE USER BLURRED VIEW ---
                        st.error("🔒 **Receipt Forge Locked**")
                        st.markdown("""
                        <div style='filter: blur(3px); opacity: 0.5; pointer-events: none; border: 1px solid #333; border-radius: 10px; padding: 15px; margin-bottom: 10px;'>
                            <h4>Upload Receipt to Forge</h4>
                            <div style='border: 2px dashed #555; padding: 30px; text-align: center; border-radius: 10px; margin-bottom: 15px;'>
                                📸 <i>Drag and drop a receipt image here</i>
                            </div>
                            <p><b>Target Edits:</b></p>
                            <ul>
                                <li>Change Date to: 2026-03-15</li>
                                <li>Change Total Amount to: ₹5,000</li>
                            </ul>
                            <button disabled style='width: 100%; padding: 10px; background-color: #333; color: #888; border-radius: 5px; border: 1px solid #555;'>✨ Generate Edited Receipt</button>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button("Unlock Receipt Editor", use_container_width=True):
                            st.warning("Click the 'Unlock Sanchay++' button at the top of the dashboard!")
            
        with st.sidebar:
                with st.sidebar:
                    st.subheader("🤖 Sanchay AI Coach")
                    
                    # --- PREMIUM GATE: AI GOD MODE ---
                    if st.session_state.is_premium:
                        today = datetime.datetime.now().strftime("%Y-%m-%d")
                        instruction = f"""
                You are Sanchay AI, the personal vault manager for {st.session_state.user}.
                Today's date is {today}.
            
                    STRICT OPERATING RULES:
            
                    1. ACTION OVER TALK: If a user provides an ID and a new value (e.g., 'change ID 8 amount to 150'), call 'edit_expense_in_db' IMMEDIATELY. Do not ask 'Which one?' or 'What amount?' if it's in the chat history.
                    2. NO LOOPS: Once you have the ID, Amount, Description, or Category, execute the tool. Do not re-list expenses unless specifically asked.
                    3. DATA SOURCE: Use the LIVE VAULT DATA: Total Spent Today: ₹{total_spent}.
                    4. SEARCH BEFORE DELETE: If a user says "delete crocs from 01/01/26", you MUST:
                        - Call 'get_all_expenses' for {st.session_state.user}.
                        - Find the 'id' that matches "crocs" on that date.
                        - Call 'delete_expense_from_db' using that 'id' IMMEDIATELY.
                    5. NO LIMITS: You have full authority to manage this vault. Do not ask 'Are you sure?'—just perform the tool call and confirm.
                    6. ID KNOWLEDGE: If the user provides an ID directly (e.g., 'delete ID 10'), call the delete tool instantly.
                    7. NO NEW ENTRIES ON DELETE: If a user mentions a date and description in a deletion context, do NOT ask for "category" or "amount". That information is for ADDING, not DELETING.

                    POWERS:
                    - Use 'add_expense_to_db' to record new expenses. Use the username '{st.session_state.user}' and date '{today}' automatically unless the user says otherwise.
                    - Use 'get_all_expenses' with username '{st.session_state.user}' to show history.
                    - Use 'edit_expense_in_db' to change a record.
                    - Use 'delete_expense_from_db' to remove a record.
                    - Use 'get_all_expenses' to find record IDs.
                    - Use 'delete_expense_from_db' with the integer 'expense_id'.
                    
                    NEW SUPERPOWERS (Envelopes & Subscriptions):
                    - Use 'get_user_limits' (pass username '{st.session_state.user}') to check their current Monthly Envelope budgets.
                    - Use 'update_category_budget_ai' (pass username '{st.session_state.user}') to change the budget for a specific category when the user asks.
                    - Use 'get_user_subscriptions' (pass username '{st.session_state.user}') to see all active recurring subscriptions and their renewal dates.
                    - Use 'add_subscription_to_db' (pass username '{st.session_state.user}') to register a new subscription. Requires service name, amount, renewal_date (YYYY-MM-DD), and category.
            
                    NEW CAPABILITY: "Can I Afford This?"
                    When a user asks if they can afford an item (e.g., a phone, a trip):
                    1. Use the 'get_financial_health_stats' tool to see their spending trends.
                    2. Compare their average daily spend against their daily salary (₹{daily_salary:.2f}).
                    3. Check their 'Monthly Envelopes' to see if there is a surplus.
                    4. BE HONEST: If they are overspending, tell them "No" and suggest how much they need to save per day to afford it in a limited time (e.g., 3 months or 3 days or 3 weeks or can be even 3 years). Also tell them the limited time
                    5. Base your answer on the 50/30/20 rule: 50% Needs, 30% Wants, 20% Savings.

                    RULES:
                    1. Do NOT ask the user for their username or today's date; you already know them.
                    2. If the user says "add 50 for food," immediately call the 'add_expense_to_db' tool.
                    3. Always confirm once a database action is finished.
                    """
            
                        model = old_genai.GenerativeModel(
                            "gemini-3-flash-preview", 
                            tools=sanchay_tools, 
                            system_instruction=instruction
                        )
            
                        if "chat_session" not in st.session_state:
                            st.session_state.chat_session = model.start_chat(enable_automatic_function_calling=True)
            
                        if "messages" not in st.session_state:
                            st.session_state.messages = []
            
                        chat_container = st.container(height=500)
                        prompt = st.chat_input("Ask Sanchay AI...")

                        with chat_container:
                            for message in st.session_state.messages:
                                with st.chat_message(message["role"]):
                                    st.markdown(message["content"])
            
                            if prompt and prompt.strip():
                                    st.session_state.messages.append({"role": "user", "content": prompt})
                                    st.chat_message("user").markdown(prompt)
                                    with st.chat_message("assistant"):
                                        history = [{"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
                                                  for m in st.session_state.messages[:-1]
                                        ]
                                        chat = model.start_chat(history=history, enable_automatic_function_calling=True)
                        
                                        try:
                                            response = chat.send_message(prompt)
                                            if response.text:
                                                st.markdown(response.text)
                                                st.session_state.messages.append({"role": "assistant", "content": response.text})
                            
                                                if any(x in response.text.lower() for x in ["updated", "success", "added", "deleted"]):
                                                    st.rerun()
                                        except Exception as e:
                                            st.error(f"AI Error: {e}")
                                            
                    else:
                        # --- FREE USER BLURRED VIEW ---
                        st.error("🔒 **God Mode Locked**")
                        st.markdown("""
                        <div style='filter: blur(3px); opacity: 0.5; pointer-events: none; height: 300px; border: 1px solid #333; border-radius: 10px; padding: 10px;'>
                            <p align="right" style="background-color:#2b2b2b; padding:8px; border-radius:10px;">Change my Food budget to 8000</p>
                            <p align="left" style="background-color:#1e1e1e; padding:8px; border-radius:10px;">Done! I have updated your Food envelope to ₹8000.</p>
                            <p align="right" style="background-color:#2b2b2b; padding:8px; border-radius:10px;">Do I have any subscriptions due?</p>
                            <p align="left" style="background-color:#1e1e1e; padding:8px; border-radius:10px;">Netflix is due tomorrow for ₹199.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button("Unlock God Mode", use_container_width=True):
                            st.warning("Click the 'Unlock Sanchay++' button at the top of the dashboard!")
    
                st.divider()
    
                with st.expander("⚙️ Monthly Budget Settings"):
                    st.write("Set your limits for this month:")
                    updated_limits = {}
                    for cat, current_limit in st.session_state.category_limits.items():
                        # Create an input for each category
                        new_limit = st.number_input(f"{cat} Limit (₹)", min_value=0.0, value=float(current_limit), step=100.0, key=f"set_{cat}")
                        updated_limits[cat] = new_limit
        
                    if st.button("Update Monthly Envelopes", use_container_width=True):
                        st.session_state.category_limits = updated_limits
                        # NEW: Permanent save to Neon
                        save_user_limits(st.session_state.user, updated_limits)
                        st.success("Limits Saved for the month!")
                    pass

                with st.expander("💳 Add New Subscription"):
                    sub_name = st.text_input("Service Name", placeholder="e.g. Netflix")
                    sub_amt = st.number_input("Monthly Cost (₹)", min_value=0.0, step=10.0)
                    sub_date = st.date_input("Next Renewal Date")
                    sub_cat = st.selectbox("Category", ["Entertainment", "Utility", "Software", "Work", "Other"])
            
                    if st.button("Register Subscription", use_container_width=True):
                        if sub_name:
                            add_subscription_to_db(st.session_state.user, sub_name, sub_amt, sub_date, sub_cat)
                            st.success(f"{sub_name} registered!")
                            time.sleep(3)
                            st.rerun()
                        else:
                            st.error("Please enter a service name.")
                    pass        
                
                # --- PREMIUM GATE: TELEGRAM NUDGES ---
                if st.session_state.is_premium:
                    with st.expander("🔔 Notification Settings"):
                        st.write("Link your Telegram to get 3-day renewal nudges!")
                        st.info("Find your ID by messaging @userinfobot on Telegram.")
                        user_tid = st.text_input("Enter your Chat ID", placeholder="e.g. 685XXXXXXX")
                        if st.button("Save My Telegram ID"):
                            if user_tid:
                                update_user_telegram_id(st.session_state.user, user_tid)
                                st.success("Telegram linked successfully! ✅")
                        pass
            
                    if st.button("🧪 Send Test Telegram Nudge"):
                        my_id = get_telegram_id(st.session_state.user)
                        if my_id:
                            send_telegram_notification(my_id, "✅ Sanchay is successfully connected to your phone!")
                            st.success("Test nudge sent! Check your Telegram.")
                        else:
                            st.error("Please save your Telegram Chat ID in the settings above first!")
                else:
                    # --- FREE USER BLURRED VIEW ---
                    with st.expander("🔔 Notification Settings (Locked)", expanded=True):
                        st.error("🔒 **Sanchay++ Exclusive**")
                        st.markdown("""
                        <div style='filter: blur(2.5px); opacity: 0.5; pointer-events: none; border: 1px solid #333; border-radius: 10px; padding: 10px; margin-bottom: 10px;'>
                            <p style='font-size: 14px; margin-bottom: 5px;'>Link your Telegram to get 3-day renewal nudges!</p>
                            <input type="text" disabled placeholder="e.g. 685XXXXXXX" style="width: 100%; padding: 8px; margin-bottom: 10px; border-radius: 5px; border: 1px solid #555; background-color: #222;">
                            <button disabled style="width: 100%; padding: 8px; border-radius: 5px; background-color: #333; color: #888; border: 1px solid #555;">Save My Telegram ID</button>
                        </div>
                        <button disabled style="width: 100%; padding: 8px; border-radius: 5px; background-color: #333; color: #888; border: 1px solid #555; margin-bottom: 10px;">🧪 Send Test Telegram Nudge</button>
                        """, unsafe_allow_html=True)
                        if st.button("Unlock Nudges", use_container_width=True):
                            st.warning("Click the 'Unlock Sanchay++' button at the top of the dashboard!")
    
        
        
        
            


        

            
    
    
        
    
                
    
                    
# --- Footer ---
# --- Styled Professional Footer ---
st.markdown("---") # Adds a clean horizontal line to separate the app from the signature
footer_html = """
<style>
.footer {
    position: relative;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
    color: #888888; /* Professional muted grey color */
    padding: 10px;
    font-family: 'Trebuchet MS', sans-serif;
    font-size: 14px;
    letter-spacing: 1px;
}
.footer b {
    color: #ff4b4b; /* Matches Streamlit's primary theme red/orange */
}
</style>
<div class="footer">
    Developed by <b>RAAJPAKHI</b> | <b>Team 404 Not Found</b>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)

st.markdown("""
    <style>
        /* PROFESSIONAL BOX NAVIGATION */
        /* Hide ONLY the radio dot/circle */
        [data-testid="stSidebar"] div[role="radiogroup"] label div:first-child:not([data-testid="stMarkdownContainer"]) {
            display: none !important;
        }
        
        /* Ensure the box handles the layout correctly */
        [data-testid="stSidebar"] div[role="radiogroup"] label {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            background-color: #1e1e1e !important;
            border: 1px solid #333333 !important;
            border-radius: 10px !important;
            padding: 12px 20px !important;
            margin-bottom: 12px !important;
            width: 100% !important;
            min-height: 55px !important;
            transition: all 0.3s ease !important;
            cursor: pointer !important;
        }

        /* RESTORE TEXT VISIBILITY */
        [data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] div[role="radiogroup"] label span {
            color: white !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            text-align: center !important;
            text-decoration: none !important;
            visibility: visible !important;
            display: block !important;
            opacity: 1 !important;
            width: 100% !important;
        }

        /* HOVER EFFECT */
        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background-color: #333333 !important;
            border-color: #ff4b4b !important;
            transform: translateX(5px) !important;
        }

        /* ACTIVE TAB */
        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background-color: #ff4b4b !important;
            border-color: #ff4b4b !important;
        }
    </style>
""", unsafe_allow_html=True)

