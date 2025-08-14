import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import time

# --- MODERN THEME AND STYLING ---
st.set_page_config(
    page_title="🏠 Home Vision",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏠"
)

# Custom CSS for modern styling and animations
def load_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        .stApp {
            font-family: 'Inter', sans-serif;
        }

        /* Modern gradient background */
        .main-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }

        /* Animated cards */
        .metric-card {
            background: linear-gradient(145deg, #ffffff, #f0f0f0);
            border-radius: 20px;
            padding: 20px;
            margin: 10px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            border: 1px solid rgba(255,255,255,0.2);
        }

        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        }

        /* Animated buttons - Unified style for all buttons */
        .stButton>button {
            background: none;
            border: 2px solid #667eea;
            color: #667eea;
            border-radius: 15px;
            padding: 12px 24px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: none;
        }

        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: 2px solid transparent;
        }

        /* Glassmorphism effect */
        .glass-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 25px;
            margin: 15px 0;
        }

        /* Light-themed boxy card for About section */
        .light-card {
            background: #e9ecef; /* The requested slightly darker white color */
            border-radius: 15px;
            padding: 25px;
            margin: 15px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
        }

        .light-card:hover {
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }

        /* Loading animations */
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .pulse {
            animation: pulse 2s infinite;
        }

        @keyframes heartbeat {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        .heartbeat {
            animation: heartbeat 1.5s infinite;
            display: inline-block;
        }

        @keyframes slideInFromLeft {
            0% { transform: translateX(-100%); opacity: 0; }
            100% { transform: translateX(0); opacity: 1; }
        }
        .slide-in {
            animation: slideInFromLeft 0.7s ease-out forwards;
        }

        /* Modern input styling */
        .stNumberInput>div>div>input {
            border-radius: 12px;
            border: 2px solid #e0e0e0;
            transition: all 0.3s ease;
        }

        .stNumberInput>div>div>input:focus {
            border-color: #667eea;
            box-shadow: 0 0 10px rgba(102, 126, 234, 0.2);
        }

        /* Success/Info styling */
        .stSuccess {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            border-radius: 15px;
            padding: 15px;
            color: white;
            font-weight: 500;
        }

        .stInfo {
            background: linear-gradient(45deg, #2196F3, #21cbf3);
            border-radius: 15px;
            padding: 15px;
            color: white;
            font-weight: 500;
        }

        /* Left-aligned and animated main header */
        .main-header {
            color: black;
            font-size: 2.5em;
            font-weight: 700;
            text-align: left;
            margin: 20px 0;
            animation: slideInFromLeft 1s ease-out forwards;
        }

        .page-header {
            color: #667eea;
            font-weight: 600;
            font-size: 2em;
            text-align: left;
            margin-bottom: 20px;
        }

        /* Custom style for the home icon in the main title */
        .main-header .icon {
            display: inline-block;
            font-size: 1.2em;
            margin-right: 10px;
        }

        .price-card {
            background: linear-gradient(135deg, #A8EDEA 0%, #FED6E3 100%);
            border-radius: 20px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }

        .price-card:hover {
            transform: translateY(-5px);
        }
        </style>
    """, unsafe_allow_html=True)

load_css()

# Set default plotting style to whitegrid
sns.set_style("whitegrid", {"axes.facecolor": '#ffffff', "figure.facecolor": '#ffffff'})
plt.rcParams.update({'text.color': '#000000', 'axes.labelcolor': '#000000'})
plot_fg = '#000000' # A variable to be used for text color in plots

# --- CENTRALIZED PRICING LOGIC --- <<< ADDED
def calculate_price(sqft, bedrooms, bathrooms):
    """Calculates the house price based on its features using the new formula."""
    if sqft is None or bedrooms is None or bathrooms is None:
        return None
    
    # Define the components of the price
    base_price = 60000
    price_per_sqft = 20000  # <<< UPDATED: User's requested price per sqft
    price_per_bedroom = 25000
    price_per_bathroom = 20000
    
    # Calculate the final price
    price = base_price + (sqft * price_per_sqft) + (bedrooms * price_per_bedroom) + (bathrooms * price_per_bathroom)
    return price

# --- NAVIGATION ---
st.markdown('<h1 class="main-header"><span class="icon">🏠</span>Home Vision</h1>', unsafe_allow_html=True)

# Initialize section in session_state
if 'section' not in st.session_state:
    st.session_state['section'] = 'Home'
if 'show_compare' not in st.session_state:
    st.session_state['show_compare'] = False
if 'show_book_btn' not in st.session_state:
    st.session_state['show_book_btn'] = False
if 'booked_this_session' not in st.session_state:
    st.session_state['booked_this_session'] = False

nav = st.columns(3)
with nav[0]:
    if st.button("Home", key="nav_home"):
        st.session_state['section'] = 'Home'
        st.session_state['show_compare'] = False
        st.session_state['show_book_btn'] = False
with nav[1]:
    if st.button("Data Analysis", key="nav_data"):
        st.session_state['section'] = 'Data Analysis'
        st.session_state['show_compare'] = False
        st.session_state['show_book_btn'] = False
with nav[2]:
    if st.button("About Our App", key="nav_about"):
        st.session_state['section'] = 'About'
        st.session_state['show_compare'] = False
        st.session_state['show_book_btn'] = False

section = st.session_state['section']

# --- DUMMY DATA --- <<< UPDATED
np.random.seed(42)
sample_size = 200
sqft = np.random.randint(500, 4000, sample_size)
bedrooms = np.random.randint(1, 6, sample_size)
bathrooms = np.random.randint(1, 5, sample_size)

# Generate prices using the new centralized function for consistency
# We iterate to apply the function row-wise and add some noise for realism
prices = [calculate_price(s, b, ba) for s, b, ba in zip(sqft, bedrooms, bathrooms)]
price = np.array(prices) + np.random.normal(0, 500000, sample_size) # Increased noise for larger price scale

booked = np.random.choice([0, 1], size=sample_size, p=[0.7, 0.3])

df = pd.DataFrame({
    'sqft': sqft,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'price': price,
    'booked': booked
})

features = ['sqft', 'bedrooms', 'bathrooms']
X = df[features]
y = df['price']
lr = LinearRegression()
lr.fit(X, y)

# Calculate model performance metrics once, so they are available for all sections
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
y_pred = lr.predict(X)
r2 = r2_score(y, y_pred)
rmse = mean_squared_error(y, y_pred) ** 0.5
mae = mean_absolute_error(y, y_pred)
n_samples = X.shape[0]

# --- NAVIGATION LOGIC ---
if section == 'Home':
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="page-header">🏠 Predict Your Dream House Price</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("⬛ **Square Footage**")
        user_sqft = st.number_input("", min_value=500, max_value=4000, value=None, placeholder="Enter sqft", key="input_sqft1")
        st.markdown("*Typical range: 800-3500 sqft*")

    with col2:
        st.markdown("🛏️ **Bedrooms**")
        user_bedrooms = st.number_input("", min_value=1, max_value=6, value=None, placeholder="Enter bedrooms", key="input_bed1")
        st.markdown("*Most popular: 2-4 bedrooms*")

    with col3:
        st.markdown("🛁 **Bathrooms**")
        user_bathrooms = st.number_input("", min_value=1, max_value=5, value=None, placeholder="Enter bathrooms", key="input_bath1")
        st.markdown("*Modern standard: 2-3 bathrooms*")

    st.markdown('</div>', unsafe_allow_html=True)

    # Action buttons with modern styling
    st.markdown('<div style="margin: 30px 0;">', unsafe_allow_html=True)
    btn_row1_col1, btn_row1_col2 = st.columns(2)
    with btn_row1_col1:
        price_btn = st.button("💰 Get Current Price", key="price_btn_main")
    with btn_row1_col2:
        compare_btn_main = st.button("⚖️ Compare Properties", key="compare_btn_main2")
        if compare_btn_main:
            st.session_state['show_compare'] = True

    btn_row2_col1, btn_row2_col2 = st.columns(2)
    with btn_row2_col1:
        future_price_btn = st.button("📈 Future Price Forecast", key="future_price_btn_main")
    with btn_row2_col2:
        graph_btn = st.button("📊 3D Visualization", key="graph_btn_main")
    st.markdown('</div>', unsafe_allow_html=True)

    # Show only the output for the button pressed
    if price_btn:
        # <<< UPDATED: Use the centralized function
        price = calculate_price(user_sqft, user_bedrooms, user_bathrooms)
        if price is None:
            st.warning("⚠️ Please enter all house details to show price.")
        else:
            with st.spinner('🔄 Calculating your property value...'):
                time.sleep(1)  # Animation effect

                # Modern price display card
                st.markdown(f'''
                <div class="price-card">
                    <h2 style="color: #667eea; margin: 0;">💰 Current Market Value</h2>
                    <h1 style="color: #333; font-size: 3em; margin: 10px 0;">₹ {price:,.0f}</h1>
                    <p style="color: #666; margin: 0;">Based on current market trends and property features</p>
                </div>
                ''', unsafe_allow_html=True)
    if future_price_btn:
        # <<< UPDATED: Use the centralized function
        price = calculate_price(user_sqft, user_bedrooms, user_bathrooms)
        if price is None:
            st.warning("⚠️ Please enter all house details to show future price.")
        else:
            with st.spinner('🔮 Generating price forecast...'):
                time.sleep(1.5)  # Animation effect
                
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 📈 5-Year Price Forecast Analysis")
                st.markdown("*Predictions based on market trends, inflation, and property appreciation*")

                # Enhanced forecast with different scenarios using matplotlib
                years = np.array(range(2024, 2030))
                conservative_growth = np.array([price * (1.03) ** i for i in range(1, 7)])  # 3% conservative
                moderate_growth = np.array([price * (1.05) ** i for i in range(1, 7)])     # 5% moderate
                optimistic_growth = np.array([price * (1.07) ** i for i in range(1, 7)])    # 7% optimistic

                plt.style.use('ggplot')
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(years, conservative_growth, label='Conservative (3%)', color='#FF6B6B', marker='o')
                ax.plot(years, moderate_growth, label='Moderate (5%)', color='#4ECDC4', marker='o')
                ax.plot(years, optimistic_growth, label='Optimistic (7%)', color='#45B7D1', marker='o')
                ax.set_title("Property Value Forecast - Multiple Scenarios", color=plot_fg)
                ax.set_xlabel("Year", color=plot_fg)
                ax.set_ylabel("Predicted Value (₹)", color=plot_fg)
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

                # Summary table with all scenarios
                forecast_df = pd.DataFrame({
                    "Year": years,
                    "Conservative (3%)": [f"₹ {val:,.0f}" for val in conservative_growth],
                    "Moderate (5%)": [f"₹ {val:,.0f}" for val in moderate_growth],
                    "Optimistic (7%)": [f"₹ {val:,.0f}" for val in optimistic_growth]
                })

                st.markdown("#### 📊 Detailed Forecast Table")
                st.dataframe(forecast_df, use_container_width=True)

                # Key insights
                st.markdown("#### 🔍 Key Insights:")
                st.info(f"""
                **Current Value:** ₹ {price:,.0f}

                **5-Year Projections:**
                - Conservative: ₹ {conservative_growth[-1]:,.0f} (+{((conservative_growth[-1]/price - 1) * 100):.1f}%)
                - Moderate: ₹ {moderate_growth[-1]:,.0f} (+{((moderate_growth[-1]/price - 1) * 100):.1f}%)
                - Optimistic: ₹ {optimistic_growth[-1]:,.0f} (+{((optimistic_growth[-1]/price - 1) * 100):.1f}%)

                **Investment Potential:** Based on historical market data, real estate typically appreciates 4-6% annually in stable markets.
                """)

                st.markdown('</div>', unsafe_allow_html=True)
    if graph_btn:
        # <<< UPDATED: Use the centralized function
        user_price = calculate_price(user_sqft, user_bedrooms, user_bathrooms)
        if user_price is None:
            st.warning("⚠️ Please enter all house details to view the 3D visualization.")
        else:
            with st.spinner('🎨 Creating interactive 3D visualization...'):
                time.sleep(1)
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 📊 Interactive 3D Market Analysis")
                st.markdown("*Explore the relationship between property features and market prices*")

                # Create 3D scatter plot with Matplotlib
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')

                # Add scatter points for market data
                scatter = ax.scatter(df['sqft'], df['bedrooms'], df['price'], c=df['price'], cmap='viridis', label='Market Properties')

                # Add user's property as a highlighted point
                ax.scatter([user_sqft], [user_bedrooms], [user_price], color='red', s=100, label='Your Property', marker='D')

                ax.set_title("3D Property Market Analysis", color=plot_fg)
                ax.set_xlabel("Square Footage", color=plot_fg)
                ax.set_ylabel("Number of Bedrooms", color=plot_fg)
                ax.set_zlabel("Price (₹)", color=plot_fg)
                ax.legend()

                st.pyplot(fig)

                # Market position analysis
                similar_properties = df[
                    (df['sqft'].between(user_sqft - 200, user_sqft + 200)) &
                    (df['bedrooms'] == user_bedrooms)
                ]

                if len(similar_properties) > 0:
                    avg_similar_price = similar_properties['price'].mean()
                    price_comparison = ((user_price - avg_similar_price) / avg_similar_price) * 100

                    st.markdown("#### 🏡 Market Position Analysis")
                    if abs(price_comparison) < 5:
                        st.success(f"✔️ Your property is competitively priced! It's within {abs(price_comparison):.1f}% of similar properties.")
                    elif price_comparison > 0:
                        st.info(f"📈 Your property is priced {price_comparison:.1f}% above similar properties, indicating premium features.")
                    else:
                        st.info(f"📉 Your property is priced {abs(price_comparison):.1f}% below similar properties, indicating good value.")

                st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced Compare functionality UI
    if st.session_state.get('show_compare', False):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="page-header">⚖️ Advanced Property Comparison</h2>', unsafe_allow_html=True)
        st.markdown("*Compare two properties side-by-side with detailed analytics and insights*")

        col_label1, col_label2 = st.columns(2)
        with col_label1:
            st.markdown('<h3 style="color: #4ECDC4; text-align: left;">🏠 Property A (Your Selection)</h3>', unsafe_allow_html=True)
        with col_label2:
            st.markdown('<h3 style="color: #FF6B6B; text-align: left;">🏡 Property B (Comparison)</h3>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("⬛ **Square Footage (Property B)**")
            user_sqft2 = st.number_input("", min_value=500, max_value=4000, value=None, placeholder="Enter sqft", key="input_sqft2")

        with c2:
            st.markdown("🛏️ **Bedrooms (Property B)**")
            user_bedrooms2 = st.number_input("", min_value=1, max_value=6, value=None, placeholder="Enter bedrooms", key="input_bed2")

        with c3:
            st.markdown("🛁 **Bathrooms (Property B)**")
            user_bathrooms2 = st.number_input("", min_value=1, max_value=5, value=None, placeholder="Enter bathrooms", key="input_bath2")
        # Enhanced action buttons for comparison
        st.markdown('<div style="margin: 20px 0;">', unsafe_allow_html=True)
        btn_row1_col1_2, btn_row1_col2_2 = st.columns(2)
        with btn_row1_col1_2:
            compare_now = st.button("🔍 Compare Properties", key="compare_now_btn")
        with btn_row1_col2_2:
            cancel_compare = st.button("❌ Cancel Comparison", key="cancel_compare_btn")
        st.markdown('</div>', unsafe_allow_html=True)
        if cancel_compare:
            st.session_state['show_compare'] = False
            st.rerun()
        if compare_now:
            # <<< UPDATED: Use the centralized function for both properties
            price1 = calculate_price(user_sqft, user_bedrooms, user_bathrooms)
            price2 = calculate_price(user_sqft2, user_bedrooms2, user_bathrooms2)
            
            if price1 is None or price2 is None:
                st.warning("⚠️ Please enter all details for both properties to compare.")
            else:
                with st.spinner('🔍 Analyzing properties and generating comparison...'):
                    time.sleep(1.5)

                    # Side-by-side property cards
                    st.markdown("### 🏠 Property Comparison Dashboard")
                    col_prop1, col_prop2 = st.columns(2)

                    with col_prop1:
                        st.markdown(f'''
                        <div class="comparison-card" style="background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);">
                            <h2 style="text-align: left; color: white; margin: 0;">🏠 Property A</h2>
                            <div style="text-align: left; margin: 20px 0;">
                                <h1 style="color: white; font-size: 2.5em; margin: 0;">₹ {price1:,.0f}</h1>
                                <p style="color: white; opacity: 0.9;">Market Value</p>
                            </div>
                            <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; margin: 10px 0;">
                                <p style="color: white; margin: 5px 0;"><strong>⬛ Area:</strong> {user_sqft:,} sqft</p>
                                <p style="color: white; margin: 5px 0;"><strong>🛏️ Bedrooms:</strong> {user_bedrooms}</p>
                                <p style="color: white; margin: 5px 0;"><strong>🛁 Bathrooms:</strong> {user_bathrooms}</p>
                                <p style="color: white; margin: 5px 0;"><strong>💰 Price/sqft:</strong> ₹ {price1/user_sqft:.0f}</p>
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)

                    with col_prop2:
                        st.markdown(f'''
                        <div class="comparison-card" style="background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);">
                            <h2 style="text-align: left; color: white; margin: 0;">🏡 Property B</h2>
                            <div style="text-align: left; margin: 20px 0;">
                                <h1 style="color: white; font-size: 2.5em; margin: 0;">₹ {price2:,.0f}</h1>
                                <p style="color: white; opacity: 0.9;">Market Value</p>
                            </div>
                            <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; margin: 10px 0;">
                                <p style="color: white; margin: 5px 0;"><strong>⬛ Area:</strong> {user_sqft2:,} sqft</p>
                                <p style="color: white; margin: 5px 0;"><strong>🛏️ Bedrooms:</strong> {user_bedrooms2}</p>
                                <p style="color: white; margin: 5px 0;"><strong>🛁 Bathrooms:</strong> {user_bathrooms2}</p>
                                <p style="color: white; margin: 5px 0;"><strong>💰 Price/sqft:</strong> ₹ {price2/user_sqft2:.0f}</p>
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)

                    # Detailed comparison analysis
                    st.markdown("### 📊 Detailed Comparison Analysis")

                    # Advanced analytics and insights
                    diff = abs(price1 - price2)
                    better_value = "Property A" if (price1/user_sqft) < (price2/user_sqft2) else "Property B"
                    price_diff_percent = (diff / min(price1, price2)) * 100

                    # Investment analysis
                    st.markdown("### 💡 Investment Analysis & Recommendations")

                    insights_col1, insights_col2 = st.columns(2)

                    with insights_col1:
                        st.markdown("#### 📈 Financial Analysis")
                        if price1 > price2:
                            st.info(f"""
                            **Price Difference:** ₹ {diff:,.0f} ({price_diff_percent:.1f}% higher)

                            **Property A** is more expensive due to:
                            {f"• Larger area (+{user_sqft - user_sqft2:,} sqft)" if user_sqft > user_sqft2 else ""}
                            {f"• More bedrooms (+{user_bedrooms - user_bedrooms2})" if user_bedrooms > user_bedrooms2 else ""}
                            {f"• More bathrooms (+{user_bathrooms - user_bathrooms2})" if user_bathrooms > user_bathrooms2 else ""}

                            **Better Value:** {better_value} (₹ {min(price1/user_sqft, price2/user_sqft2):.0f}/sqft)
                            """)
                        elif price2 > price1:
                            st.info(f"""
                            **Price Difference:** ₹ {diff:,.0f} ({price_diff_percent:.1f}% higher)

                            **Property B** is more expensive due to:
                            {f"• Larger area (+{user_sqft2 - user_sqft:,} sqft)" if user_sqft2 > user_sqft else ""}
                            {f"• More bedrooms (+{user_bedrooms2 - user_bedrooms})" if user_bedrooms2 > user_bedrooms else ""}
                            {f"• More bathrooms (+{user_bathrooms2 - user_bathrooms})" if user_bathrooms2 > user_bathrooms else ""}

                            **Better Value:** {better_value} (₹ {min(price1/user_sqft, price2/user_sqft2):.0f}/sqft)
                            """)
                        else:
                            st.success("🎯 Both properties have identical pricing! Choose based on your preferences.")

                    with insights_col2:
                        st.markdown("#### 🏆 Expert Recommendations")

                        # Smart recommendations based on analysis
                        if (user_sqft > user_sqft2) and (price1/user_sqft <= price2/user_sqft2):
                            recommendation = "Property A"
                            reason = "Larger space at better value per sqft"
                        elif (user_sqft2 > user_sqft) and (price2/user_sqft2 <= price1/user_sqft):
                            recommendation = "Property B"
                            reason = "Larger space at better value per sqft"
                        elif price1 < price1:
                            recommendation = "Property A"
                            reason = "Lower overall cost"
                        elif price2 < price1:
                            recommendation = "Property B"
                            reason = "Lower overall cost"
                        else:
                            recommendation = "Either property"
                            reason = "Similar value proposition"

                        st.success(f"""
                        **🎯 Recommended Choice:** {recommendation}

                        **Reason:** {reason}

                        **Key Benefits:**
                        • {"✔️ More space for your money" if recommendation != "Either property" else "✔️ Equal value"}
                        • {"✔️ Better long-term investment" if recommendation != "Either property" else "✔️ Safe investment choice"}
                        • {"✔️ Higher resale potential" if recommendation != "Either property" else "✔️ Stable market value"}

                        **Next Steps:**
                        1. 📋 Schedule property viewing
                        2. 🏦 Arrange financing pre-approval
                        3. 🔍 Conduct professional inspection
                        """)

        st.markdown('</div>', unsafe_allow_html=True)

elif section == 'Data Analysis':
    st.markdown('<h1 class="page-header">📊 Advanced Market Analytics</h1>', unsafe_allow_html=True)

    # Modern dashboard with animated metrics
    total_properties = len(df)
    booked_properties = df['booked'].sum()
    remaining_properties = total_properties - booked_properties
    highest_price = df['price'].max()
    average_price = df['price'].mean()
    lowest_price = df['price'].min()
    average_area = df['sqft'].mean()

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🏘️ Property Portfolio Overview")

    # Create animated metric cards
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown(f'''
        <div class="metric-card">
            <h3 style="color: #667eea; margin: 0;">🏠 Total Properties</h3>
            <h1 style="color: #333; font-size: 2.5em; margin: 10px 0;">{total_properties}</h1>
            <p style="color: #666;">Active listings in database</p>
        </div>
        ''', unsafe_allow_html=True)

    with colB:
        st.markdown(f'''
        <div class="metric-card">
            <h3 style="color: #FF6B6B; margin: 0;">📋 Booked Properties</h3>
            <h1 style="color: #333; font-size: 2.5em; margin: 10px 0;">{booked_properties}</h1>
            <p style="color: #666;">{(booked_properties/total_properties*100):.1f}% occupancy rate</p>
        </div>
        ''', unsafe_allow_html=True)

    with colC:
        st.markdown(f'''
        <div class="metric-card">
            <h3 style="color: #4ECDC4; margin: 0;">🎯 Available Properties</h3>
            <h1 style="color: #333; font-size: 2.5em; margin: 10px 0;">{remaining_properties}</h1>
            <p style="color: #666;">Ready for booking</p>
        </div>
        ''', unsafe_allow_html=True)

    # Price analytics
    st.markdown("### 💰 Price Range Analysis")
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown(f'''
        <div class="metric-card">
            <h3 style="color: #45B7D1; margin: 0;">🔝 Highest Price</h3>
            <h1 style="color: #333; font-size: 2em; margin: 10px 0;">₹ {highest_price:,.0f}</h1>
            <p style="color: #666;">Premium property</p>
        </div>
        ''', unsafe_allow_html=True)

    with colB:
        st.markdown(f'''
        <div class="metric-card">
            <h3 style="color: #96CEB4; margin: 0;">📊 Average Price</h3>
            <h1 style="color: #333; font-size: 2em; margin: 10px 0;">₹ {average_price:,.0f}</h1>
            <p style="color: #666;">Market benchmark</p>
        </div>
        ''', unsafe_allow_html=True)

    with colC:
        st.markdown(f'''
        <div class="metric-card">
            <h3 style="color: #FFEAA7; margin: 0;">🔻 Lowest Price</h3>
            <h1 style="color: #333; font-size: 2em; margin: 10px 0;">₹ {lowest_price:,.0f}</h1>
            <p style="color: #666;">Entry-level option</p>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Statistical Analysis section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Statistical Deep Dive")

    # Buttons for graphs
    col_graph_buttons = st.columns(4)
    with col_graph_buttons[0]:
        if st.button("📈 View Price Distribution", key="btn_price_dist_toggle"):
            st.session_state['show_price_dist_graph'] = not st.session_state.get('show_price_dist_graph', False)
    with col_graph_buttons[1]:
        if st.button("📐 View Area Distribution", key="btn_area_dist_toggle"):
            st.session_state['show_area_dist_graph'] = not st.session_state.get('show_area_dist_graph', False)
    with col_graph_buttons[2]:
        if st.button("📋 View Booking Status", key="btn_booked_status_toggle"):
            st.session_state['show_booked_status_graph'] = not st.session_state.get('show_booked_status_graph', False)
    with col_graph_buttons[3]:
        if st.button("📊 View Market Trends", key="btn_market_trends_toggle"):
            st.session_state['show_market_trends_graph'] = not st.session_state.get('show_market_trends_graph', False)

    if st.session_state.get('show_price_dist_graph'):
        st.markdown("#### 📈 Price Distribution Analysis")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df['price'], bins=25, kde=True, ax=ax, color='#667eea')
        ax.set_title('Property Price Distribution')
        ax.set_xlabel('Price (₹)')
        ax.set_ylabel('Number of Properties')
        st.pyplot(fig)

        st.markdown(f"""
        **Price Statistics:**
        - Mean: ₹ {df['price'].mean():,.0f}
        - Median: ₹ {df['price'].median():,.0f}
        - Std Dev: ₹ {df['price'].std():,.0f}
        - Price Range: ₹ {df['price'].max() - df['price'].min():,.0f}
        """)

    if st.session_state.get('show_area_dist_graph'):
        st.markdown("#### 📐 Area Analytics")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df['sqft'], bins=25, kde=True, ax=ax, color='#4ECDC4')
        ax.set_title('Property Area Distribution')
        ax.set_xlabel('Area (Square Feet)')
        ax.set_ylabel('Number of Properties')
        st.pyplot(fig)

        st.markdown(f"""
        **Area Statistics:**
        - Mean: {df['sqft'].mean():,.0f} sqft
        - Median: {df['sqft'].median():,.0f} sqft
        - Std Dev: {df['sqft'].std():,.0f} sqft
        - Size Range: {df['sqft'].max() - df['sqft'].min():,.0f} sqft
        """)

    if st.session_state.get('show_booked_status_graph'):
        st.markdown("#### 📋 Property Booking Status Analysis")

        labels = ['Available Properties', 'Booked Properties']
        values = [remaining_properties, booked_properties]
        colors = ['#4ECDC4', '#FF6B6B']

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.set_title('Property Availability Status')
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)

        # Booking insights
        occupancy_rate = (booked_properties / total_properties) * 100
        st.info(f"""
        **📊 Booking Status Insights:**
        - **Occupancy Rate:** {occupancy_rate:.1f}%
        - **Available Properties:** {remaining_properties} ({((remaining_properties/total_properties)*100):.1f}%)
        - **Market Status:** {"🔥 High Demand" if occupancy_rate > 50 else "📈 Growing Market" if occupancy_rate > 30 else "🌱 Emerging Market"}
        - **Investment Opportunity:** {"Limited availability - Act fast!" if remaining_properties < 50 else "Good selection available"}
        """)

    if st.session_state.get('show_market_trends_graph'):
        st.markdown("#### 📊 Market Trends & Patterns")

        # Price vs Area scatter plot with seaborn
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='sqft', y='price', hue='bedrooms', data=df, palette='viridis', s=100, ax=ax)
        ax.set_title("Market Trends: Price vs Area by Bedroom Count")
        ax.set_xlabel("Area (Square Feet)")
        ax.set_ylabel("Price (₹)")
        ax.legend(title='Bedrooms')
        st.pyplot(fig)

        # Market trend insights
        price_per_sqft = df['price'] / df['sqft']
        avg_price_per_sqft = price_per_sqft.mean()

        st.info(f"""
        **📊 Market Trend Analysis:**
        - **Average Price per sqft:** ₹{avg_price_per_sqft:.0f}
        - **Premium Properties:** Above ₹{price_per_sqft.quantile(0.8):.0f} per sqft
        - **Value Properties:** Below ₹{price_per_sqft.quantile(0.2):.0f} per sqft
        - **Market Pattern:** {"Larger properties command premium pricing" if df['sqft'].corr(df['price']) > 0.7 else "Mixed pricing patterns"}
        """)

    st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced Bedroom and Bathroom Analysis (without graphs)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🏠 Bedroom & Bathroom Market Intelligence")

    st.markdown(f"""
    **🛏️ Bedroom Insights:**
    - **Average:** {df['bedrooms'].mean():.1f} bedrooms
    - **Most Popular:** {df['bedrooms'].mode().iloc[0]} bedrooms ({df['bedrooms'].value_counts(normalize=True).iloc[0]*100:.1f}% of properties)
    - **Range:** {df['bedrooms'].min()} - {df['bedrooms'].max()} bedrooms
    - **Market Trend:** {df['bedrooms'].mode().iloc[0]}-bedroom properties dominate the market

    💡 **Professional Insight:** Properties with 2-4 bedrooms offer the best investment potential due to high demand from families.
    """)

    st.markdown(f"""
    **🛁 Bathroom Insights:**
    - **Average:** {df['bathrooms'].mean():.1f} bathrooms
    - **Most Popular:** {df['bathrooms'].mode().iloc[0]} bathrooms ({df['bathrooms'].value_counts(normalize=True).iloc[0]*100:.1f}% of properties)
    - **Range:** {df['bathrooms'].min()} - {df['bathrooms'].max()} bathrooms
    - **Luxury Factor:** Properties with 3+ bathrooms are considered premium

    🏆 **Expert Analysis:** Modern buyers prefer a minimum of 2 bathrooms for convenience and resale value.
    """)

    # Bedroom-Bathroom correlation analysis
    st.markdown("#### 🔍 Bedroom-Bathroom Correlation Heatmap")

    # Create correlation heatmap with seaborn
    correlation_matrix = df[['bedrooms', 'bathrooms', 'sqft', 'price']].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Feature Correlation Analysis')
    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)
    # Enhanced Model Performance Section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🤖 Model Performance Dashboard")
    st.markdown("*Real-time insights into our machine learning model's accuracy and reliability*")

    # Performance metrics with animated cards
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

    with perf_col1:
        # R² Score with interpretation
        r2_percentage = r2 * 100
        color = "#4CAF50" if r2 > 0.8 else "#FF9800" if r2 > 0.6 else "#F44336"
        st.markdown(f'''
        <div class="metric-card">
            <h3 style="color: {color}; margin: 0;">📈 R² Score</h3>
            <h1 style="color: #333; font-size: 2.5em; margin: 10px 0;">{r2:.3f}</h1>
            <p style="color: #666;">{r2_percentage:.1f}% accuracy</p>
        </div>
        ''', unsafe_allow_html=True)

    with perf_col2:
        st.markdown(f'''
        <div class="metric-card">
            <h3 style="color: #2196F3; margin: 0;">📊 RMSE</h3>
            <h1 style="color: #333; font-size: 1.8em; margin: 10px 0;">₹ {rmse:,.0f}</h1>
            <p style="color: #666;">Prediction error</p>
        </div>
        ''', unsafe_allow_html=True)

    with perf_col3:
        st.markdown(f'''
        <div class="metric-card">
            <h3 style="color: #9C27B0; margin: 0;">🎯 MAE</h3>
            <h1 style="color: #333; font-size: 1.8em; margin: 10px 0;">₹ {mae:,.0f}</h1>
            <p style="color: #666;">Average error</p>
        </div>
        ''', unsafe_allow_html=True)

    with perf_col4:
        st.markdown(f'''
        <div class="metric-card">
            <h3 style="color: #FF5722; margin: 0;">🗂️ Data Points</h3>
            <h1 style="color: #333; font-size: 2.5em; margin: 10px 0;">{n_samples}</h1>
            <p style="color: #666;">Training samples</p>
        </div>
        ''', unsafe_allow_html=True)

    # Model insights
    st.markdown("### 🧠 Model Insights")

    accuracy_level = "Excellent" if r2 > 0.9 else "Very Good" if r2 > 0.8 else "Good" if r2 > 0.7 else "Fair"

    st.info(f"""
    **🤖 Model Performance Summary:**

    **Accuracy Level:** {accuracy_level} ({r2_percentage:.1f}%)

    **Key Metrics Explained:**
    - **R² Score ({r2:.3f}):** Our model explains {r2_percentage:.1f}% of price variations
    - **RMSE (₹{rmse:,.0f}):** Average prediction error magnitude
    - **MAE (₹{mae:,.0f}):** Typical deviation from actual prices
    - **Training Data:** {n_samples} properties analyzed

    **Reliability:** {"✔️ Highly Reliable" if r2 > 0.8 else "🟡 Moderately Reliable" if r2 > 0.6 else "🟠 Developing Accuracy"}

    **Model Type:** Linear Regression optimized for real estate valuation
    """)

    st.markdown('</div>', unsafe_allow_html=True)

elif section == 'About':
    st.markdown('<h1 class="main-header">ℹ️ About Our App</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="light-card">', unsafe_allow_html=True)
    st.markdown('<h2>🏠 Our Vision: Democratizing Real Estate Insights</h2>', unsafe_allow_html=True)
    st.markdown("""
    Our mission is to empower homebuyers, sellers, and investors with transparent, data-driven insights. We believe that informed decisions lead to better outcomes, and our <span class="heartbeat">tool</span> is designed to cut through the complexity of the real estate market, providing clear, actionable information at your fingertips. We are committed to making property valuation and market analysis accessible to everyone, not just industry professionals.
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="light-card">', unsafe_allow_html=True)
    st.markdown('<h2>⚙️ How It Works: The Technology Behind the Predictor</h2>', unsafe_allow_html=True)
    st.markdown("""
    Home Vision is built on a robust machine learning foundation. We use a **Linear Regression model**, a simple yet powerful algorithm, trained on a comprehensive synthetic dataset of real estate properties. This model learns the intricate relationships between property features like size, number of rooms, and price. When you enter your property's details, our model uses these learned relationships to provide a highly accurate and data-driven price prediction.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="light-card">', unsafe_allow_html=True)
    st.markdown('<h2>🌟 Our Model Performance</h2>', unsafe_allow_html=True)
    st.markdown(f"""
    Transparency is at the heart of our mission. We openly share our model's performance metrics so you can have complete confidence in our predictions.

    - **R² Score (Coefficient of Determination): {r2:.3f}**
      This score indicates how well our model's predictions fit the real data. An R² value of {r2:.3f} means our model explains approximately {r2*100:.1f}% of the variability in house prices. A score close to 1.0 is considered excellent.

    - **RMSE (Root Mean Squared Error): ₹{rmse:,.2f}**
      The RMSE value represents the square root of the average squared difference between our predicted prices and the actual prices in our dataset. In simple terms, it tells you the typical magnitude of our prediction error. A lower number indicates greater precision.

    - **MAE (Mean Absolute Error): ₹{mae:,.2f}**
      The MAE is the average absolute difference between predicted and actual prices. It provides a more intuitive understanding of the typical error margin. For example, an MAE of ₹{mae:,.0f} means our predictions are, on average, off by about this amount.

    These metrics collectively demonstrate the reliability and accuracy of our prediction model, providing a solid, data-backed foundation for your real estate decisions.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="light-card">', unsafe_allow_html=True)
    st.markdown('<h2>🤝 Our Commitment</h2>', unsafe_allow_html=True)
    st.markdown("""
    We are dedicated to improving this tool based on user feedback and expanding its capabilities. Whether you're a first-time homebuyer or an experienced investor, Home Vision is designed to be your trusted partner in the world of real estate.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="light-card">', unsafe_allow_html=True)
    st.markdown('<h2>📞 Get in Touch</h2>', unsafe_allow_html=True)
    st.markdown("""
    - **Email:** support@homevision.com
    - **Website:** [www.homevision.com](https://www.homevision.com)
    - **GitHub:** [github.com/Yuvaraj-ui132](https://github.com/Yuvaraj-ui132)
    - **Twitter:** [@HomeVisionApp](https://twitter.com/HomeVisionApp)
    """)
    st.markdown('</div>', unsafe_allow_html=True)