import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import time

# Add utils to path
sys.path.append(os.path.dirname(__file__))

from utils import ExplanationGenerator, calculate_feature_impact
from model_training import AttritionModel

# Page configuration
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        padding: 25px;
        border-radius: 15px;
        color: black;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-low {
        background: linear-gradient(135deg, #51cf66, #40c057);
        padding: 25px;
        border-radius: 15px;
        color: black;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .feature-card {
        background-color: #f8f9fa;
        color: black;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .explanation-box {
        background-color: blue;
        color: white;
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        white-space: pre-wrap;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .risk-factor {
        color: #e74c3c;
        font-weight: bold;
    }
    .protective-factor {
        color: #27ae60;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class HRAttritionApp:
    def __init__(self):
        self.model = None
        self.explainer = ExplanationGenerator()
        self.load_model()
    
    def load_model(self):
        """Load or train the model"""
        try:
            self.model = AttritionModel()
            with st.spinner("🔄 Training AI model with synthetic data... This may take a few seconds"):
                self.model.train()
            st.success("✅ Model trained successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
    
    def render_sidebar(self):
        """Render the input sidebar"""
        st.sidebar.title("🏢 Employee Details")
        st.sidebar.markdown("Enter employee information to predict attrition risk")
        
        features = {}
        
        with st.sidebar.form("employee_form"):
            st.subheader("💼 Work Details")
            
            features['OverTime'] = st.selectbox(
                "Overtime Status",
                ['No', 'Yes'],
                help="Does the employee frequently work overtime?"
            )
            
            features['MonthlyIncome'] = st.slider(
                "Monthly Income ($)",
                min_value=20000,
                max_value=150000,
                value=65000,
                step=1000,
                help="Employee's monthly salary"
            )
            
            features['StockOptionLevel'] = st.slider(
                "Stock Option Level",
                min_value=0,
                max_value=3,
                value=1,
                help="Level of stock options (0=None, 3=Highest)"
            )
            
            st.subheader("😊 Job Satisfaction & Balance")
            
            features['JobSatisfaction'] = st.slider(
                "Job Satisfaction",
                min_value=1,
                max_value=4,
                value=3,
                help="1=Very Low, 2=Low, 3=High, 4=Very High"
            )
            
            features['WorkLifeBalance'] = st.slider(
                "Work-Life Balance",
                min_value=1,
                max_value=4,
                value=3,
                help="1=Very Poor, 2=Poor, 3=Good, 4=Very Good"
            )
            
            features['YearsAtCompany'] = st.slider(
                "Years at Company",
                min_value=0,
                max_value=25,
                value=4,
                help="Number of years with current company"
            )
            
            submitted = st.form_submit_button("🚀 Predict Attrition Risk")
        
        return features, submitted
    
    def render_prediction_result(self, probability, features):
        """Render the prediction result"""
        
        prediction = "High Risk" if probability > 0.5 else "Low Risk"
        risk_color = "#ff4444" if prediction == "High Risk" else "#44ff44"
        
        # Main prediction card
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div class="prediction-{'high' if prediction == 'High Risk' else 'low'}">
                <h2>Prediction: {prediction}</h2>
                <h3>Probability: {probability:.1%}</h3>
                <p>Risk Level: {'🚨 HIGH' if probability > 0.7 else '⚠️ MEDIUM' if probability > 0.5 else '✅ LOW'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Attrition Risk Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature impact analysis
            st.subheader("🔍 Feature Impact Analysis")
            impacts = calculate_feature_impact(features)
            
            for feature, impact in impacts.items():
                emoji = "🔴" if impact['direction'] == 'negative' else "🟢" if impact['direction'] == 'positive' else "🟡"
                impact_class = "risk-factor" if impact['direction'] == 'negative' else "protective-factor" if impact['direction'] == 'positive' else ""
                
                st.markdown(f"""
                <div class="feature-card">
                    <strong>{emoji} {feature}</strong><br>
                    <span class="{impact_class}">
                    Impact: {impact['direction'].title()}</span><br>
                    <small>{impact['reason']}</small>
                </div>
                """, unsafe_allow_html=True)
    
    def render_mistral_explanation(self, prediction, probability, features):
        """Render Mistral AI explanation"""
        st.markdown("---")
        st.subheader("🤖 Mistral AI Analysis & Recommendations")
        
        # Create a progress bar for better UX
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            progress_bar.progress(i + 1)
            status_text.text(f"Analyzing employee data... {i+1}%")
            time.sleep(0.01)
        
        status_text.text("🎯 Generating comprehensive analysis with Mistral AI...")
        
        try:
            explanation = self.explainer.generate_comprehensive_explanation(
                prediction, probability, features
            )
            
            progress_bar.empty()
            status_text.empty()
            
            st.markdown(f"""
            <div class="explanation-box">
                {explanation}
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error generating explanation: {e}")
            st.info("Please check your Mistral API key and try again.")
    
    def render_feature_analysis(self, features, probability):
        """Render feature analysis charts"""
        st.markdown("---")
        st.subheader("📊 Detailed Feature Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature impact chart
            impacts = calculate_feature_impact(features)
            impact_data = []
            
            for feature, impact in impacts.items():
                impact_data.append({
                    'Feature': feature,
                    'Impact Score': impact['score'],
                    'Direction': impact['direction'],
                    'Reason': impact['reason']
                })
            
            df_impact = pd.DataFrame(impact_data)
            fig = px.bar(df_impact, x='Impact Score', y='Feature', orientation='h',
                        title="Feature Impact on Attrition Risk",
                        color='Impact Score',
                        color_continuous_scale=['green', 'yellow', 'red'])
            fig.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk distribution pie chart
            risk_factors = []
            protective_factors = []
            
            impacts = calculate_feature_impact(features)
            for feature, impact in impacts.items():
                if impact['direction'] == 'negative':
                    risk_factors.append(feature)
                elif impact['direction'] == 'positive':
                    protective_factors.append(feature)
            
            labels = ['Risk Factors', 'Protective Factors', 'Neutral Factors']
            values = [len(risk_factors), len(protective_factors), 6 - len(risk_factors) - len(protective_factors)]
            colors = ['#e74c3c', '#27ae60', '#f39c12']
            
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, marker_colors=colors)])
            fig.update_layout(title_text="Risk Factor Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Main application runner"""
        st.markdown('<h1 class="main-header">🏢 HR Employee Attrition Predictor</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #666;'>
        AI-powered tool to predict employee attrition risk with Mistral AI explanations
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check for Mistral API key
        if not os.getenv('MISTRAL_API_KEY'):
            st.warning("""
            ⚠️ **Mistral API Key Required**
            
            To get AI-generated explanations, please set up your Mistral API key:
            
            1. Get a free API key from [Mistral AI](https://console.mistral.ai/)
            2. Create a `.env` file in your project directory
            3. Add: `MISTRAL_API_KEY=your_api_key_here`
            
            The app will still work with basic predictions, but explanations will be limited.
            """)
        
        # Get user inputs
        features, submitted = self.render_sidebar()
        
        if submitted:
            if self.model:
                try:
                    # Make prediction
                    probability = self.model.predict_proba(features)
                    prediction = "High Risk" if probability > 0.5 else "Low Risk"
                    
                    # Render results
                    self.render_prediction_result(probability, features)
                    self.render_mistral_explanation(prediction, probability, features)
                    self.render_feature_analysis(features, probability)
                    
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.info("Please try again with different input values.")
            else:
                st.error("Model not loaded. Please check the model files.")
        
        # Demo section when no prediction is made
        else:
            self.render_demo_section()
    
    def render_demo_section(self):
        """Render demo and information section"""
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 How It Works")
            st.info("""
            **1. Enter Employee Data**
            - Fill in the form with employee details
            - Include work history and satisfaction metrics
            - Click 'Predict Attrition Risk'
            
            **2. Get AI-Powered Analysis**
            - View attrition probability score
            - Understand feature impacts
            - Read Mistral AI-generated explanations
            
            **3. Take Action**
            - Get specific HR recommendations
            - Develop retention strategies
            - Monitor at-risk employees
            """)
        
        with col2:
            st.subheader("🤖 Mistral AI Integration")
            st.success("""
            **Dynamic Explanations:**
            - Real-time analysis using Mistral Large
            - Industry-specific insights
            - Actionable HR recommendations
            - Root cause analysis
            
            **Features Analyzed:**
            - Overtime patterns
            - Compensation levels  
            - Job satisfaction
            - Work-life balance
            - Company tenure
            - Stock options
            """)
        
        # Sample scenarios
        with st.expander("📋 Sample Employee Scenarios", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🚨 High Risk Scenario:**")
                st.code("""
                OverTime: Yes
                MonthlyIncome: $45,000
                StockOptionLevel: 0
                JobSatisfaction: 2
                WorkLifeBalance: 2
                YearsAtCompany: 1
                """)
                st.write("Expected: 75-90% attrition risk")
            
            with col2:
                st.write("**✅ Low Risk Scenario:**")
                st.code("""
                OverTime: No
                MonthlyIncome: $85,000
                StockOptionLevel: 2
                JobSatisfaction: 4
                WorkLifeBalance: 4
                YearsAtCompany: 8
                """)
                st.write("Expected: 10-25% attrition risk")

if __name__ == "__main__":
    app = HRAttritionApp()
    app.run()