import requests
import json
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

class MistralAI:
    def __init__(self):
        self.api_key = os.getenv('MISTRAL_API_KEY')
        self.base_url = "https://api.mistral.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_explanation(self, prompt):
        """Generate explanation using Mistral AI"""
        if not self.api_key:
            return self._get_fallback_response()
        
        try:
            payload = {
                "model": "mistral-small-latest",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 1500
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                print(f"Mistral API Error: {response.status_code} - {response.text}")
                return self._get_fallback_response()
                
        except Exception as e:
            print(f"Mistral API call failed: {e}")
            return self._get_fallback_response()
    
    def _get_fallback_response(self):
        """Fallback response when API fails"""
        return "I apologize, but I'm currently unable to generate a detailed analysis. Please check your API configuration."

class ExplanationGenerator:
    def __init__(self):
        self.mistral = MistralAI()
    
    def generate_comprehensive_explanation(self, prediction, probability, features):
        """Generate comprehensive explanation using Mistral AI"""
        
        # Prepare feature analysis
        feature_analysis = self._analyze_features(features)
        risk_level = self._get_risk_level(probability)
        
        prompt = self._build_prompt(prediction, probability, risk_level, features, feature_analysis)
        
        explanation = self.mistral.generate_explanation(prompt)
        
        return explanation
    
    def _analyze_features(self, features):
        """Analyze features for the prompt"""
        analysis = {
            'critical_risks': [],
            'moderate_risks': [],
            'positive_factors': [],
            'neutral_factors': []
        }
        
        # Analyze each feature
        if features.get('OverTime') == 'Yes':
            analysis['critical_risks'].append("Frequent overtime indicating potential burnout")
        
        if features.get('JobSatisfaction', 3) <= 2:
            analysis['critical_risks'].append(f"Low job satisfaction (score: {features['JobSatisfaction']}/4)")
        elif features.get('JobSatisfaction', 3) >= 4:
            analysis['positive_factors'].append(f"High job satisfaction (score: {features['JobSatisfaction']}/4)")
        
        if features.get('WorkLifeBalance', 3) <= 2:
            analysis['critical_risks'].append(f"Poor work-life balance (score: {features['WorkLifeBalance']}/4)")
        elif features.get('WorkLifeBalance', 3) >= 4:
            analysis['positive_factors'].append(f"Excellent work-life balance (score: {features['WorkLifeBalance']}/4)")
        
        if features.get('MonthlyIncome', 65000) < 45000:
            analysis['critical_risks'].append(f"Below-average income (${features['MonthlyIncome']:,.0f})")
        elif features.get('MonthlyIncome', 65000) > 80000:
            analysis['positive_factors'].append(f"Competitive income (${features['MonthlyIncome']:,.0f})")
        
        if features.get('StockOptionLevel', 1) == 0:
            analysis['moderate_risks'].append("No stock options reducing financial incentives")
        elif features.get('StockOptionLevel', 1) >= 2:
            analysis['positive_factors'].append("Good stock option level providing retention incentive")
        
        if features.get('YearsAtCompany', 4) < 1:
            analysis['moderate_risks'].append(f"Short tenure ({features['YearsAtCompany']} years) indicating early-stage risk")
        elif features.get('YearsAtCompany', 4) > 5:
            analysis['positive_factors'].append(f"Long tenure ({features['YearsAtCompany']} years) indicating stability")
        
        return analysis
    
    def _get_risk_level(self, probability):
        """Determine risk level based on probability"""
        if probability >= 0.7:
            return "CRITICAL"
        elif probability >= 0.5:
            return "HIGH"
        elif probability >= 0.3:
            return "MODERATE"
        else:
            return "LOW"
    
    def _build_prompt(self, prediction, probability, risk_level, features, feature_analysis):
        """Build comprehensive prompt for Mistral AI"""
        
        feature_details = "\n".join([f"- {k}: {v}" for k, v in features.items()])
        
        prompt = f"""
        As an expert HR analytics consultant with 15+ years of experience in employee retention and talent management, provide a comprehensive analysis of this employee attrition prediction.

        PREDICTION OVERVIEW:
        - Prediction: {prediction}
        - Probability: {probability:.1%}
        - Risk Level: {risk_level}

        EMPLOYEE PROFILE DETAILS:
        {feature_details}

        FEATURE ANALYSIS:
        Critical Risk Factors: {', '.join(feature_analysis['critical_risks'])}
        Moderate Risk Factors: {', '.join(feature_analysis['moderate_risks'])}
        Positive Factors: {', '.join(feature_analysis['positive_factors'])}
        Neutral Factors: {', '.join(feature_analysis['neutral_factors'])}

        Please provide a detailed, structured analysis with the following sections:

        1. PREDICTION INTERPRETATION:
           - Explain in detail why the model predicts {prediction} with {probability:.1%} probability
           - Connect specific employee features to industry-standard attrition patterns
           - Discuss how these factors typically influence employee retention decisions

        2. ROOT CAUSE ANALYSIS:
           - Identify the primary drivers behind this prediction
           - Explain the psychological and organizational factors at play
           - Reference relevant HR research and industry benchmarks

        3. IMMEDIATE HR ACTIONS (Next 7 days):
           - Specific, actionable steps for HR and management
           - Conversation starters for one-on-one meetings
           - Immediate interventions to prevent attrition

        4. STRATEGIC RETENTION PLAN (Next 30-90 days):
           - Medium-term strategies tailored to this employee's profile
           - Development opportunities and career path suggestions
           - Compensation and benefits considerations

        5. MONITORING AND FOLLOW-UP:
           - Key metrics to track in the coming weeks
           - Early warning signs to watch for
           - Timeline for follow-up assessments

        6. RISK MITIGATION STRATEGIES:
           - Proactive measures to reduce attrition risk
           - Team and organizational-level interventions
           - Success metrics for retention efforts

        Please provide specific, data-driven recommendations based on the employee's unique profile. Use bullet points for clarity and focus on actionable insights that HR professionals can implement immediately.

        Format the response with clear section headings and make it professional yet accessible.
        """

        return prompt

def calculate_feature_impact(features):
    """Calculate impact of each feature for visualization"""
    impacts = {}
    
    # Overtime impact
    if features.get('OverTime') == 'Yes':
        impacts['Overtime'] = {'score': 8, 'direction': 'negative', 'reason': 'Frequent overtime increases burnout risk'}
    else:
        impacts['Overtime'] = {'score': -5, 'direction': 'positive', 'reason': 'Regular hours support work-life balance'}
    
    # Job Satisfaction impact
    js_score = features.get('JobSatisfaction', 3)
    if js_score <= 2:
        impacts['Job Satisfaction'] = {'score': 9, 'direction': 'negative', 'reason': f'Low satisfaction (score: {js_score}/4) indicates disengagement'}
    elif js_score >= 4:
        impacts['Job Satisfaction'] = {'score': -7, 'direction': 'positive', 'reason': f'High satisfaction (score: {js_score}/4) promotes retention'}
    else:
        impacts['Job Satisfaction'] = {'score': 0, 'direction': 'neutral', 'reason': f'Moderate satisfaction level'}
    
    # Work-Life Balance impact
    wlb_score = features.get('WorkLifeBalance', 3)
    if wlb_score <= 2:
        impacts['Work-Life Balance'] = {'score': 8, 'direction': 'negative', 'reason': f'Poor balance (score: {wlb_score}/4) affects well-being'}
    elif wlb_score >= 4:
        impacts['Work-Life Balance'] = {'score': -6, 'direction': 'positive', 'reason': f'Excellent balance (score: {wlb_score}/4) supports retention'}
    else:
        impacts['Work-Life Balance'] = {'score': 0, 'direction': 'neutral', 'reason': f'Adequate balance level'}
    
    # Income impact
    income = features.get('MonthlyIncome', 65000)
    if income < 45000:
        impacts['Monthly Income'] = {'score': 7, 'direction': 'negative', 'reason': f'Below-market compensation (${income:,.0f})'}
    elif income > 80000:
        impacts['Monthly Income'] = {'score': -5, 'direction': 'positive', 'reason': f'Competitive compensation (${income:,.0f})'}
    else:
        impacts['Monthly Income'] = {'score': 0, 'direction': 'neutral', 'reason': f'Market-rate compensation'}
    
    # Stock Options impact
    stock_level = features.get('StockOptionLevel', 1)
    if stock_level == 0:
        impacts['Stock Options'] = {'score': 4, 'direction': 'negative', 'reason': 'No equity reduces financial incentives'}
    elif stock_level >= 2:
        impacts['Stock Options'] = {'score': -3, 'direction': 'positive', 'reason': 'Good equity package provides retention incentive'}
    else:
        impacts['Stock Options'] = {'score': 0, 'direction': 'neutral', 'reason': 'Standard equity level'}
    
    # Tenure impact
    tenure = features.get('YearsAtCompany', 4)
    if tenure < 1:
        impacts['Company Tenure'] = {'score': 5, 'direction': 'negative', 'reason': f'Short tenure ({tenure} years) indicates early-stage risk'}
    elif tenure > 5:
        impacts['Company Tenure'] = {'score': -4, 'direction': 'positive', 'reason': f'Long tenure ({tenure} years) indicates stability'}
    else:
        impacts['Company Tenure'] = {'score': 0, 'direction': 'neutral', 'reason': f'Standard tenure period'}
    
    return impacts