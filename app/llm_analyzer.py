import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import requests
import numpy as np

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AnalysisResult:
    """Structured analysis result"""
    executive_summary: str
    risk_level: RiskLevel
    key_findings: List[str]
    recommendations: List[str]
    temporal_patterns: Dict[str, Any]
    security_insights: str
    operational_recommendations: List[str]
    predicted_concerns: List[str]
    confidence_score: float

class OllamaClassroomAnalyzer:
    """Classroom monitoring data analyzer using Ollama LLM"""
    
    def __init__(self, model_name: str = "security_manager:latest", host: str = "http://192.168.14.10:11434"):
        """
        Initialize the analyzer
        
        Args:
            model_name: Ollama model to use (qwen2.5vl:3b or security_manager:latest)
            host: Ollama server host
        """
        self.model_name = model_name
        self.host = host
        self.api_url = f"{host}/api/generate"
        
        print(f"Initialized analyzer with model: {model_name}")
    
    def convert_to_json_serializable(self, obj):
        """Convert numpy/pandas types to JSON serializable types"""
        if isinstance(obj, (np.integer)):
            return int(obj)
        elif isinstance(obj, (np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    def prepare_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare structured data summary for LLM analysis"""
        
        # Basic statistics
        total_events = len(df)
        category_dist = df['category'].value_counts().to_dict()
        
        # Time analysis
        df = df.copy()
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['date'] = df['timestamp'].dt.date
        
        # Critical events analysis
        critical_events = df[df['category'].isin(['alert', 'human_intervention_needed'])]
        
        # Temporal patterns
        hourly_patterns = df.groupby('hour')['category'].count().to_dict()
        daily_patterns = df.groupby('day_of_week')['category'].count().to_dict()
        
        # Headcount analysis
        headcount_stats = {
            'mean': float(df['headcount'].mean()),
            'median': float(df['headcount'].median()),
            'max': int(df['headcount'].max()),
            'min': int(df['headcount'].min()),
            'std': float(df['headcount'].std())
        }
        
        # Alert frequency by time
        alert_by_hour = critical_events.groupby('hour').size().to_dict() if not critical_events.empty else {}
        
        # Recent trends (last 7 days)
        recent_date = df['timestamp'].max() - timedelta(days=7)
        recent_data = df[df['timestamp'] >= recent_date]
        recent_trends = recent_data.groupby('date')['category'].count().to_dict()
        
        # Create summary and ensure all values are JSON serializable
        summary = {
            'total_events': total_events,
            'date_range': {
                'start': df['timestamp'].min().strftime('%Y-%m-%d'),
                'end': df['timestamp'].max().strftime('%Y-%m-%d'),
                'duration_days': (df['timestamp'].max() - df['timestamp'].min()).days
            },
            'category_distribution': category_dist,
            'critical_events_count': len(critical_events),
            'critical_percentage': (len(critical_events) / total_events * 100) if total_events > 0 else 0,
            'temporal_patterns': {
                'hourly': hourly_patterns,
                'daily': daily_patterns,
                'alert_by_hour': alert_by_hour
            },
            'headcount_analysis': headcount_stats,
            'recent_trends': {str(k): v for k, v in recent_trends.items()},
            'peak_activity_hour': df.groupby('hour').size().idxmax() if total_events > 0 else None,
            'busiest_day': df.groupby('day_of_week').size().idxmax() if total_events > 0 else None
        }
        
        # Convert all numpy types to native Python types
        return self.convert_to_json_serializable(summary)
    
    def create_analysis_prompt(self, data_summary: Dict[str, Any]) -> str:
        """Create structured prompt for LLM analysis"""
        
        try:
            # Ensure data_summary is JSON serializable
            json_data = json.dumps(data_summary, indent=2)
        except TypeError as e:
            print(f"JSON serialization error: {e}")
            # Convert problematic types
            data_summary = self.convert_to_json_serializable(data_summary)
            json_data = json.dumps(data_summary, indent=2)
        
        prompt = f"""You are an expert security analyst specializing in classroom monitoring systems. Analyze the following classroom monitoring data and provide insights in the EXACT JSON format specified below.

DATA SUMMARY:
{json_data}

Provide your analysis in this EXACT JSON format (no additional text before or after):
{{
    "executive_summary": "Brief 2-3 sentence summary of overall situation",
    "risk_level": "low|medium|high|critical",
    "key_findings": [
        "Finding 1",
        "Finding 2", 
        "Finding 3"
    ],
    "recommendations": [
        "Recommendation 1",
        "Recommendation 2",
        "Recommendation 3"
    ],
    "temporal_patterns": {{
        "peak_times": "Description of peak activity patterns",
        "concerning_periods": "Any time periods of concern",
        "normal_operations": "Description of normal operational patterns"
    }},
    "security_insights": "Detailed security analysis focusing on alert patterns and potential threats",
    "operational_recommendations": [
        "Operational improvement 1",
        "Operational improvement 2",
        "Operational improvement 3"
    ],
    "predicted_concerns": [
        "Potential future concern 1",
        "Potential future concern 2"
    ],
    "confidence_score": 0.85
}}

Base your risk_level assessment on:
- LOW: <3% critical alerts, regular patterns
- MEDIUM: 3-7% critical alerts, some anomalies  
- HIGH: 7-15% critical alerts, significant anomalies
- CRITICAL: >15% critical alerts, severe anomalies

Focus on actionable insights that help administrators make informed decisions."""
        return prompt
    
    def analyze_data(self, df: pd.DataFrame) -> AnalysisResult:
        """Perform complete analysis of classroom monitoring data"""
        
        # Prepare data summary
        data_summary = self.prepare_data_summary(df)
        
        # Create analysis prompt
        prompt = self.create_analysis_prompt(data_summary)
        
        try:
            # Call Ollama API directly
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 2000
                }
            }
            
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            response_text = response_data.get('response', '').strip()
            
            print(f"LLM Response length: {len(response_text)}")
            
            # Find JSON content (in case there's extra text)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_text = response_text[json_start:json_end]
                analysis_dict = json.loads(json_text)
            else:
                raise ValueError("No valid JSON found in response")
            
            # Create structured result
            result = AnalysisResult(
                executive_summary=analysis_dict.get('executive_summary', 'Analysis completed'),
                risk_level=RiskLevel(analysis_dict.get('risk_level', 'medium')),
                key_findings=analysis_dict.get('key_findings', []),
                recommendations=analysis_dict.get('recommendations', []),
                temporal_patterns=analysis_dict.get('temporal_patterns', {}),
                security_insights=analysis_dict.get('security_insights', ''),
                operational_recommendations=analysis_dict.get('operational_recommendations', []),
                predicted_concerns=analysis_dict.get('predicted_concerns', []),
                confidence_score=float(analysis_dict.get('confidence_score', 0.5))
            )
            
            return result
            
        except Exception as e:
            # Return fallback analysis if LLM fails
            print(f"LLM analysis failed: {e}")
            return self._create_fallback_analysis(data_summary)
    
    def _create_fallback_analysis(self, data_summary: Dict[str, Any]) -> AnalysisResult:
        """Create basic analysis if LLM is unavailable"""
        
        critical_percentage = data_summary.get('critical_percentage', 0)
        
        # Determine risk level
        if critical_percentage > 15:
            risk_level = RiskLevel.CRITICAL
        elif critical_percentage > 7:
            risk_level = RiskLevel.HIGH
        elif critical_percentage > 3:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        peak_hour = data_summary.get('peak_activity_hour', 'N/A')
        
        return AnalysisResult(
            executive_summary=f"Analysis of {data_summary['total_events']} events with {critical_percentage:.1f}% critical alerts",
            risk_level=risk_level,
            key_findings=[
                f"Total events monitored: {data_summary['total_events']}",
                f"Critical alert rate: {critical_percentage:.1f}%",
                f"Peak activity hour: {peak_hour}:00" if peak_hour != 'N/A' else "Peak activity data not available"
            ],
            recommendations=[
                "Continue regular monitoring",
                "Review alert thresholds if needed",
                "Ensure adequate staffing during peak hours"
            ],
            temporal_patterns={
                "peak_times": f"Highest activity at hour {peak_hour}" if peak_hour != 'N/A' else "Peak times analysis pending",
                "concerning_periods": "None identified in basic analysis",
                "normal_operations": "Standard classroom patterns observed"
            },
            security_insights="Basic analysis indicates standard operational patterns",
            operational_recommendations=[
                "Maintain current monitoring protocols",
                "Regular equipment calibration",
                "Staff training updates"
            ],
            predicted_concerns=[
                "Monitor for pattern changes",
                "Watch for equipment degradation"
            ],
            confidence_score=0.7
        )

def test_analyzer():
    """Test the analyzer with sample data"""
    # Create sample data
    dates = pd.date_range('2025-08-20', '2025-08-24', freq='H')
    sample_data = {
        'id': range(len(dates)),
        'timestamp': dates,
        'category': ['normal'] * int(len(dates) * 0.8) + 
                   ['pre_alert'] * int(len(dates) * 0.15) + 
                   ['alert'] * int(len(dates) * 0.04) + 
                   ['human_intervention_needed'] * int(len(dates) * 0.01),
        'headcount': [20, 25, 30, 15, 22] * (len(dates) // 5 + 1)
    }
    
    df = pd.DataFrame(sample_data)
    df = df.head(len(dates))  # Ensure exact length
    
    analyzer = OllamaClassroomAnalyzer()
    result = analyzer.analyze_data(df)
    
    print("Analysis completed:")
    print(f"Risk Level: {result.risk_level.value}")
    print(f"Executive Summary: {result.executive_summary}")
    print(f"Key Findings: {result.key_findings}")

if __name__ == "__main__":
    test_analyzer()
