"""
Report Generator for Classroom Monitoring System
Generates PDF, HTML, and JSON reports from analysis results
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict
import base64
from io import BytesIO

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import Color, HexColor
from jinja2 import Template

from llm_analyzer import AnalysisResult, OllamaClassroomAnalyzer

class ReportGenerator:
    """Generate comprehensive reports from classroom monitoring analysis"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.custom_styles = self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles"""
        styles = {}
        
        # Title style
        styles['CustomTitle'] = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=HexColor('#1f4e79'),
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        
        # Heading style
        styles['CustomHeading'] = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=HexColor('#2d5aa0'),
            spaceBefore=20,
            spaceAfter=10
        )
        
        # Risk level styles
        styles['RiskHigh'] = ParagraphStyle(
            'RiskHigh',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=HexColor('#d32f2f'),
            fontName='Helvetica-Bold'
        )
        
        styles['RiskMedium'] = ParagraphStyle(
            'RiskMedium',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=HexColor('#f57c00'),
            fontName='Helvetica-Bold'
        )
        
        styles['RiskLow'] = ParagraphStyle(
            'RiskLow',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=HexColor('#388e3c'),
            fontName='Helvetica-Bold'
        )
        
        styles['RiskCritical'] = ParagraphStyle(
            'RiskCritical',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=HexColor('#d32f2f'),
            fontName='Helvetica-Bold'
        )
        
        return styles
    
    def generate_pdf_report(self, 
                          analysis: AnalysisResult, 
                          df: pd.DataFrame, 
                          output_path: str) -> str:
        """Generate comprehensive PDF report"""
        
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        story = []
        
        # Title
        title = Paragraph("Classroom Monitoring Analysis Report", self.custom_styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Generation info
        gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        gen_info = Paragraph(f"<b>Generated:</b> {gen_time}<br/><b>Period:</b> {df['timestamp'].min().date()} to {df['timestamp'].max().date()}", self.styles['Normal'])
        story.append(gen_info)
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.custom_styles['CustomHeading']))
        story.append(Paragraph(analysis.executive_summary, self.styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Risk Assessment
        story.append(Paragraph("Risk Assessment", self.custom_styles['CustomHeading']))
        risk_style = self.custom_styles.get(f'Risk{analysis.risk_level.value.title()}', self.styles['Normal'])
        risk_text = f"Risk Level: {analysis.risk_level.value.upper()}"
        story.append(Paragraph(risk_text, risk_style))
        story.append(Paragraph(f"Confidence Score: {analysis.confidence_score:.2%}", self.styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Key Statistics Table
        story.append(Paragraph("Key Statistics", self.custom_styles['CustomHeading']))
        
        stats_data = [
            ['Metric', 'Value'],
            ['Total Events', str(len(df))],
            ['Critical Alerts', str(len(df[df['category'].isin(['alert', 'human_intervention_needed'])]))],
            ['Average Headcount', f"{df['headcount'].mean():.1f}"],
            ['Peak Activity Hour', f"{df.groupby(df['timestamp'].dt.hour).size().idxmax()}:00"],
            ['Monitoring Period', f"{(df['timestamp'].max() - df['timestamp'].min()).days} days"]
        ]
        
        stats_table = Table(stats_data)
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#000000')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#ffffff')),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#cccccc'))
        ]))
        
        story.append(stats_table)
        story.append(Spacer(1, 20))
        
        # Key Findings
        story.append(Paragraph("Key Findings", self.custom_styles['CustomHeading']))
        for finding in analysis.key_findings:
            story.append(Paragraph(f"• {finding}", self.styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Security Insights
        story.append(Paragraph("Security Analysis", self.custom_styles['CustomHeading']))
        story.append(Paragraph(analysis.security_insights, self.styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Temporal Patterns
        story.append(Paragraph("Temporal Patterns", self.custom_styles['CustomHeading']))
        for key, value in analysis.temporal_patterns.items():
            story.append(Paragraph(f"<b>{key.replace('_', ' ').title()}:</b> {value}", self.styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Recommendations
        story.append(Paragraph("Recommendations", self.custom_styles['CustomHeading']))
        for rec in analysis.recommendations:
            story.append(Paragraph(f"• {rec}", self.styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Operational Recommendations
        story.append(Paragraph("Operational Improvements", self.custom_styles['CustomHeading']))
        for op_rec in analysis.operational_recommendations:
            story.append(Paragraph(f"• {op_rec}", self.styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Predicted Concerns
        story.append(Paragraph("Potential Future Concerns", self.custom_styles['CustomHeading']))
        for concern in analysis.predicted_concerns:
            story.append(Paragraph(f"• {concern}", self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        return output_path
    
    def generate_html_report(self, 
                           analysis: AnalysisResult, 
                           df: pd.DataFrame) -> str:
        """Generate HTML report"""
        
        template_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Classroom Monitoring Report</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; color: #1f4e79; border-bottom: 3px solid #1f4e79; padding-bottom: 20px; margin-bottom: 30px; }
        .section { margin: 25px 0; }
        .section h2 { color: #2d5aa0; border-left: 4px solid #2d5aa0; padding-left: 15px; }
        .risk-critical { color: #d32f2f; font-weight: bold; font-size: 18px; }
        .risk-high { color: #f57c00; font-weight: bold; font-size: 18px; }
        .risk-medium { color: #ffa726; font-weight: bold; font-size: 18px; }
        .risk-low { color: #388e3c; font-weight: bold; font-size: 18px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .stat-card { background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; border-left: 4px solid #2d5aa0; }
        .stat-value { font-size: 24px; font-weight: bold; color: #1f4e79; }
        ul { padding-left: 20px; }
        li { margin: 8px 0; }
        .footer { text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ccc; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏫 Classroom Monitoring Analysis Report</h1>
            <p><strong>Generated:</strong> {{ generation_time }}</p>
            <p><strong>Analysis Period:</strong> {{ date_range }}</p>
        </div>
        
        <div class="section">
            <h2>📊 Executive Summary</h2>
            <p>{{ analysis.executive_summary }}</p>
        </div>
        
        <div class="section">
            <h2>⚠️ Risk Assessment</h2>
            <p class="risk-{{ analysis.risk_level.value }}">Risk Level: {{ analysis.risk_level.value.upper() }}</p>
            <p><strong>Confidence Score:</strong> {{ "%.1f"|format(analysis.confidence_score * 100) }}%</p>
        </div>
        
        <div class="section">
            <h2>📈 Key Statistics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{{ total_events }}</div>
                    <div>Total Events</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ critical_events }}</div>
                    <div>Critical Alerts</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ "%.1f"|format(avg_headcount) }}</div>
                    <div>Avg Headcount</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ peak_hour }}:00</div>
                    <div>Peak Hour</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>🔍 Key Findings</h2>
            <ul>
                {% for finding in analysis.key_findings %}
                <li>{{ finding }}</li>
                {% endfor %}
            </ul>
        </div>
        
        <div class="section">
            <h2>🔒 Security Analysis</h2>
            <p>{{ analysis.security_insights }}</p>
        </div>
        
        <div class="section">
            <h2>⏰ Temporal Patterns</h2>
            {% for key, value in analysis.temporal_patterns.items() %}
            <p><strong>{{ key.replace('_', ' ').title() }}:</strong> {{ value }}</p>
            {% endfor %}
        </div>
        
        <div class="section">
            <h2>💡 Recommendations</h2>
            <ul>
                {% for rec in analysis.recommendations %}
                <li>{{ rec }}</li>
                {% endfor %}
            </ul>
        </div>
        
        <div class="section">
            <h2>⚙️ Operational Improvements</h2>
            <ul>
                {% for op_rec in analysis.operational_recommendations %}
                <li>{{ op_rec }}</li>
                {% endfor %}
            </ul>
        </div>
        
        <div class="section">
            <h2>🔮 Predicted Concerns</h2>
            <ul>
                {% for concern in analysis.predicted_concerns %}
                <li>{{ concern }}</li>
                {% endfor %}
            </ul>
        </div>
        
        <div class="footer">
            <p>Report generated by Classroom Monitoring System | {{ generation_time }}</p>
        </div>
    </div>
</body>
</html>
        """
        
        template = Template(template_html)
        
        return template.render(
            analysis=analysis,
            generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            date_range=f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}",
            total_events=len(df),
            critical_events=len(df[df['category'].isin(['alert', 'human_intervention_needed'])]),
            avg_headcount=df['headcount'].mean(),
            peak_hour=df.groupby(df['timestamp'].dt.hour).size().idxmax()
        )
    
    def generate_json_report(self, 
                           analysis: AnalysisResult, 
                           df: pd.DataFrame) -> str:
        """Generate JSON report"""
        
        # Prepare comprehensive data
        report_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_version": "1.0",
                "data_period": {
                    "start": df['timestamp'].min().isoformat(),
                    "end": df['timestamp'].max().isoformat(),
                    "total_days": (df['timestamp'].max() - df['timestamp'].min()).days
                }
            },
            "analysis_results": asdict(analysis),
            "statistics": {
                "total_events": len(df),
                "category_distribution": df['category'].value_counts().to_dict(),
                "critical_events": len(df[df['category'].isin(['alert', 'human_intervention_needed'])]),
                "headcount_stats": {
                    "mean": float(df['headcount'].mean()),
                    "median": float(df['headcount'].median()),
                    "max": int(df['headcount'].max()),
                    "min": int(df['headcount'].min()),
                    "std": float(df['headcount'].std())
                },
                "temporal_analysis": {
                    "peak_hour": int(df.groupby(df['timestamp'].dt.hour).size().idxmax()),
                    "hourly_distribution": df.groupby(df['timestamp'].dt.hour).size().to_dict(),
                    "daily_distribution": df.groupby(df['timestamp'].dt.day_name()).size().to_dict()
                }
            }
        }
        
        return json.dumps(report_data, indent=2, default=str)
    
    def generate_complete_report_package(self, 
                                       df: pd.DataFrame, 
                                       output_dir: str = "reports") -> Dict[str, str]:
        """Generate complete report package with all formats"""
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate analysis
        analyzer = OllamaClassroomAnalyzer()
        analysis = analyzer.analyze_data(df)
        
        # Generate timestamp for files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate reports
        files = {}
        
        # PDF Report
        pdf_path = output_path / f"classroom_report_{timestamp}.pdf"
        files['pdf'] = self.generate_pdf_report(analysis, df, str(pdf_path))
        
        # HTML Report
        html_content = self.generate_html_report(analysis, df)
        html_path = output_path / f"classroom_report_{timestamp}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        files['html'] = str(html_path)
        
        # JSON Report
        json_content = self.generate_json_report(analysis, df)
        json_path = output_path / f"classroom_report_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(json_content)
        files['json'] = str(json_path)
        
        return files

def test_report_generation():
    """Test report generation with sample data"""
    # Create sample data
    dates = pd.date_range('2025-08-20', '2025-08-24', freq='2H')
    sample_data = {
        'id': range(len(dates)),
        'timestamp': dates,
        'category': ['normal'] * int(len(dates) * 0.75) + 
                   ['pre_alert'] * int(len(dates) * 0.15) + 
                   ['alert'] * int(len(dates) * 0.08) + 
                   ['human_intervention_needed'] * int(len(dates) * 0.02),
        'headcount': [15, 20, 25, 18, 22] * (len(dates) // 5 + 1)
    }
    
    df = pd.DataFrame(sample_data)
    df = df.head(len(dates))
    
    generator = ReportGenerator()
    files = generator.generate_complete_report_package(df)
    
    print("Reports generated:")
    for format_type, file_path in files.items():
        print(f"- {format_type.upper()}: {file_path}")

if __name__ == "__main__":
    test_report_generation()
