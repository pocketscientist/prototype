"""
HTML report generator for PocketScientist analysis results.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from jinja2 import Template

from .agents.base import AgentState


class ReportGenerator:
    """Generates HTML reports from analysis results."""
    
    def __init__(self):
        self.template = self._get_report_template()
    
    def generate_report(
        self,
        state: AgentState,
        output_path: Path,
        notebook_path: Optional[Path] = None
    ) -> None:
        """
        Generate an HTML report from the analysis state.
        
        Args:
            state: AgentState containing analysis results
            output_path: Path where to save the HTML report
            notebook_path: Optional path to the Jupyter notebook
        """
        
        # Prepare data for template
        template_data = self._prepare_template_data(state, notebook_path)
        
        # Render HTML
        html_content = self.template.render(**template_data)
        
        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _prepare_template_data(
        self,
        state: AgentState,
        notebook_path: Optional[Path]
    ) -> Dict[str, Any]:
        """Prepare data for the HTML template."""
        
        # Calculate execution statistics
        execution_time = (datetime.now() - state.start_time).total_seconds()
        
        # Extract key insights from findings
        insights = self._extract_key_insights(state.findings)
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(state)
        
        # Prepare phase timeline
        timeline = self._create_phase_timeline(state)
        
        return {
            "title": f"PocketScientist Analysis Report",
            "dataset_name": Path(state.dataset_path).name,
            "user_context": state.user_context,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "execution_time": f"{execution_time:.1f} seconds",
            "iterations": state.iteration_count,
            "phases_completed": list(state.completed_phases),
            "phase_history": state.phase_history,
            "timeline": timeline,
            "executive_summary": executive_summary,
            "key_insights": insights,
            "findings": state.findings,
            "errors": state.errors,
            "notebook_path": notebook_path.name if notebook_path else None,
            "success_rate": self._calculate_success_rate(state),
            "recommendations": self._extract_recommendations(state)
        }
    
    def _extract_key_insights(self, findings: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract key insights from findings."""
        
        insights = []
        
        for phase, phase_findings in findings.items():
            if not isinstance(phase_findings, dict):
                continue
                
            # Extract insights based on phase
            if phase == "business_understanding":
                if phase_findings.get("objectives_defined"):
                    insights.append({
                        "category": "Objectives",
                        "insight": f"Analysis type identified: {phase_findings.get('analysis_type', 'Unknown')}",
                        "phase": phase
                    })
            
            elif phase == "data_understanding":
                if phase_findings.get("data_loaded"):
                    insights.append({
                        "category": "Data Quality",
                        "insight": "Dataset successfully loaded and explored",
                        "phase": phase
                    })
            
            elif phase == "data_preparation":
                if phase_findings.get("data_preparation_complete"):
                    strategy = phase_findings.get("preparation_strategy", {})
                    insights.append({
                        "category": "Data Processing",
                        "insight": f"Data prepared using {strategy.get('description', 'standard')} approach",
                        "phase": phase
                    })
            
            elif phase == "modeling":
                if phase_findings.get("models_built"):
                    algorithms = phase_findings.get("algorithms_used", [])
                    insights.append({
                        "category": "Modeling",
                        "insight": f"Built {len(algorithms)} models: {', '.join(algorithms)}",
                        "phase": phase
                    })
            
            elif phase == "evaluation":
                if phase_findings.get("evaluation_complete"):
                    insights.append({
                        "category": "Model Performance",
                        "insight": "Model performance evaluated and validated",
                        "phase": phase
                    })
            
            elif phase == "deployment_preparation":
                if phase_findings.get("deployment_ready"):
                    insights.append({
                        "category": "Recommendations",
                        "insight": "Final recommendations and implementation roadmap prepared",
                        "phase": phase
                    })
        
        return insights
    
    def _generate_executive_summary(self, state: AgentState) -> str:
        """Generate an executive summary of the analysis."""
        
        summary_parts = []
        
        # Opening statement
        summary_parts.append(
            f"This report presents the results of an automated data science analysis "
            f"conducted on the dataset '{Path(state.dataset_path).name}' using the CRISP-DM methodology."
        )
        
        # Context
        if state.user_context:
            summary_parts.append(
                f"The analysis was focused on: {state.user_context}"
            )
        
        # Execution summary
        execution_time = (datetime.now() - state.start_time).total_seconds()
        summary_parts.append(
            f"The analysis completed in {execution_time:.1f} seconds across {state.iteration_count} "
            f"iterations, successfully executing {len(state.completed_phases)} CRISP-DM phases."
        )
        
        # Key outcomes
        if state.completed_phases:
            phases_text = ", ".join(sorted(state.completed_phases))
            summary_parts.append(
                f"The following phases were completed: {phases_text}."
            )
        
        # Success indicators
        if "deployment_preparation" in state.completed_phases:
            summary_parts.append(
                "The analysis reached the deployment phase, providing actionable recommendations "
                "and a comprehensive implementation roadmap."
            )
        
        return " ".join(summary_parts)
    
    def _create_phase_timeline(self, state: AgentState) -> List[Dict[str, str]]:
        """Create a timeline of phase execution."""
        
        timeline = []
        phase_names = {
            "business_understanding": "Business Understanding",
            "data_understanding": "Data Understanding", 
            "data_preparation": "Data Preparation",
            "modeling": "Modeling",
            "evaluation": "Evaluation",
            "deployment_preparation": "Deployment Preparation",
            "coordinator": "Workflow Coordination"
        }
        
        for i, phase in enumerate(state.phase_history):
            timeline.append({
                "step": str(i + 1),
                "phase": phase_names.get(phase, phase.title()),
                "status": "completed" if phase in state.completed_phases else "attempted",
                "description": self._get_phase_description(phase)
            })
        
        return timeline
    
    def _get_phase_description(self, phase: str) -> str:
        """Get description for a phase."""
        
        descriptions = {
            "business_understanding": "Defined analysis objectives and success criteria",
            "data_understanding": "Explored dataset structure and quality",
            "data_preparation": "Cleaned and prepared data for analysis", 
            "modeling": "Built and trained machine learning models",
            "evaluation": "Evaluated model performance and validity",
            "deployment_preparation": "Prepared final recommendations and insights",
            "coordinator": "Coordinated workflow and phase transitions"
        }
        
        return descriptions.get(phase, f"Executed {phase} phase")
    
    def _calculate_success_rate(self, state: AgentState) -> float:
        """Calculate overall success rate."""
        
        total_phases = len(set(state.phase_history))
        successful_phases = len(state.completed_phases)
        
        if total_phases == 0:
            return 0.0
        
        return (successful_phases / total_phases) * 100
    
    def _extract_recommendations(self, state: AgentState) -> List[str]:
        """Extract recommendations from deployment phase."""
        
        recommendations = []
        
        deployment_findings = state.findings.get("deployment_preparation", {})
        if isinstance(deployment_findings, dict):
            if deployment_findings.get("recommendations_prepared"):
                recommendations.append("Review and implement the actionable recommendations")
                recommendations.append("Monitor key metrics identified in the analysis")
                recommendations.append("Consider regular re-analysis to track changes over time")
        
        # Add default recommendations if none found
        if not recommendations:
            recommendations = [
                "Review the analysis findings in the Jupyter notebook",
                "Validate results with domain expertise", 
                "Implement data-driven decision making based on insights"
            ]
        
        return recommendations
    
    def _get_report_template(self) -> Template:
        """Get the HTML template for the report."""
        
        template_string = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .meta-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .meta-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        
        .meta-card h3 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        
        .section {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .section h2 {
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }
        
        .timeline {
            position: relative;
            padding-left: 30px;
        }
        
        .timeline::before {
            content: '';
            position: absolute;
            left: 15px;
            top: 0;
            bottom: 0;
            width: 2px;
            background: #667eea;
        }
        
        .timeline-item {
            position: relative;
            margin-bottom: 20px;
            padding-left: 30px;
        }
        
        .timeline-item::before {
            content: attr(data-step);
            position: absolute;
            left: -22px;
            top: 0;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background: #667eea;
            color: white;
            text-align: center;
            line-height: 30px;
            font-size: 0.9em;
            font-weight: bold;
        }
        
        .timeline-item.completed::before {
            background: #28a745;
        }
        
        .timeline-item h4 {
            margin-bottom: 5px;
            color: #333;
        }
        
        .insights-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .insight-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #28a745;
        }
        
        .insight-category {
            font-weight: bold;
            color: #28a745;
            margin-bottom: 5px;
        }
        
        .recommendations ul {
            list-style: none;
            padding: 0;
        }
        
        .recommendations li {
            background: #e8f4fd;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #0066cc;
            position: relative;
            padding-left: 30px;
        }
        
        .recommendations li::before {
            content: "→";
            position: absolute;
            left: 10px;
            color: #0066cc;
            font-weight: bold;
        }
        
        .error-list {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 15px;
        }
        
        .error-list ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        
        .success-badge {
            display: inline-block;
            background: #28a745;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            margin: 10px 0;
        }
        
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
            border-top: 1px solid #eee;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header {
                padding: 20px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .section {
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ title }}</h1>
            <div class="subtitle">Dataset: {{ dataset_name }}</div>
            <div class="subtitle">Generated on {{ generated_at }}</div>
        </div>
        
        <div class="meta-info">
            <div class="meta-card">
                <h3>Analysis Context</h3>
                <p>{{ user_context }}</p>
            </div>
            <div class="meta-card">
                <h3>Execution Summary</h3>
                <p><strong>Time:</strong> {{ execution_time }}</p>
                <p><strong>Iterations:</strong> {{ iterations }}</p>
                <p><strong>Success Rate:</strong> {{ "%.1f"|format(success_rate) }}%</p>
            </div>
            <div class="meta-card">
                <h3>Phases Completed</h3>
                <p>{{ phases_completed|length }} of {{ timeline|length }} phases</p>
                {% if notebook_path %}
                <p><strong>Notebook:</strong> {{ notebook_path }}</p>
                {% endif %}
            </div>
        </div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <p>{{ executive_summary }}</p>
            <div class="success-badge">
                Analysis {{ "Completed Successfully" if success_rate > 80 else "Partially Completed" }}
            </div>
        </div>
        
        {% if key_insights %}
        <div class="section">
            <h2>Key Insights</h2>
            <div class="insights-grid">
                {% for insight in key_insights %}
                <div class="insight-card">
                    <div class="insight-category">{{ insight.category }}</div>
                    <p>{{ insight.insight }}</p>
                    <small style="color: #666;">From {{ insight.phase|title }}</small>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}
        
        <div class="section">
            <h2>Analysis Timeline</h2>
            <div class="timeline">
                {% for item in timeline %}
                <div class="timeline-item {{ item.status }}" data-step="{{ item.step }}">
                    <h4>{{ item.phase }}</h4>
                    <p>{{ item.description }}</p>
                </div>
                {% endfor %}
            </div>
        </div>
        
        {% if recommendations %}
        <div class="section recommendations">
            <h2>Recommendations</h2>
            <ul>
                {% for rec in recommendations %}
                <li>{{ rec }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}
        
        {% if errors %}
        <div class="section">
            <h2>Issues & Warnings</h2>
            <div class="error-list">
                <p><strong>The following issues were encountered during analysis:</strong></p>
                <ul>
                    {% for error in errors %}
                    <li>{{ error }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>
        {% endif %}
        
        <div class="footer">
            <p>Generated by PocketScientist v0.1.0 - AI-Powered Data Science</p>
            <p>Following CRISP-DM Methodology with Iterative Agent Orchestration</p>
        </div>
    </div>
</body>
</html>
        """
        
        return Template(template_string)