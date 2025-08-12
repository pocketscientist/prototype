# PocketScientist

> **🏆 OpenAI Open Model Hackathon 2025 Entry**  
> This project is being developed for the [OpenAI Open Model Hackathon](https://openai.devpost.com/) using gpt-oss models. Following the hackathon, it will be released as open source under the Apache 2.0 license.

**AI-Powered Data Science Agent Orchestration following CRISP-DM Methodology**

PocketScientist is an innovative AI agent orchestration system that emulates the work of a data scientist by following the industry-standard CRISP-DM (Cross Industry Standard Process for Data Mining) methodology. It automatically iterates through data science processes, creating comprehensive analyses with minimal human intervention.

## 🌟 Features

- **CRISP-DM Methodology**: Follows the complete CRISP-DM workflow with intelligent phase transitions
- **Iterative Process**: Agents can revisit and refine earlier phases based on new findings
- **Living Documentation**: Generates comprehensive Jupyter notebooks showing the complete analytical journey
- **Comprehensive Reports**: Creates professional HTML reports with insights and recommendations
- **Safety Monitoring**: Built-in infinite loop prevention and time management
- **LLM Agnostic**: Generic interface supporting multiple LLM providers (Ollama by default)
- **Automated Insights**: AI agents generate code, analysis, and recommendations automatically

## ⚠️ Security Notice

**Important**: PocketScientist executes AI-generated code directly on your local machine without sandboxing. Only run this tool on trusted datasets and review generated notebooks before executing them in production environments.

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Ollama running locally (default) or access to another LLM provider

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pocketscientist/prototype
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Test the installation:
```bash
python test_run.py
```

### Basic Usage

```bash
# Analyze a dataset with a research question
python -m pocketscientist.cli sample_data.csv "what interesting patterns can you find in this customer data?"

# With custom options
python -m pocketscientist.cli data.csv "predict customer churn" \
  --max-time 3600 \
  --output-dir results \
  --model-name llama3.1 \
  --verbose
```

### Example Output

Each run generates:
- **Jupyter Notebook**: Complete analysis with code, visualizations, and insights (`analysis.ipynb`)
- **HTML Report**: Executive summary with recommendations (`report.html`)
- **Timestamped Directory**: All outputs organized by execution time

## 🧠 CRISP-DM Workflow

PocketScientist implements all six phases of CRISP-DM:

1. **Business Understanding** - Define objectives and requirements from user context
2. **Data Understanding** - Initial data exploration and quality assessment  
3. **Data Preparation** - Data cleaning, transformation, and feature engineering
4. **Modeling** - Algorithm selection and model building (traditional ML)
5. **Evaluation** - Model performance assessment and validation
6. **Deployment Preparation** - Final recommendations and actionable insights

### Intelligent Coordination

- **Adaptive Flow**: Agents decide when to move forward or revisit previous phases
- **Context Awareness**: Analysis approach adapts based on user questions and data characteristics  
- **Safety Monitoring**: Prevents infinite loops with time limits and iteration caps
- **Error Recovery**: Graceful handling of failures with fallback strategies

### Code Execution System

- **Local Python Execution**: Executes generated code directly on your machine (not sandboxed)
- **Persistent Namespace**: Maintains variables and state across code cells and phases
- **Automatic Visualization**: Captures and saves matplotlib/seaborn/plotly plots
- **Error Analysis**: Analyzes execution failures and automatically retries with corrected code
- **Working Directory**: Changes to and operates in the specified output directory

## 📊 Supported Analysis Types

- **Descriptive Analytics**: Data exploration and pattern discovery
- **Diagnostic Analytics**: Root cause analysis and relationship investigation
- **Predictive Analytics**: Machine learning model building and forecasting
- **Prescriptive Analytics**: Recommendations and optimization strategies

## 🛠️ Configuration Options

### Command Line Arguments

- `dataset`: Path to CSV dataset file (required)
- `context`: Analysis question or objective (required)
- `--output-dir`: Output directory (default: results/)
- `--max-time`: Maximum execution time in seconds (default: 1800)
- `--model-endpoint`: LLM endpoint URL (default: http://localhost:11434)
- `--model-name`: Model name (default: llama3.1)
- `--api-key`: API key for LLM provider
- `--verbose`: Enable detailed logging

### Environment Variables

- `POCKETSCIENTIST_API_KEY`: API key for LLM provider (only supported environment variable)

## 🔧 LLM Provider Support

Currently supported:
- **Ollama** (default) - Local LLM deployment

Note: The system is designed specifically for Ollama. Other providers are not currently implemented.

## 📈 Example Analysis Flow

```
User: "Is there anything interesting in my sales data?"

1. Business Understanding: Define exploratory analysis objectives
2. Data Understanding: Load data, assess quality, create initial visualizations  
3. Data Preparation: Clean missing values, handle outliers
4. Coordinator Decision: Skip modeling (exploratory focus)
5. Deployment Preparation: Generate insights and recommendations

Output: Complete notebook + HTML report with findings
```

## 🧪 Testing

Run the integration test suite:

```bash
python test_run.py          # System integration tests
python test_execution.py    # Code execution engine tests
```

Tests cover:
- Module imports and basic functionality
- Dataset validation and loading
- Code execution engine
- Core integration testing

Note: Current testing focuses on integration rather than comprehensive unit testing.

## 🏗️ Architecture

```
pocketscientist/
├── agents/                 # CRISP-DM phase agents
│   ├── base.py            # Base agent class
│   ├── business_understanding.py
│   ├── data_understanding.py
│   ├── data_preparation.py
│   ├── modeling.py
│   ├── evaluation.py
│   ├── deployment_preparation.py
│   ├── analysis_report.py # Final analysis report agent
│   └── coordinator.py     # Workflow coordination
├── llm/                   # LLM provider interface
│   ├── base.py           # Abstract provider
│   ├── ollama.py         # Ollama implementation
│   └── factory.py        # Provider factory
├── execution/             # Code execution system
│   └── cell_executor.py  # Python code execution engine
├── utils/                 # Utilities and safety
│   ├── safety.py         # Loop prevention
│   └── validation.py     # Data validation
├── orchestrator.py        # Main workflow engine
├── notebook_builder.py    # Jupyter notebook generation
├── report_generator.py    # HTML report creation
└── cli.py                # Command line interface (Click-based)
```

## 🔒 Safety Features

- **Time Limits**: Configurable maximum execution time
- **Iteration Caps**: Prevents infinite phase loops
- **Phase Monitoring**: Detects rapid cycling and excessive backtracking
- **Graceful Degradation**: Continues analysis even if individual phases fail
- **Resource Monitoring**: Tracks execution statistics and efficiency

## 🎯 Limitations

Current version limitations:
- **CSV Only**: Currently supports only CSV datasets
- **Traditional ML**: No deep learning or GPU acceleration support
- **Single Dataset**: One dataset per analysis run
- **English Only**: LLM interactions in English

## 🚧 Future Roadmap

- [ ] Support for additional data formats (JSON, Excel, Parquet)
- [ ] Deep learning and GPU acceleration support
- [ ] Multi-dataset analysis capabilities
- [ ] Integration with cloud data sources
- [ ] Advanced visualization options
- [ ] Model deployment automation
- [ ] Collaborative analysis features

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style and standards
- Testing requirements
- Pull request process
- Issue reporting

## 📄 License

This project will be released under the Apache 2.0 License following the OpenAI Open Model Hackathon.

## 🙏 Acknowledgments

- Built with the CRISP-DM methodology
- Uses Click for command-line interface
- Jupyter ecosystem for notebook generation
- Inspired by the data science community's need for automated analysis tools

---

**PocketScientist** - Bringing AI-powered data science to everyone 🔬✨