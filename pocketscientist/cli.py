"""
Command-line interface for PocketScientist.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click


@click.command()
@click.argument("dataset", type=click.Path(exists=True, readable=True))
@click.argument("context", type=str)
@click.option(
    "--output-dir",
    "-o",
    default="results",
    help="Output directory for results (default: results/)",
)
@click.option(
    "--max-time",
    "-t",
    default=1800,
    type=int,
    help="Maximum execution time in seconds (default: 1800 = 30 minutes)",
)
@click.option(
    "--model-endpoint",
    default="http://localhost:11434",
    help="LLM endpoint URL (default: http://localhost:11434 for Ollama)",
)
@click.option(
    "--model-name",
    default="llama3.1",
    help="Model name to use (default: llama3.1)",
)
@click.option(
    "--api-key",
    envvar="POCKETSCIENTIST_API_KEY",
    help="API key for LLM provider (can also set POCKETSCIENTIST_API_KEY env var)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    dataset: str,
    context: str,
    output_dir: str,
    max_time: int,
    model_endpoint: str,
    model_name: str,
    api_key: Optional[str],
    verbose: bool,
) -> None:
    """
    PocketScientist - AI agent orchestration for data science.
    
    DATASET: Path to the CSV dataset file
    CONTEXT: Instructions or questions about the data
    """
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = Path(output_dir) / timestamp
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        click.echo(f"Starting PocketScientist analysis...")
        click.echo(f"Dataset: {dataset}")
        click.echo(f"Context: {context}")
        click.echo(f"Output directory: {run_output_dir}")
        click.echo(f"Max time: {max_time} seconds")
        click.echo(f"Model endpoint: {model_endpoint}")
        click.echo(f"Model name: {model_name}")
    
    try:
        # Import here to avoid circular imports and improve startup time
        from pocketscientist.orchestrator import DataScienceOrchestrator
        
        orchestrator = DataScienceOrchestrator(
            dataset_path=dataset,
            context=context,
            output_dir=run_output_dir,
            max_time=max_time,
            model_endpoint=model_endpoint,
            model_name=model_name,
            api_key=api_key,
            verbose=verbose,
        )
        
        # Run the analysis
        result = orchestrator.run()
        
        if result["success"]:
            click.echo(f"✅ Analysis completed successfully!")
            click.echo(f"📊 Notebook: {result['notebook_path']}")
            click.echo(f"📋 Report: {result['report_path']}")
        else:
            click.echo(f"❌ Analysis failed: {result['error']}", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Fatal error: {str(e)}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()