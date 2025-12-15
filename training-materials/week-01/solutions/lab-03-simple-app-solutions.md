# Lab 03 – Simple AI Application (Instructor Solution)

## Exercise 2.1 – `EnhancedConfig`

```python
class EnhancedConfig(Config):
    def __init__(self, config_file: Optional[str] = None, profile: str = "default"):
        super().__init__(config_file)
        self.profile = profile
        self.profiles = {
            "development": {"model": "gpt-3.5-turbo", "temperature": 0.5, "max_tokens": 300, "log_level": "DEBUG"},
            "production": {"model": "gpt-4", "temperature": 0.2, "max_tokens": 500, "log_level": "WARNING"},
        }
        self.load_from_env()
        if profile in self.profiles:
            self.load_profile(profile)
        self.validate()

    def load_from_env(self) -> None:
        overrides = {
            "model": os.getenv("SUMMARIZER_MODEL"),
            "temperature": os.getenv("SUMMARIZER_TEMPERATURE"),
            "max_tokens": os.getenv("SUMMARIZER_MAX_TOKENS"),
            "log_level": os.getenv("SUMMARIZER_LOG_LEVEL"),
            "cost_warning_threshold": os.getenv("SUMMARIZER_COST_THRESHOLD"),
        }
        for key, value in overrides.items():
            if value is not None:
                if key in {"temperature", "cost_warning_threshold"}:
                    self.config[key] = float(value)
                elif key == "max_tokens":
                    self.config[key] = int(value)
                else:
                    self.config[key] = value

    def validate(self) -> None:
        if not 0 <= self.config["temperature"] <= 1:
            raise ValueError("Temperature must be between 0 and 1.")
        if self.config["max_tokens"] <= 0:
            raise ValueError("max_tokens must be positive.")
        if self.config["cost_warning_threshold"] < 0:
            raise ValueError("Cost threshold must be non-negative.")

    def load_profile(self, name: str) -> None:
        profile_config = self.profiles.get(name)
        if not profile_config:
            raise ValueError(f"Unknown profile: {name}")
        self.config.update(profile_config)
```

---

## Exercise 4.1 – `BatchSummarizer`

```python
class BatchSummarizer(SummarizerEngine):
    def __init__(self, config: Config):
        super().__init__(config)
        self.batch_results: list[dict] = []

    def summarize_batch(self, texts: list[str], mode: str = "concise", show_progress: bool = True) -> list[dict]:
        self.batch_results.clear()
        iterator = enumerate(texts, start=1)

        def process(index: int, text: str) -> None:
            try:
                result = self.summarize(text, mode=mode)
                result["index"] = index
                result["status"] = "success"
                self.batch_results.append(result)
            except Exception as exc:
                self.batch_results.append({"index": index, "status": "failed", "error": str(exc)})

        if show_progress:
            with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
                task_id = progress.add_task(f"Summarizing {len(texts)} documents", total=len(texts))
                for idx, text in iterator:
                    process(idx, text)
                    progress.advance(task_id)
        else:
            for idx, text in iterator:
                process(idx, text)
        return self.batch_results

    def get_batch_report(self) -> dict[str, float | int]:
        successes = [r for r in self.batch_results if r.get("status") == "success"]
        failures = [r for r in self.batch_results if r.get("status") == "failed"]
        total_cost = sum(r.get("cost", 0.0) for r in successes)
        total_tokens = sum(r.get("total_tokens", 0) for r in successes)
        return {
            "items": len(self.batch_results),
            "success": len(successes),
            "failed": len(failures),
            "success_rate": len(successes) / max(1, len(self.batch_results)),
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "avg_cost": total_cost / max(1, len(successes))
        }
```

---

## Exercise 6.1 – Click-Based CLI

```python
@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Text Summarizer AI – command line."""
    pass

@cli.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--mode", "-m", default="concise", type=click.Choice(["concise", "detailed", "bullets", "key_points", "executive", "technical"]))
@click.option("--output", "-o", type=click.Path(dir_okay=False))
@click.option("--stream/--no-stream", default=False)
@click.option("--config", type=click.Path(exists=True, dir_okay=False), help="Optional config file")
def summarize(input_file, mode, output, stream, config):
    """Summarize a single document."""
    app = TextSummarizerApp(config_file=config)
    result = app.summarize_file(input_file, output_file=output, mode=mode, stream=stream)
    if not stream:
        console.print(Panel(result["summary"], title="Summary", border_style="green"))
        console.print(f"Tokens: {result['total_tokens']} | Cost: ${result['cost']:.6f}")

@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--mode", "-m", default="concise")
@click.option("--pattern", "-p", default="*.txt")
@click.option("--config", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", type=click.Path(dir_okay=False), help="Optional JSON report path")
def batch(directory, mode, pattern, config, output):
    """Batch summarize files under DIRECTORY matching PATTERN."""
    app = TextSummarizerApp(config_file=config)
    batcher = BatchSummarizer(app.config)
    paths = sorted(Path(directory).glob(pattern))
    texts = [app.file_handler.read_file(str(path)) for path in paths]
    results = batcher.summarize_batch(texts, mode=mode)
    report = batcher.get_batch_report()
    console.print(Table.grid().add_row(f"Processed {report['items']} files"))
    console.print(report)
    if output:
        app.file_handler.write_json(output, {"report": report, "results": results})

@cli.command()
@click.option("--config", type=click.Path(exists=True, dir_okay=False))
def config(config):
    """Display current configuration values."""
    cfg = EnhancedConfig(config_file=config) if config else EnhancedConfig()
    cfg.display()
```

---

## Challenge Guidance

- **MultiDocumentSummarizer** – Use `BatchSummarizer` to generate individual summaries, then prompt the model with those summaries to extract themes: *"Given the following bullet summaries, list common topics and discrepancies."* Construct a comparison matrix via `rich.Table` and persist results with `FileHandler`.
- **MeetingSummarizer** – Split transcript by speaker tags, summarise per speaker using `SummarizerEngine.summarize`, and collate action items with a targeted prompt: *"Extract action items with assignee and due date."* Combine everything into a Markdown minutes template.
- **ResearchPaperSummarizer** – Require structured prompts per section: *"Summarise the METHOD section focusing on datasets and evaluation."* Use the file handler to output JSON with keys like `abstract_summary`, `key_findings`, and `related_workComparison`.

These implementations align the Week 1 capstone with production patterns (configurable, testable, observable) while keeping the code approachable for learners.
