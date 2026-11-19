# Project Purpose and Working Principles

## Purpose

This project is for analyzing National Hockey League (NHL) data obtained through NHL APIs. Its overall goal is to turn game, team, and player data into clear, reproducible analysis that helps answer hockey questions.

## Keep work aligned with the purpose

- Prefer work that improves access to NHL data, data quality, analysis, or communication of findings.
- Keep the path from source data to result understandable: document relevant API endpoints, parameters, assumptions, and transformations.
- Preserve useful source identifiers and dates so records can be traced back to the NHL data they came from.
- Treat seasons, game dates, team and player identifiers, and home/away context consistently. Make time zone and season-boundary assumptions explicit when they affect results.
- Handle missing, delayed, or changed API fields explicitly. Do not silently invent values or present estimates as source facts.
- Make analyses reproducible where practical, and label derived metrics with their definitions.
- Present results in hockey terms that readers can interpret, with units, time periods, and comparison groups stated clearly.

## Repository guidance

- Follow the existing project structure and conventions as the repository grows.
- Keep API access, data preparation, analysis, and presentation responsibilities clear and separated where practical.
- Avoid committing credentials, access tokens, or unnecessary copies of raw data.
- When adding dependencies or changing data assumptions, document why they are needed and how they affect the analysis.

## Coding principles

- Prefer simple, readable code with clear names and small functions that do one thing.
- Follow the language and formatting conventions already used in the repository. Avoid broad rewrites when a focused change will do.
- Validate inputs at boundaries, handle expected failures explicitly, and provide useful error messages. Do not hide failures behind broad exception handling or silent fallbacks.
- Keep configuration and secrets out of source code. Use the project's established configuration pattern and provide safe examples where needed.
- Make changes easy to review: keep scope focused, remove dead code introduced by the change, and document non-obvious decisions.
- Add or update checks when behavior changes, following the project's existing test and validation conventions. Do not claim something was verified unless it was run.

## Data principles

- Treat source data as immutable; write cleaned or derived data to a separate, clearly identified output.
- Validate schemas, types, uniqueness, expected ranges, and join cardinality before relying on a dataset. Report unexpected records instead of silently dropping them.
- Make missing-value handling, deduplication, filtering, and aggregation rules explicit. Preserve the distinction between zero, unknown, and not applicable.
- Use stable NHL identifiers for joins where available; do not rely on display names as keys when IDs exist.
- Prevent leakage across games or seasons when constructing time-based features or comparisons. Respect when each value would have been known.
- Keep units, time zones, season definitions, and game-state context consistent through transformations. Record material changes to data or assumptions.
- Make derived tables and metrics traceable to their inputs and transformation steps. Avoid unnecessary duplication of raw API responses.

## Statistical and machine learning principles

- Start with a clearly stated hockey question and a simple baseline before introducing more complex methods.
- Choose validation that matches the intended use. For future-game predictions, split by time or season; do not randomly mix future and past observations across splits.
- Fit preprocessing, feature selection, and tuning only on training data, then evaluate on held-out data. Keep related observations together when they could otherwise leak across splits.
- Check for target leakage, survivorship bias, class imbalance, selection effects, and changes in NHL rules or data collection that can distort results.
- Compare models with appropriate baselines and metrics. Explain what the metric means for the hockey question, and include uncertainty or variation where practical.
- Distinguish prediction from causation. Describe limitations and avoid claims broader than the data and validation support.
- Keep experiments reproducible: record data periods, feature definitions, split strategy, model settings, and random seeds when applicable.
- Prefer interpretable outputs and inspect errors across relevant groups, such as seasons, teams, or game situations, while avoiding unsupported conclusions from small samples.

If the project documentation establishes a more specific goal or convention, use it to refine this guidance while keeping the overall NHL data analysis purpose intact.
