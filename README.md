# Final-Project

### Project members

- Jason W Park
- Di (Woody) Wu 
- Kiyoshi Watanabe

### Project Structure

```bash
project/
  data/            # gitignored (raw + processed)
  models/          # saved artifacts (checkpoints), gitignored
  configs/         # YAML/JSON configs for runs (committed)
  notebooks/       # exploration only (committed, but no core logic)
  src/             # all reusable code (committed)
  results/         # tables/figures (committed small), big runs gitignored
  scratch/         # Shares and junks
  scripts/         # thin CLI wrappers / bash entrypoints (committed)
  README.md
  environment.yml  # or pyproject.toml
  .gitignore
```
