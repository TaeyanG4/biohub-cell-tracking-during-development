import zipfile
import pandas as pd
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

with zipfile.ZipFile('tmp_lb/biohub-cell-tracking-during-development.zip') as z:
    name = z.namelist()[0]
    df = pd.read_csv(z.open(name))

print(f'Total teams: {len(df)}')
print('\nTop 30:')
print(df[['Rank', 'TeamName', 'Score']].head(30).to_string())

n_teams = len(df)
# Kaggle standard medal formula:
# Gold: Top 10 + 0.2% of teams (rounded)
# Silver: Top 5% of teams (if > 1000, capped at 50; if 500-1000: 5%)
# Bronze: Top 10% of teams (if > 1000, capped at 100; if 500-1000: 10%)
gold_rank = max(10, 10 + int(round(n_teams * 0.002)))
silver_rank = max(int(round(n_teams * 0.05)), 10)
bronze_rank = max(int(round(n_teams * 0.10)), 10)

print(f'\nTotal teams on Leaderboard: {n_teams}')
print(f'Gold medal threshold (Rank <= {gold_rank}): Score >= {df.iloc[gold_rank-1]["Score"]:.5f}')
print(f'Silver medal threshold (Rank <= {silver_rank}): Score >= {df.iloc[silver_rank-1]["Score"]:.5f}')
print(f'Bronze medal threshold (Rank <= {bronze_rank}): Score >= {df.iloc[bronze_rank-1]["Score"]:.5f}')

# Check score distribution
scores = df['Score'].dropna()
print('\nScore quantiles:')
print(scores.describe())

# Find where 0.946, 0.947, 0.950, 0.960, 0.970 fall:
for target_score in [0.946, 0.947, 0.948, 0.949, 0.950, 0.955, 0.960, 0.965, 0.970]:
    matching = df[df['Score'] >= target_score]
    if len(matching) > 0:
        worst_rank = matching['Rank'].max()
        print(f'Score >= {target_score:.3f}: {len(matching)} teams, lowest rank = {worst_rank}')
    else:
        print(f'Score >= {target_score:.3f}: 0 teams')

print('\nSearching for user team...')
# Search by submission ref or name
for col in df.columns:
    print(col)
