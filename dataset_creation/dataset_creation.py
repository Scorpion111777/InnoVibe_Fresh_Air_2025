# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

np.random.seed(42)

num_rows = 2000

data = {
    "id": np.arange(1, num_rows + 1),
    "name": [f"Location {i}" for i in range(1, num_rows + 1)],
    "region": np.random.randint(1, 27, num_rows),
    "type": np.random.randint(1, 6, num_rows),
    "area_sq_km": np.round(np.random.uniform(1.0, 50.0, num_rows), 2),
    "damage_percent": np.round(np.random.uniform(30, 100, num_rows), 1),
    "type_of_damage": np.random.randint(1, 8, num_rows),
    "new_type": np.random.randint(1, 6, num_rows),
    "budget_million_usd": np.round(np.random.uniform(10, 500, num_rows), 2),
    "success_rate_percent": np.round(np.random.uniform(50, 95, num_rows), 1),
    "recovery_priority": np.random.randint(1, 4, num_rows),
    "duration": np.random.randint(1, 701, num_rows),
    "funding_source": np.random.randint(1, 7, num_rows)
}

df = pd.DataFrame(data)

success_rates = []
for _, row in df.iterrows():
    # Rule 1: Higher budget → Higher success
    base_success = np.interp(row["budget_million_usd"], [10, 500], [50, 95])

    # Rule 2: Higher damage → Lower success
    damage_impact = np.interp(row["damage_percent"], [30, 100], [10, -10])

    # Rule 3: Same type retained & low damage → Bonus
    type_bonus = 5 if row["type"] == row["new_type"] and row["damage_percent"] < 50 else 0

    # Rule 4: Higher priority (1 is highest) → More success
    priority_bonus = {1: 5, 2: 2, 3: 0}.get(row["recovery_priority"], 0)

    # Rule 5: Longer duration may improve outcome
    duration_bonus = np.interp(row["duration"], [1, 700], [-5, 5])

    # Rule 6: Area too large may reduce effectiveness
    area_penalty = -3 if row["area_sq_km"] > 40 else 0

    # Rule 7: High funding source index may indicate complexity → small penalty
    funding_penalty = -2 if row["funding_source"] >= 5 else 0

    # Rule 8: Severe type_of_damage → lower success
    damage_type_penalty = -4 if row["type_of_damage"] in [6, 7] else 0

    # Rule 9: Small area and low damage → high success potential
    area_bonus = 4 if row["area_sq_km"] < 10 and row["damage_percent"] < 40 else 0

    # Rule 10: Region-based minor variance
    region_variance = (row["region"] % 5) - 2  # Range -2 to +2

    # Aggregate fuzzy score
    success_rate = (base_success + damage_impact + type_bonus + priority_bonus +
                    duration_bonus + area_penalty + funding_penalty +
                    damage_type_penalty + area_bonus + region_variance)

    success_rate = np.clip(success_rate, 50, 95)
    success_rates.append(round(success_rate, 1))

df["success_rate_percent"] = success_rates
df.head()
csv_filename = "ukraine_recovery_data.csv"
df.to_csv(csv_filename, index=False)
csv_filename