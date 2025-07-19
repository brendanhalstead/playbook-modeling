import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to dataset
DATA_PATH = 'final_merged_dataset_interpolated.csv'

# Check dataset exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please ensure the file is available there.")

# Load dataset
df = pd.read_csv(DATA_PATH)

# --- Step 1: Compute corporate hires, researchers, and AGI researchers ---
df = df.sort_values(['Organization', 'Year'])
# Yearly hires per organization
df['Hires'] = df.groupby('Organization')['Employees'].diff().fillna(df['Employees'])

# Corporate researcher fraction: 80% up to 2022, then 50%
df['Researcher_Fraction'] = df['Year'].apply(lambda y: 0.8 if y <= 2022 else 0.5)
# Researchers hires and cumulative corporate researchers
df['Researcher_Hires'] = df['Hires'] * df['Researcher_Fraction']
df['Researchers_Corp'] = df.groupby('Organization')['Researcher_Hires'].cumsum()

# AGI relevance fraction by org-year
def agi_frac(org, year):
    if org == 'DeepMind': return 0.33 if year <= 2021 else 0.45 if year <= 2022 else 0.60
    if org == 'OpenAI':   return 0.50 if year <= 2019 else 0.65 if year <= 2021 else 0.80
    if org == 'Anthropic': return 0.90
    if org == 'DeepSeek':  return 0.80
    if org == 'xAI':       return 0.90
    return 0.0

# df['AGI_Researcher_Increase'] = df.apply(lambda r: r['Researcher_Hires'] * agi_frac(r['Organization'], r['Year']), axis=1)
# df['AGI_Researchers_Corp'] = df.groupby('Organization')['AGI_Researcher_Increase'].cumsum()

df['AGI_Researchers_Corp'] = df.apply(lambda r: r['Researchers_Corp'] * agi_frac(r['Organization'], r['Year']), axis=1)
df['AGI_Researcher_Increase'] = df.groupby('Organization')['AGI_Researchers_Corp'].diff().fillna(df['AGI_Researchers_Corp'])

# --- Step 2: Aggregate corporate totals by year ---
agg = df.groupby('Year', as_index=False).agg({
    'Employees': 'sum',
    'Researchers_Corp': 'sum',
    'AGI_Researchers_Corp': 'sum'
}).sort_values('Year')

# --- Step 4: External AGI researchers (30% of corporate AGI hires each year) ---
# Aggregate corporate AGI hires by year from the original hires data
df_yearly_agi_hires = df.groupby('Year', as_index=False)['AGI_Researcher_Increase'].sum()
# Compute other AGI researchers as 30% of these annual corporate AGI hires
df_yearly_agi_hires['Other_AGI_Researchers'] = (0.3 * df_yearly_agi_hires['AGI_Researcher_Increase']).cumsum()
# Merge into agg
agg = agg.merge(df_yearly_agi_hires[['Year', 'Other_AGI_Researchers']], on='Year')
# Total AGI researchers including corporate, academia, and others
agg['AGI_Researchers_All_Companies'] = agg['AGI_Researchers_Corp'] + agg['Other_AGI_Researchers']

# --- Step 3: Academic AGI researchers ---
initial_acad = agg.loc[agg['Year'] == 2011, 'AGI_Researchers_All_Companies'].iloc[0] * 2
agg['Academic_AGI'] = initial_acad * (1.25 ** (agg['Year'] - 2011))

# Combine series
agg['AGI_Researchers_Total'] = agg['AGI_Researchers_All_Companies'] + agg['Academic_AGI']

# --- Step 4.5: Productivity-adjusted AGI researchers ---
yearly_hires = pd.DataFrame(index=agg['Year'])

# Corporate AGI hires
corp_hires = df.groupby('Year')['AGI_Researcher_Increase'].sum()
yearly_hires['Corp_AGI_Hires'] = corp_hires

# Other AGI hires (non-cumulative)
yearly_hires['Other_AGI_Hires'] = 0.3 * yearly_hires['Corp_AGI_Hires']

# Academic AGI hires
agg_indexed = agg.set_index('Year')
yearly_hires['Academic_AGI_Hires'] = agg_indexed['Academic_AGI'].diff().fillna(agg_indexed['Academic_AGI'])

yearly_hires = yearly_hires.fillna(0)
yearly_hires['Total_AGI_Hires'] = yearly_hires.sum(axis=1)

# Calculate productivity-adjusted workforce
start_year = yearly_hires.index.min()

# Base productivity for cohort hired in year `y` (decreases by 5% each year, starting from 2016)
cohort_base_productivity = {year: 0.95**max(0, year - 2015) for year in yearly_hires.index}

pa_workforce = []
for t in yearly_hires.index:
    effective_workforce_t = 0
    for y in range(start_year, t + 1):
        hires_in_y = yearly_hires.loc[y, 'Total_AGI_Hires']
        # Productivity of cohort from year y, in year t (increases by 15% each year after hiring)
        productivity_y_at_t = cohort_base_productivity[y] * (1.15**(t - y))
        effective_workforce_t += hires_in_y * productivity_y_at_t
    pa_workforce.append(effective_workforce_t)

agg['PA_AGI_Researchers_Total'] = pa_workforce

# Initialize cohort productivity
yearly_hires = yearly_hires.sort_index()
yearly_hires['Cohort_Productivity'] = 1.0
for i in range(1, len(yearly_hires)):
    yearly_hires.iloc[i, yearly_hires.columns.get_loc('Cohort_Productivity')] = yearly_hires.iloc[i-1, yearly_hires.columns.get_loc('Cohort_Productivity')] * 0.95

# Compute effective employees
effective = []
for year, row in yearly_hires.iterrows():
    eff = 0.0
    for hire_year, hire_row in yearly_hires.iterrows():
        if hire_year <= year:
            age = year - hire_year
            p0 = hire_row['Cohort_Productivity']
            eff += hire_row['Total_AGI_Hires'] * p0 * (1.15 ** age)
    effective.append(eff)
agg['Effective_Employees'] = effective


# --- Step 5: Compute CAGRs ---
def compute_cagr(start, end, years):
    return (end / start) ** (1 / years) - 1 if years > 0 else float('nan')

periods = [(2011, 2024), (2011, 2022), (2022, 2024)]
series = [
    ('Total Employees (companies in data)', 'Employees'),
    ('Total Researchers (companies in data)', 'Researchers_Corp'),
    ('Total AGI Researchers (companies in data)', 'AGI_Researchers_Corp'),
    ('Total AGI Researchers (all companies)', 'AGI_Researchers_All_Companies'),
    ('Total AGI Reserarchers (incl acad)', 'AGI_Researchers_Total'),
    ('Productivity-Adjusted AGI Researchers', 'PA_AGI_Researchers_Total'),
    ('Effective Employees', 'Effective_Employees'),
]
cagr_results = []
for label, col in series:
    for start, end in periods:
        start_val = agg.loc[agg['Year'] == start, col].iloc[0]
        end_val = agg.loc[agg['Year'] == end, col].iloc[0]
        yrs = end - start
        cagr_results.append({
            'Series': label,
            'Period': f"{start}-{end}",
            'CAGR': round(compute_cagr(start_val, end_val, yrs), 4)
        })
cagr_df = pd.DataFrame(cagr_results)

# --- Save DFs to CSV ---
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
agg.to_csv(os.path.join(output_dir, 'agg_df.csv'), index=False)
df.to_csv(os.path.join(output_dir, 'full_df.csv'), index=False)
cagr_df.to_csv(os.path.join(output_dir, 'cagr_df.csv'), index=False)


# --- Step 7: Future Projections ---
def get_growth_rate(year):
    if year <= 2028: return 0.80
    if year == 2029: return 0.70
    if year == 2030: return 0.60
    if year == 2031: return 0.50
    if year == 2032: return 0.40
    if year <= 2036: return 0.30
    if year <= 2040: return 0.20
    if year <= 2060: return 0.10
    if year <= 2080: return 0.05
    if year <= 2100: return 0.03
    return 0.01 # 2101 onward

future_years = range(2025, 2151)
projection_df = pd.DataFrame(index=future_years)

# Project AGI_Researchers_Total
last_researchers = agg.loc[agg['Year'] == 2024, 'AGI_Researchers_Total'].iloc[0]
projected_researchers = []
for year in future_years:
    growth_rate = get_growth_rate(year)
    last_researchers *= (1 + growth_rate)
    projected_researchers.append(last_researchers)

projection_df['AGI_Researchers_Total'] = projected_researchers

# Combine historical and projected data
full_projection = pd.concat([
    agg.set_index('Year')[['AGI_Researchers_Total']],
    projection_df[['AGI_Researchers_Total']]
])
full_projection = full_projection.sort_index()

# Calculate Total_AGI_Hires for all years
full_projection['Total_AGI_Hires'] = full_projection['AGI_Researchers_Total'].diff().fillna(
    full_projection['AGI_Researchers_Total'].iloc[0]
)

# Calculate Productivity-Adjusted AGI Researchers for all years
start_year = full_projection.index.min()
cohort_base_productivity = {year: 0.95**max(0, year - 2015) for year in full_projection.index}
pa_workforce = []
for t in full_projection.index:
    effective_workforce_t = 0
    for y in range(start_year, t + 1):
        hires_in_y = full_projection.loc[y, 'Total_AGI_Hires']
        productivity_y_at_t = cohort_base_productivity[y] * (1.15**(t - y))
        effective_workforce_t += hires_in_y * productivity_y_at_t
    pa_workforce.append(effective_workforce_t)

full_projection['PA_AGI_Researchers_Total'] = pa_workforce

# Calculate growth rate of PA_AGI_Researchers_Total
full_projection['PA_AGI_Researchers_Growth_Rate'] = full_projection['PA_AGI_Researchers_Total'].pct_change()

# Save projection to CSV
projection_output_path = os.path.join(output_dir, 'labor_growth_projection.csv')
full_projection.to_csv(projection_output_path)


# --- Step 8: Output results ---
print("\n=== Combined Headcounts by Year ===")
print(agg[['Year', 'Employees', 'AGI_Researchers_All_Companies', 'Other_AGI_Researchers', 'AGI_Researchers_Total', 'PA_AGI_Researchers_Total', 'Effective_Employees']])

print("\n=== CAGR Comparison ===")
print(cagr_df)

print(f"\nProjection data saved to {projection_output_path}")
print(full_projection.tail())

print("\n=== Projected Total AGI Researchers at Key Milestones ===")
key_years = [2028, 2029, 2030, 2031, 2032, 2036, 2040, 2061]
for year in range(2081, 2151, 20):
    key_years.append(year)

for year in key_years:
    if year in full_projection.index:
        researchers = full_projection.loc[year, 'AGI_Researchers_Total']
        print(f"Year {year}: {researchers:,.0f} Total AGI Researchers")


# --- Step 9: Plotting ---
plt.figure(figsize=(12, 7))

# Plot historical data
plt.plot(agg['Year'], agg['AGI_Researchers_Total'], 'o-', label='Historical Total AGI Researchers (incl acad)', color='blue')
plt.plot(agg['Year'], agg['PA_AGI_Researchers_Total'], 's-', label='Historical Productivity-Adjusted AGI Researchers', color='green')

# Plot projected data
future_projection = full_projection.loc[full_projection.index >= 2024]
plt.plot(future_projection.index, future_projection['AGI_Researchers_Total'], '--', label='Projected Total AGI Researchers', color='blue')
plt.plot(future_projection.index, future_projection['PA_AGI_Researchers_Total'], '--', label='Projected Productivity-Adjusted AGI Researchers', color='green')

plt.yscale('log')
plt.xlabel('Year')
plt.ylabel('Number of Researchers (log scale)')
plt.title('Historical and Projected AGI Researcher Growth (2011–2150)')
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
projection_plot_path = os.path.join(output_dir, 'projected_labor_growth.png')
plt.savefig(projection_plot_path)
plt.show()

print(f"Projection plot saved to {projection_plot_path}")

plt.figure(figsize=(10, 6))
for col, label in [
    # ('Employees', 'Total Employees'),
    # ('Researchers_All', 'Researchers (incl acad)'),
    # ('AGI_Researchers_All', 'Total AGI Researchers (incl acad)'),
    # ('AGI_Researchers_Total', 'AGI Researchers + Others')
    ('Effective_Employees', 'Effective Employees'),
    ('PA_AGI_Researchers_Total', 'Productivity-Adjusted AGI Researchers'),
    ('AGI_Researchers_Total', 'Total AGI Researchers (incl acad)'),
    ('AGI_Researchers_All_Companies', 'Total AGI Researchers (all companies)'),
    ('AGI_Researchers_Corp', 'Total AGI Researchers (companies in data)'),
    ('Researchers_Corp', 'Total Researchers (companies in data)'),
    ('Employees', 'Total Employees (companies in data)'),
]:
    plt.plot(agg['Year'], agg[col], marker='o', label=label)
plt.yscale('log')
plt.xlabel('Year')
plt.ylabel('Headcount (log scale)')
plt.title('Employee and Researcher Growth Comparison (2011–2024)')
plt.legend()
plt.grid(True, ls='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'employee_researcher_growth.png'))
plt.show()