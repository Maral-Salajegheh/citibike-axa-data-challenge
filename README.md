# CitiBike Insurance Risk Analysis

This proposal presents a data-driven framework for an insurance partnership between AXA and Citi Bike, based on analysis of Citi Bike trip data and optional integration with NYPD crash proximity data.

This analysis demonstrates that actionable insurance insights can be derived from **two complementary approaches**:

- **Full model (NYC):** Combining trip exposure with crash proximity data for risk-adjusted station scoring
- **Exposure-only model (Jersey City):** Using trip volume patterns alone when crash data is unavailable

> **Key Feature:** This pipeline is fully reproducible for **any year**, **any months**, and **any proximity radius** — making it portable to other bike-share systems worldwide.

---



### Installation & Run

## Dependencies

### Core Requirements

This project uses **Python 3.10+** and 20 direct dependencies:

**Data Processing** (1):
- `pandas` - DataFrame operations
- `numpy` - Numerical computing
- `pyarrow` - Fast Parquet I/O

**Analysis** (2):
- `scipy` - Scientific computing
- `scikit-learn` - Machine learning (DBSCAN clustering)
- `statsmodels` - Statistical modeling

**Visualization** (3):
- `matplotlib` - Charts and plots

**Jupyter Ecosystem** (4):
- `jupyter` - Interactive analysis
- `jupyterlab` - Modern notebook interface
- `nbconvert` - HTML report generation
- `ipykernel` - Jupyter kernel

**Utilities** (5):
- `tqdm` - Progress bars
- `requests` - HTTP downloads
- `beautifulsoup4` + `lxml` - Table extraction
- `openpyxl` - Excel file support

### Installation

```bash
# The Makefile handles everything
make setup

# Or manually:
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Note**: `requirements.txt` lists packages you directly use.

## Run

```bash
# 1. Clone the repository
git clone https://github.com/Maral-Salajegheh/citibike-axa-data-challenge.git
cd repo root

# 2. Download NYPD crash data more information is given 
curl -L "https://data.cityofnewyork.us/api/views/h9gi-nx95/rows.csv?accessType=DOWNLOAD" -o h9gi-nx95_full.csv
# Save to: your repo root/data/raw/nypd/

# 3. Run the complete pipeline (downloads data, analyzes, generates reports) you can run the whole pipline for any year, month abd raduis 
make all-both YEARS="YYYY YYYY" MONTHS="1 2 ... 12" RADII_M="Any Raduis" AXA_RADIUS="Any Raduism"
Or you can fixe Raiduis in makefile then run 
make all-both YEARS="2017 2023" MONTHS="1 2" 
NYC or Jersey City
make all MODE=nyc    # or MODE=jc



# 4. View the report
firefox reports/latest_nyc/06_insurer_story.executed.html
# or: open reports/latest_nyc/06_insurer_story.executed.html
```

**That's it!** The Makefile orchestrates everything:
-  Creates virtual environment
-  Installs dependencies  
-  Downloads CitiBike data (incremental)
-  Converts to Parquet
-  Filters crash data
-  Calculates risk scores
-  Generates HTML reports


---

## Project Structure

```
citibike-insurance-challenge/
├──  data/
│   ├── raw/
│   │   ├── citibike/                    # Downloaded ZIP files (by run)
│   │   └── nypd/                        # NYPD crash CSV (you provide)
│   └── processed/
│       ├── citibike_parquet/            # Optimized trip data
│       └── nypd_crashes_*.csv           # Filtered crashes
│
├──  summaries/                         # Analysis outputs
│   ├── yYYYY.._m1_2_3..._modenyc/       # Per-run summaries
│   │   ├── citibike_trips_by_*.csv      # Usage patterns
│   │   ├── axa_partner_scorecard_500m.csv     #  Risk scores
│   │   └── axa_target_windows_*.csv     # Business recommendations
│   ├── latest_nyc/                      # → Symlink to latest NYC
│   └── _compare/                        # Multi-run comparisons
│
├──  reports/                           # HTML reports & charts
│   ├── yYYYY_m1_2_3_modenyc/
│   │   ├── 06_insurer_story.executed.html     #  Main report
│   │   ├── 07_risk_deep_dive.executed.html    # Technical analysis
│   │   ├── figures/                     # All charts (PNG)
│   │   └── The Same for JC
│   └── latest_nyc/                      # → Symlink to latest
│   └── latest_JC/                      # → Symlink to latest
│
├──  notebooks/
│   ├── 06_insurer_story.ipynb           # Business analysis
│   └── 07_risk_deep_dive.ipynb          # Technical deep dive
│
├──  scripts/                           # Pipeline automation
│   ├── download_tripdata.py             # Download CitiBike data
│   ├── filter_nypd_crashes.py           # Filter crash data
│   ├── summarize_citibike_usage.py      # Usage summaries
│   ├── build_axa_scorecard.py           # Risk scoring (EB)
│   ├── build_axa_target_windows.py      # Business recommendations
│   └── aggregate_usage_summaries.py     # Multi-year comparison
│
├──  src/
│   └── ingest_tripdata.py               # ZIP → Parquet conversion
│
├──   Makefile                          # Pipeline orchestration
├──  requirements.txt                   # Python dependencies
└──  README.md                          # This file
```

---

##  Key Features

### 1. **Reproducible Pipeline** 

```bash
# Single command = complete analysis
make all-both YEARS="YYYY YYYY" MONTHS="1 2 ... 12" 

# Automatic dependency tracking (only re-runs what changed)
# Incremental downloads (skips existing files)
# Per-run isolation (never overwrites old results)
# 

### 3. **Business-Ready Outputs** 

- **Scorecards**: `axa_partner_scorecard_500m.csv` (station risk percentiles)
- **Target Windows**: `axa_target_windows_*.csv` (time-based opportunities)
- **Executive Tables**: Excel-ready tables for PowerPoint
- **Interactive Reports**: HTML report + figures folder
```
### 4. **Flexible Analysis** 

```bash
# NYC or Jersey City
make all MODE=nyc    # or MODE=jc

# Any time period
make all YEARS="2017 2018 2019" MONTHS="1 2 3"  # Q1 2017-2019
make all YEARS="2024" MONTHS="6 7 8"             # Summer 2024

# Both modes simultaneously
make all-both YEARS="2023 2024" MONTHS="1 2 3"

# Compare across all runs
make compare-years
```

### **Two Analysis Modes** 

| Mode | Crash Data | Risk Scoring | Use Case |
|------|------------|--------------|----------|
| **NYC** |  NYPD data | Full EB model | Risk-adjusted pricing |
| **Jersey City** | Not available | Exposure-only | Volume-based targeting |

> **Note:** NYPD crash data covers NYC only. The Jersey City analysis demonstrates that the framework produces actionable insights even without crash data — important for portability to other bike-share systems.

---

##  Sample Results (NYC 2019-2025) This code has been running for years the numbers might differ based on what you run.

| Metric | Value | Insight |
|--------|-------|---------|
| **Total Trips** | 121M+ | Across all analyzed years |
| **2025 Annual Trips** | 45.8M | +123% growth since 2019 |
| **Credible Stations** | 16,454 | ≥5,000 trips (reliable estimates) |
| **High-Risk + High-Exposure** | 2,676 | Prevention hotspots |
| **Risk Trend** | −67% | Per-trip risk declining (safer system) |

**Geographic Patterns:**
-  Highest risk: Midtown Manhattan (Times Square, Port Authority)
-  Lowest risk: Outer boroughs, residential areas
-  Clustering: High-risk stations near major transit hubs

**Temporal Patterns:**
-  Peak months: June–October (54% of annual trips)
-  Peak days: Tuesday–Wednesday (15–16% each)
-  September highest single month (11.5%)

---

## Methodology

### Risk Proxy

**Hypothesis**: Stations near frequent crashes have higher accident risk for cyclists.

**Method**:
1. Load NYPD crash reports (cyclist injuries/fatalities)
2. For each station, count crashes within configurable radius (default: 500m)
3. Calculate raw rate: `crashes / trips × 100,000`
4. Apply **Empirical Bayes shrinkage** toward global mean
5. Assign **credibility flags** based on sample size

---

### Empirical Bayes Formula

```
EB_rate = (α₀ + crashes) / (α₀ + β₀ + trips) × 100,000

where:
  α₀, β₀ = Prior parameters (fitted from data via method of moments)
  
Interpretation:
  - Small samples → pulled toward global average
  - Large samples → stay close to raw rate
  - Prevents unreliable stations from dominating rankings
```

**Example**:
```
Station A: 2 crashes, 100 trips     → Raw rate = 2,000 per 100k
Station B: 20 crashes, 100,000 trips → Raw rate = 20 per 100k

After EB smoothing:
Station A: EB rate ≈ 50 per 100k   (shrunk heavily toward mean)
Station B: EB rate ≈ 18 per 100k   (stays close to raw rate)
```

### Credibility Threshold

- **Credible**: ≥5,000 trips → Included in rankings
- **Insufficient data**: <5,000 trips → Flagged, excluded from percentiles
- **Rationale**: Prevents low-volume stations from skewing business decisions

### Business Logic

**Prevention Hotspots** = High risk (≥90th percentile) + High exposure (≥75th percentile)  
→ Focus safety campaigns here

**Product Hotspots** = High exposure (≥75th percentile), any risk  
→ Target for insurance sales

**Acquisition Hotspots** = Low risk (≤10th percentile) + High exposure  
→ Attract risk-averse customers with discounts

---

##  Usage Examples

### Common Workflows

```bash
# Full year analysis
make all-both YEARS="YYYY YYYY..." MONTHS="1 2 .. 12" 

# Just regenerate reports If you already did the Full year analysis and just made changes in notebooks
make report-both YEARS="YYYY YYYY..." MONTHS="1 2 .. 12" 

# Re-run notebooks with updated data !! If you already did the Full year analysis and just made changes in notebooks
make run-notebooks MODE=nyc YEARS="YYYY YYYY..." MONTHS="1 2 .. 12" 

# Build just the scorecard
make axa-scorecard MODE=nyc YEARS="YYYY YYYY..." MONTHS="1 2 .. 12" 

# Compare all completed runs
make compare-years YEARS="YYYY YYYY..." MONTHS="1 2 .. 12" 


##  Key Output Files

### 1. **AXA Partner Scorecard** (`axa_partner_scorecard_500m.csv`)

Station-level risk assessment with business flags:

| Column | Description |
|--------|-------------|
| `start_station_id` | Unique station identifier |
| `start_station_name` | Human-readable name |
| `latitude`, `longitude` | Geographic coordinates |
| `exposure_trips` | Total trips from station |
| `nearby_crashes_500m` | Crashes within 500m |
| `raw_risk_rate_per_100k_trips` | Unadjusted rate |
| `eb_risk_rate_per_100k_trips` | **Empirical Bayes adjusted**  |
| `risk_index_pct` | Percentile rank (0-100) |
| `credibility_flag` | `credible` or `insufficient_data` |
| `prevention_hotspot` | Boolean: High risk + High exposure |
| `product_hotspot` | Boolean: High exposure |
| `acquisition_hotspot` | Boolean: Low risk + High exposure |


### 3. **Usage Summaries**

- `citibike_trips_by_year.csv` - Annual totals, YoY growth
- `citibike_trips_by_month.csv` - Monthly patterns, seasonality
- `citibike_trips_by_dow.csv` - Weekday vs weekend
- `citibike_trips_by_hour.csv` - Hour-of-day patterns

---

##  NYPD Data Setup

**Required**: Download the NYPD crash dataset before running the pipeline.

### Option 1: NYC Open Data Portal (Recommended)

1. Visit: https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95
2. Click **Export** → **CSV** (full dataset)
3. Save as: `data/raw/nypd/h9gi-nx95_full.csv`

**File size**: ~600MB (2M+ records)  
**Time**: ~5 minutes to download

### Verify Download

```bash
# Check file exists and has data
wc -l data/raw/nypd/h9gi-nx95_full.csv
# Should show: ~2,000,000 lines

head -5 data/raw/nypd/h9gi-nx95_full.csv
# Should show: CSV header + crash records
```

**Note**: The pipeline automatically filters this to cyclist-involved crashes in your selected time period.

---

## Configuration

### Makefile Variables

Customize the pipeline behavior:

```makefile
YEARS  ?= 2023 2024          # Years to analyze
MONTHS ?= 1 2 3              # Months to include (1=Jan, 12=Dec)
MODE   ?= nyc                # "nyc" or "jc" (Jersey City)
PURGE_OLD_ZIPS ?= ASK        # ZIP handling: YES/NO/ASK
TIMEOUT ?= 600               # Notebook timeout (seconds)
```

Override via command line:
```bash
make all-both YEARS="2024" MONTHS="6 7 8" PURGE_OLD_ZIPS=NO
```

### Environment Variables

Notebooks automatically receive these from the Makefile:

```bash
CITIBIKE_PARQUET_DIR  # Path to trip data
CITIBIKE_RUN_DIR      # Path to summaries
CITIBIKE_MODE         # "nyc" or "jc"
CITIBIKE_YEARS        # Space-separated years
CITIBIKE_MONTHS       # Space-separated months
NYPD_CRASH_CSV        # Path to crash data


-
---

##  Contact

**Maral Salajegheh**  
maral.salajegheh@gmail.com 