# Makefile — reproducible Citi Bike + NYPD pipeline (+ stable latest outputs)

SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

VENV ?= .venv
PIP := $(VENV)/bin/pip
PYTHON := $(VENV)/bin/python
JUPYTER := $(VENV)/bin/jupyter

# ---- Reproducibility knobs ----
YEARS  ?= 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
MONTHS ?= 1 2 3 4 5 6 7 8 9 10 11 12
MODE   ?= jc
# Convenience: run for multiple modes with one command
MODES  ?= nyc jc

# ---- Crash radius knobs (NEW) ----
# Radii computed during summarize (comma-separated meters)
RADII_M ?= 500
# Radius chosen by AXA scorecard (e.g., 500m, 750m, 1km, auto)
AXA_RADIUS ?= 500m
# ---- AXA scorecard knobs (forwarded to build_axa_scorecard.py) ----
# Which risk table to score:
#   station_risk_exposure_plus_crashproximity.csv
#   station_risk_exposure_plus_crashproximity_by_year.csv
#   station_risk_exposure_plus_crashproximity_by_year_month.csv
AXA_RISK_FILE ?= station_risk_exposure_plus_crashproximity.csv

# EB grouping scope:
#   mode (default; overall station scorecard per mode)
#   mode_year
#   mode_year_month
AXA_EB_SCOPE ?= mode

# Credibility threshold for ranking
AXA_MIN_TRIPS ?= 5000

# Mode-specific AXA defaults (keeps terminal clean)
ifeq ($(MODE),nyc)
AXA_RISK_FILE := station_risk_exposure_plus_crashproximity_by_year_month.csv
AXA_EB_SCOPE  := mode_year_month
AXA_MIN_TRIPS := 5000
endif

# Pass extra flags to scripts/download_tripdata.py
DOWNLOAD_FLAGS ?=
ifeq ($(MODE),nyc)
DOWNLOAD_FLAGS += --allow-yearly-fallback
endif

# ZIP retention:
#   NO  = keep existing ZIPs (incremental download will still run and skip existing)
#   YES = delete existing ZIPs before downloading
#   ASK = prompt interactively (if no TTY, keeps existing)
PURGE_OLD_ZIPS ?= NO

# Notebook execution settings
TIMEOUT ?= 600
KERNEL  ?= python3

# ---- Tags for filenames/folders (derived) ----
empty :=
space := $(empty) $(empty)
YEARS_TAG  := $(subst $(space),_,$(strip $(YEARS)))
MONTHS_TAG := $(subst $(space),_,$(strip $(MONTHS)))
RUN_TAG    := y$(YEARS_TAG)_m$(MONTHS_TAG)_mode$(MODE)

# ---- Paths (per-run; prevents overwrites) ----
CB_RAW_DIR     ?= data/raw/citibike/$(RUN_TAG)
CB_PARQUET_DIR ?= data/processed/citibike_parquet/$(RUN_TAG)

NYPD_RAW ?= data/raw/nypd/h9gi-nx95_full.csv
# include mode in filename to avoid jc/nyc collisions
NYPD_OUT ?= data/processed/nypd_crashes_y$(YEARS_TAG)_m$(MONTHS_TAG)_mode$(MODE).csv

NB_DIR      ?= notebooks
NB_OUT_DIR  ?= notebooks/executed/$(RUN_TAG)
REPORT_DIR  ?= reports/$(RUN_TAG)

# ---- Summaries (per-run + global compare) ----
SUMMARY_DIR       ?= summaries/$(RUN_TAG)
COMPARE_DIR       ?= summaries/_compare

# Stamps to trigger notebook re-execution when data changes
SUMMARY_STAMP := $(SUMMARY_DIR)/.stamp
AXA_STAMP     := $(SUMMARY_DIR)/.axa.stamp

# ---- Convenience output folders (always overwritten) ----
LATEST_SUMMARY_DIR ?= summaries/latest
LATEST_REPORT_DIR  ?= reports/latest
LATEST_SUMMARY_DIR_MODE ?= summaries/latest_$(MODE)
LATEST_REPORT_DIR_MODE  ?= reports/latest_$(MODE)

## ---- Notebooks to execute ----
NOTEBOOKS ?= \
	$(NB_DIR)/06_insurer_story.ipynb \
	$(NB_DIR)/07_risk_deep_dive.ipynb

EXECUTED_NOTEBOOKS := $(patsubst $(NB_DIR)/%.ipynb,$(NB_OUT_DIR)/%.executed.ipynb,$(NOTEBOOKS))

.DEFAULT_GOAL := help

.PHONY: help show-config setup \
        tripdata ingest nypd summarize summarize-only \
        publish-latest-summary \
        axa-scorecard axa-windows \
        run-notebooks report report-all publish-latest-report open-latest \
        compare-years \
        all all-one all-both summarize-both report-both \
        open-latest-nyc open-latest-jc \
        clean-notebooks clean-report clean-summary clean-compare clean-latest

help:
	@echo ""
	@echo "Targets:"
	@echo "  make setup"
	@echo "  make tripdata        (download Citi Bike zips)"
	@echo "  make ingest          (zip -> parquet)"
	@echo "  make nypd            (filter NYPD crashes for YEARS/MONTHS)"
	@echo "  make summarize       (write per-run usage CSVs + crash proximity if NYPD present)"
	@echo "  make axa-scorecard   (build station-level AXA partner scorecard CSV)"
	@echo "  make axa-windows     (build targeting windows CSV)"
	@echo "  make run-notebooks   (execute notebooks)"
	@echo "  make report          (export executed notebook to HTML)"
	@echo "  make all             (single-mode full pipeline for MODE=$(MODE))"
	@echo "  make all-both        (run full pipeline for MODEs in $(MODES), e.g. nyc + jc)"
	@echo "  make compare-years   (combine per-run summaries into ALL CSVs)"
	@echo ""
	@echo "Convenience (always overwritten):"
	@echo "  summaries/latest/"
	@echo "  reports/latest/ (last run)"
	@echo "  reports/latest_nyc/ and reports/latest_jc/ (mode-specific)"
	@echo "  make open-latest     (prints the latest report path)"
	@echo ""
	@echo "Current run config:"
	@$(MAKE) --no-print-directory show-config

show-config:
	@echo "  YEARS=$(YEARS)"
	@echo "  MONTHS=$(MONTHS)"
	@echo "  MODE=$(MODE)"
	@echo "  RUN_TAG=$(RUN_TAG)"
	@echo "  CB_RAW_DIR=$(CB_RAW_DIR)"
	@echo "  CB_PARQUET_DIR=$(CB_PARQUET_DIR)"
	@echo "  SUMMARY_DIR=$(SUMMARY_DIR)"
	@echo "  NB_OUT_DIR=$(NB_OUT_DIR)"
	@echo "  REPORT_DIR=$(REPORT_DIR)"
	@echo "  NYPD_OUT=$(NYPD_OUT)"
	@echo "  PURGE_OLD_ZIPS=$(PURGE_OLD_ZIPS)"
	@echo "  DOWNLOAD_FLAGS=$(DOWNLOAD_FLAGS)"
	@echo "  RADII_M=$(RADII_M)"
	@echo "  AXA_RADIUS=$(AXA_RADIUS)"
	@echo ""

# ---- Single-mode full pipeline ----
all: all-one

all-one: tripdata ingest nypd summarize axa-scorecard axa-windows run-notebooks report

# ---- Environment setup ----
setup: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	python -m venv "$(VENV)"
	"$(PIP)" install --upgrade pip
	"$(PIP)" install -r requirements.txt
	@touch "$(VENV)/bin/activate"

# ---- Download Citi Bike (incremental by default) ----
tripdata: setup
	mkdir -p "$(CB_RAW_DIR)"
	@if [ -z "$(CB_RAW_DIR)" ] || [ "$(CB_RAW_DIR)" = "/" ]; then \
		echo "Refusing to use CB_RAW_DIR=$(CB_RAW_DIR)"; exit 1; \
	fi

	@has_zips=$$(find "$(CB_RAW_DIR)" -maxdepth 1 -type f -name "*.zip" -print -quit | wc -l); \
	if [ "$(PURGE_OLD_ZIPS)" = "YES" ]; then \
		echo "PURGE_OLD_ZIPS=YES -> deleting existing ZIPs in $(CB_RAW_DIR) ..."; \
		find "$(CB_RAW_DIR)" -maxdepth 1 -type f -name "*.zip" -print -delete || true; \
	elif [ "$(PURGE_OLD_ZIPS)" = "ASK" ] && [ "$$has_zips" -gt 0 ]; then \
		if [ -t 0 ]; then \
			read -r -p "ZIPs exist in $(CB_RAW_DIR). Keep them? (incremental download will skip existing) [Y/n] " ans; \
			if [[ "$$ans" =~ ^[Nn]$$ ]]; then \
				echo "Deleting existing ZIPs in $(CB_RAW_DIR) ..."; \
				find "$(CB_RAW_DIR)" -maxdepth 1 -type f -name "*.zip" -print -delete || true; \
			else \
				echo "Keeping existing ZIPs."; \
			fi; \
		else \
			echo "No interactive TTY; keeping existing ZIPs."; \
		fi; \
	fi

	@echo "Downloading (incremental) into $(CB_RAW_DIR) ..."
	"$(PYTHON)" scripts/download_tripdata.py \
		--years $(YEARS) --months $(MONTHS) --mode $(MODE) --out-dir "$(CB_RAW_DIR)" \
		$(DOWNLOAD_FLAGS)

# ---- Ingest Citi Bike (zip -> parquet) ----
ingest: setup tripdata
	mkdir -p "$(CB_PARQUET_DIR)"
	"$(PYTHON)" src/ingest_tripdata.py \
		--raw-dir "$(CB_RAW_DIR)" \
		--out-dir "$(CB_PARQUET_DIR)" \
		--mode $(MODE) \
		--years $(YEARS) \
		--months $(MONTHS)

# ---- Filter NYPD crashes ----
nypd: setup
	@if [ ! -f "$(NYPD_RAW)" ]; then \
		echo "Missing NYPD raw file: $(NYPD_RAW)"; \
		echo "Put the full export CSV there, then re-run: make nypd"; \
		exit 1; \
	fi
	"$(PYTHON)" scripts/filter_nypd_crashes.py \
		--in-path "$(NYPD_RAW)" \
		--out-path "$(NYPD_OUT)" \
		--years $(YEARS) \
		--months $(MONTHS)

# ---- Per-run summary (usage + optional crash proximity) ----
summarize: ingest
	mkdir -p "$(SUMMARY_DIR)"
	@if [ -f "$(NYPD_OUT)" ]; then \
	  "$(PYTHON)" scripts/summarize_citibike_usage.py \
	    --parquet-dir "$(CB_PARQUET_DIR)" \
	    --out-dir "$(SUMMARY_DIR)" \
	    --mode "$(MODE)" \
	    --nypd-crash-csv "$(NYPD_OUT)" \
	    --radii-m "$(RADII_M)"; \
	else \
	  echo "NOTE: NYPD_OUT not found ($(NYPD_OUT)) -> running usage summaries only"; \
	  "$(PYTHON)" scripts/summarize_citibike_usage.py \
	    --parquet-dir "$(CB_PARQUET_DIR)" \
	    --out-dir "$(SUMMARY_DIR)" \
	    --mode "$(MODE)" \
	    --radii-m "$(RADII_M)"; \
	fi
	@touch "$(SUMMARY_STAMP)"
	@$(MAKE) --no-print-directory publish-latest-summary

# ---- Re-summarize existing parquet (no download/ingest) ----
summarize-only:
	mkdir -p "$(SUMMARY_DIR)"
	@if [ -f "$(NYPD_OUT)" ]; then \
	  "$(PYTHON)" scripts/summarize_citibike_usage.py \
	    --parquet-dir "$(CB_PARQUET_DIR)" \
	    --out-dir "$(SUMMARY_DIR)" \
	    --mode "$(MODE)" \
	    --nypd-crash-csv "$(NYPD_OUT)" \
	    --radii-m "$(RADII_M)"; \
	else \
	  echo "NOTE: NYPD_OUT not found ($(NYPD_OUT)) -> running usage summaries only"; \
	  "$(PYTHON)" scripts/summarize_citibike_usage.py \
	    --parquet-dir "$(CB_PARQUET_DIR)" \
	    --out-dir "$(SUMMARY_DIR)" \
	    --mode "$(MODE)" \
	    --radii-m "$(RADII_M)"; \
	fi
	@touch "$(SUMMARY_STAMP)"
	@$(MAKE) --no-print-directory publish-latest-summary

publish-latest-summary:
	@rm -rf "$(LATEST_SUMMARY_DIR)"
	@mkdir -p "$(LATEST_SUMMARY_DIR)"
	@cp -a "$(SUMMARY_DIR)/." "$(LATEST_SUMMARY_DIR)/"
	@rm -rf "$(LATEST_SUMMARY_DIR_MODE)"
	@mkdir -p "$(LATEST_SUMMARY_DIR_MODE)"
	@cp -a "$(SUMMARY_DIR)/." "$(LATEST_SUMMARY_DIR_MODE)/"
	@echo "Published summaries -> $(LATEST_SUMMARY_DIR) and $(LATEST_SUMMARY_DIR_MODE)"

# ---- AXA assets ----

axa-scorecard: summarize
	@set -euo pipefail; \
	extra=""; \
	if [ -n "$(AXA_M_PRIOR)" ]; then extra="--m-prior $(AXA_M_PRIOR)"; fi; \
	"$(PYTHON)" scripts/build_axa_scorecard.py \
		--in-dir "$(SUMMARY_DIR)" \
		--out-dir "$(SUMMARY_DIR)" \
		--radius "$(AXA_RADIUS)" \
		--risk-file "$(AXA_RISK_FILE)" \
		--eb-scope "$(AXA_EB_SCOPE)" \
		--min-trips "$(AXA_MIN_TRIPS)" \
		$$extra
	@touch "$(AXA_STAMP)"
	@$(MAKE) --no-print-directory publish-latest-summary


axa-windows: summarize
	"$(PYTHON)" scripts/build_axa_target_windows.py --in-dir "$(SUMMARY_DIR)" --out-dir "$(SUMMARY_DIR)"
	@touch "$(AXA_STAMP)"
	@$(MAKE) --no-print-directory publish-latest-summary

# ---- Execute notebooks ----
$(NB_OUT_DIR):
	mkdir -p "$(NB_OUT_DIR)"

run-notebooks: setup $(NB_OUT_DIR) $(EXECUTED_NOTEBOOKS)

# Re-run notebook if notebook changed OR summaries changed (stamp)
$(NB_OUT_DIR)/%.executed.ipynb: $(NB_DIR)/%.ipynb $(SUMMARY_STAMP) $(AXA_STAMP) | $(NB_OUT_DIR)
	CITIBIKE_PARQUET_DIR="$(abspath $(CB_PARQUET_DIR))" \
	CITIBIKE_RUN_DIR="$(abspath $(SUMMARY_DIR))" \
	NYPD_CRASH_CSV="$(abspath $(NYPD_OUT))" \
	CITIBIKE_YEARS="$(YEARS)" \
	CITIBIKE_MONTHS="$(MONTHS)" \
	CITIBIKE_MODE="$(MODE)" \
	AXA_RADIUS="$(AXA_RADIUS)" \
	RADII_M="$(RADII_M)" \
	"$(JUPYTER)" nbconvert --to notebook --execute "$<" \
		--ExecutePreprocessor.timeout="$(TIMEOUT)" \
		--ExecutePreprocessor.kernel_name="$(KERNEL)" \
		--output "$(@F)" \
		--output-dir "$(NB_OUT_DIR)"


# ---- Reports (HTML only) ----
$(REPORT_DIR):
	mkdir -p "$(REPORT_DIR)"

report: run-notebooks | $(REPORT_DIR)
	"$(JUPYTER)" nbconvert --to html "$(NB_OUT_DIR)/06_insurer_story.executed.ipynb" --output-dir "$(REPORT_DIR)"
	"$(JUPYTER)" nbconvert --to html "$(NB_OUT_DIR)/07_risk_deep_dive.executed.ipynb" --output-dir "$(REPORT_DIR)"
	@$(MAKE) --no-print-directory publish-latest-report

report-all: run-notebooks | $(REPORT_DIR)
	"$(JUPYTER)" nbconvert --to html "$(NB_OUT_DIR)/*.executed.ipynb" --output-dir "$(REPORT_DIR)"
	@$(MAKE) --no-print-directory publish-latest-report

publish-latest-report:
	@rm -rf "$(LATEST_REPORT_DIR)"
	@mkdir -p "$(LATEST_REPORT_DIR)"
	@cp -a "$(REPORT_DIR)/." "$(LATEST_REPORT_DIR)/"
	@rm -rf "$(LATEST_REPORT_DIR_MODE)"
	@mkdir -p "$(LATEST_REPORT_DIR_MODE)"
	@cp -a "$(REPORT_DIR)/." "$(LATEST_REPORT_DIR_MODE)/"
	@echo "Published report -> $(LATEST_REPORT_DIR) and $(LATEST_REPORT_DIR_MODE)"

# ---- Compare across all finished runs ----
compare-years:
	mkdir -p "$(COMPARE_DIR)"
	"$(PYTHON)" scripts/aggregate_usage_summaries.py \
		--summaries-root "summaries" \
		--out-dir "$(COMPARE_DIR)"
	@echo ""
	@echo "COMPARE outputs:"
	@echo "  $(COMPARE_DIR)/citibike_trips_by_year_ALL.csv"
	@echo "  $(COMPARE_DIR)/citibike_trips_by_month_ALL.csv"
	@echo "  $(COMPARE_DIR)/citibike_trips_by_monthOfYear_ALL.csv"
	@echo "  $(COMPARE_DIR)/citibike_trips_by_dow_ALL.csv"
	@echo "  $(COMPARE_DIR)/citibike_trips_by_hour_ALL.csv"

# ---- Cleaning ----
clean-notebooks:
	rm -rf "$(NB_OUT_DIR)"

clean-report:
	rm -rf "$(REPORT_DIR)"

clean-summary:
	rm -rf "$(SUMMARY_DIR)"

clean-compare:
	rm -rf "$(COMPARE_DIR)"

clean-latest:
	rm -rf "$(LATEST_SUMMARY_DIR)" "$(LATEST_REPORT_DIR)"

# ---- Multi-mode convenience ----
all-both:
	@set -euo pipefail; \
	for m in $(MODES); do \
		echo "===== Running full pipeline for MODE=$$m ====="; \
		$(MAKE) --no-print-directory all-one MODE=$$m YEARS="$(YEARS)" MONTHS="$(MONTHS)" PURGE_OLD_ZIPS="$(PURGE_OLD_ZIPS)" RADII_M="$(RADII_M)" AXA_RADIUS="$(AXA_RADIUS)"; \
	done; \
	echo "===== Building compare tables across all runs ====="; \
	$(MAKE) --no-print-directory compare-years

summarize-both:
	@set -euo pipefail; \
	for m in $(MODES); do \
		echo "===== Summarize only for MODE=$$m ====="; \
		$(MAKE) --no-print-directory summarize-only MODE=$$m YEARS="$(YEARS)" MONTHS="$(MONTHS)" RADII_M="$(RADII_M)"; \
	done; \
	$(MAKE) --no-print-directory compare-years

report-both:
	@set -euo pipefail; \
	for m in $(MODES); do \
		echo "===== Notebooks + report for MODE=$$m ====="; \
		$(MAKE) --no-print-directory report MODE=$$m YEARS="$(YEARS)" MONTHS="$(MONTHS)" RADII_M="$(RADII_M)" AXA_RADIUS="$(AXA_RADIUS)"; \
	done; \
	echo "===== Building compare tables across all runs ====="; \
	$(MAKE) --no-print-directory compare-years


open-latest-nyc:
	@echo "reports/latest_nyc/06_insurer_story.executed.html"

open-latest-jc:
	@echo "reports/latest_jc/06_insurer_story.executed.html"


