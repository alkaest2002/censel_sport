# Data Analysis Pipeline

A two-stage data analysis pipeline for processing and analyzing performance test data with statistical validation, distribution fitting, and visualization capabilities.

Authors: Pierpaolo Calanna, Gaetano Buonaiuto, Marco Silverstrini

## Overview

This project provides a robust pipeline for analyzing performance test data (physical fitness tests) with advanced statistical methods including:
- Distribution fitting
- Bootstrap percentile analysis
- Monte Carlo validation
- Data standardization
- Automated visualization

## Project Structure

The pipeline consists of two main components:

### 1. Outer Loop (`loop_outer.py`)
Validates the raw database file and ensures data integrity before analysis.

### 2. Inner Loop (`loop_inner.py`)
Performs the complete statistical analysis pipeline on validated data.

## Features

- **Data Validation**: Comprehensive checks for data integrity and format
- **Statistical Analysis**: 
  - Theoretical distribution fitting
  - Bootstrap percentile computation
  - Monte Carlo simulation validation
- **Visualization**: Automated plot generation for analysis results
- **Standardization**: Percentile-based score standardization
- **Modular Design**: Separated concerns with dedicated modules for each analysis step

## Requirements

### Python Version
- Python 3.9+ (uses type hints and modern Python features)

## Data Format

The pipeline expects a CSV database (`db/db.csv`) with the following columns:
- `recruitment_year`: Integer (e.g., 2025)
- `recruitment_type`: String (e.g., hd)
- `test`: String (one of: MT100, MT1000, SWIM25, SITUPS, PUSHUPS)
- `gender`: String ("M" or "F")
- `age`: Integer (14-79)
- `value`: Float (non-negative, finite values)
