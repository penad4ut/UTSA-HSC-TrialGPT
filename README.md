# UTSA-HSC-TrialGPT  
Locally Deployed Eligibility Evidence Extraction Framework

## Overview
UTSA-HSC-TrialGPT is a re-engineered adaptation of the NCI's TrialGPT work: Jin, Q., Wang, Z., Floudas, C.S. et al. Matching patients to clinical trials with large language models. Nat Commun 15, 9074 (2024). https://doi.org/10.1038/s41467-024-53081-z. This version preserves the core goal of TrialGPT—automated interpretation of clinical trial eligibility criteria—but extends it to operate securely on **locally deployed large language models (LLMs)** for use inside protected health information (PHI) environments.

The system is designed to support clinical trial sites by extracting eligibility-relevant evidence from local EHR-derived documentation for specific studies, enabling accurate and efficient screening workflows.
This repository contains components developed as part of Dr. Mahanaz Syed’s research software engineering (RSE) efforts to re-engineer and adapt AI-based clinical data integration tools for trial operations, addressing key gaps in evidence extraction and workflow feasibility.

## Key Features
- **PHI-safe, fully local LLM inference** (no external API calls)
- **Study-site–focused adaptation** (Trial-centric)
- **Longitudinal document understanding** across multi-encounter notes
- **Structured extraction of eligibility-relevant evidence**
- **Modular architecture** for model/criteria/guardrails customization
- **FHIR®-compatible outputs** for integration with FETCH and clinical trial workflows

## Data Privacy Notice
This repository contains only non-PHI source code and synthetic or illustrative example files. No protected health information (PHI), identifiable patient data, or confidential institutional information is stored or distributed through this repository.
