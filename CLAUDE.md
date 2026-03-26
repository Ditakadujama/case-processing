# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

病历相似度检索系统 (Medical Record Similarity Retrieval System) - converts medical record text into feature vectors and finds similar cases using vector similarity search.

## Running the System

```bash
conda activate sepsis
python main.py
```

## Architecture

```
main.py → retrieval_system.py → record_parser.py (structure)
                               → feature_extractor.py (vectorize)
                               → similarity_index.py (search)
```

**Data Flow:**
1. `MedicalRecordParser` parses raw text into structured `MedicalRecord` (patient info, diagnoses, lab results, medications, vital signs)
2. `FeatureExtractor` converts structured record into a feature vector (~600-dim): diagnosis TF-IDF, normalized lab values, medication bag-of-words, demographic features
3. `VectorIndex` (sklearn or FAISS backend) performs similarity search using cosine similarity

**Key Classes:**
- [record_parser.py](record_parser.py) - `MedicalRecord`, `PatientInfo`, `LabResult`, `Diagnosis`, `Medication` dataclasses; `MedicalRecordParser`
- [feature_extractor.py](feature_extractor.py) - `FeatureExtractor` with TF-IDF vectorizer for diagnoses, fixed-dimension vectors for labs/medications/demographics
- [similarity_index.py](similarity_index.py) - `VectorIndex` abstract interface; `SklearnIndex` (brute-force, <10k records); `FaissIndex` (IVF-PQ for large scale); `IndexManager` auto-switches sklearn→FAISS at 10k records
- [retrieval_system.py](retrieval_system.py) - `MedicalRecordSimilaritySystem` orchestrates all components; `create_system()` factory

**Core Parameters:**
- `similarity_threshold`: 0.6~0.8 (filter results below this)
- `top_k`: 10~50 (number of results to return)
- Feature weights are implicit in the concatenated vector

## Dependencies

- jieba>=0.42.1 (Chinese segmentation)
- numpy>=1.24.0
- pandas>=2.0.0
- scikit-learn>=1.3.0
- Optional: faiss-cpu/faiss-gpu for large-scale retrieval
