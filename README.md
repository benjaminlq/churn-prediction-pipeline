# End-to-End ML Pipeline for Churn Prediction
![](https://img.shields.io/badge/Status-Completed-green)
![](https://img.shields.io/badge/Domain-Telecommunication-green)
![](https://img.shields.io/badge/Cloud-Google%20Cloud%20Platform-blue)
![](https://img.shields.io/badge/Platform-Vertex%20AI-blue)
![](https://img.shields.io/badge/Package-Scikit--Learn-orange)
![](https://img.shields.io/badge/Package-TFX-orange) <br>
![](https://img.shields.io/badge/Language-Python-yellowgreen)
![](https://img.shields.io/badge/Language-Docker-yellowgreen)
![](https://img.shields.io/badge/Language-Yaml-yellowgreen)

## Business Setting
The goal of the project is to build a pipeline to deploy a model for telecom churn prediction. The end-to-end training pipeline consist of data ingestion, validation, preprocessing to model building, optimization, testing and deployment.<br>

The model objective is to predict if a telecom customer will churn. To tackle this binary classification problem, we trained and optimized a RandomForestClassifier. Features used for classification are users' demographics, customers profile and historical usage data.<br<br>

The pipeline is built on Google Cloud Platform and deployed at Vertex AI endpoints for model serving and monitoring.

## Pipeline
### 1. Training Pipeline
<img src="https://user-images.githubusercontent.com/99384454/187937724-efd80bef-2c8f-43e4-8abd-ab93f11ac350.png" width="800">

### 2. Pipeline Creation
The individual components of the pipeline are containerized inside Docker images after passing tests.<br>
Pipeline is orchestrated on Kubeflow SDK V2.<br>
<img src="https://user-images.githubusercontent.com/99384454/187939063-5ea7d938-d5c7-451d-a97d-2056530622e1.png" width="800">

### 3. Components
#### 3.1. Data Ingestion
<img src="https://user-images.githubusercontent.com/99384454/187939637-b205d0b7-f83a-4f1c-81b6-ebdfd78f1597.png" width="550">

#### 3.2. Data Validation
<img src="https://user-images.githubusercontent.com/99384454/187939788-aeffcb85-5150-4d0f-a2d1-bb03991dee68.png" width="550">

#### 3.3. Data Preprocessing
<img src="https://user-images.githubusercontent.com/99384454/187939983-c51e3eee-1137-44a4-b390-60195fb104aa.png" width="550">

#### 3.4. Model Training
<img src="https://user-images.githubusercontent.com/99384454/187942571-9d729bf3-708c-4056-8f19-935629a20ccd.png" width="550">

#### 3.5. Model Optimization
<img src="https://user-images.githubusercontent.com/99384454/187942378-fe2524e2-03db-4261-bdc5-9a978d1a4c2b.png" width="550">

#### 3.6. Model Evaluation
<img src="https://user-images.githubusercontent.com/99384454/187943061-548951e0-b3d1-4cf6-8ca2-f9e610e254e1.png" width="550">

#### 3.7. Model Deployment
<img src="https://user-images.githubusercontent.com/99384454/187943261-a28bf287-69b5-4e3a-a859-22f9c4301164.png" width="550">

### 4. Pipeline Orchestration
![image](https://user-images.githubusercontent.com/99384454/187946446-958aaa7f-1c70-464a-920b-9953a5fa3811.png)

## Model Serving
### Model Serving
<img src="https://user-images.githubusercontent.com/99384454/187943586-36f303ba-298e-4fdd-b07d-80b8835ec26e.png" width="550">

###  Model Monitoring
<img src="https://user-images.githubusercontent.com/99384454/187944057-36ff6be4-32bc-4534-a9c4-18768fb0304b.png" width="550">
