# AI-Powered Defect Prediction & Root Cause Analysis System

## Project Overview
An advanced manufacturing quality management system that combines ML/AI with GenAI to predict defects, analyze root causes, and provide intelligent operator assistance in real-time. Built for the TCS AI Hackathon, this solution aims to transform reactive defect detection into proactive prevention.

## Key Features
- **Real-time Defect Prediction**
  - ML-based failure prediction across stages and vendors
  - Temporal pattern detection in production shifts
  - Early warning system for quality issues

- **Intelligent Root Cause Analysis**
  - Error code pattern clustering
  - Vendor and stage performance analytics
  - Time-series based failure analysis

- **Interactive Dashboard**
  - Real-time monitoring and predictions
  - Cost impact analysis
  - Performance metrics visualization
  - Stage-wise and vendor-wise analytics

## Technology Stack
- **Machine Learning**
  - Random Forest Classifier
  - XGBoost
  - LSTM for temporal patterns
  
- **Data Processing**
  - Python
  - pandas
  - scikit-learn
  - TensorFlow

- **Visualization**
  - Streamlit
  - Plotly
  - Interactive dashboards

## Project Structure
```
ManufacturingAI/
├── generate_data.ipynb        # Data generation and preprocessing
├── model_training.ipynb      # ML model training and evaluation
├── dashboard.py             # Streamlit dashboard application
├── manufacturing_quality_model.joblib  # Trained model
├── feature_scaler.joblib    # Feature scaling parameters
└── feature_columns.json     # Feature configuration
```

## Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/SravanthiNC/AI-Powered-Defect-Prediction-Root-Cause-Analysis-System-
   cd ManufacturingAI
   ```

2. **Install Dependencies**
   ```bash
   pip install pandas numpy scikit-learn xgboost tensorflow plotly streamlit
   ```

3. **Generate Synthetic Data**
   - Open and run `generate_data.ipynb`
   - This will create the required CSV files

4. **Train Models**
   - Open and run `model_training.ipynb`
   - This will create the model files and scalers

5. **Launch Dashboard**
   ```bash
   streamlit run dashboard.py
   ```

## Usage Guide

### Data Generation
- Run `generate_data.ipynb` to create synthetic manufacturing data
- Configurable parameters for failure rates and patterns
- Generates both main unit and component assembly data

### Model Training
- Execute `model_training.ipynb` for:
  - Feature engineering
  - Model training
  - Performance evaluation
  - Model persistence

### Dashboard Navigation
1. **Overview Page**
   - Key metrics and KPIs
   - Temporal trends
   - Overall statistics

2. **Failure Analysis**
   - Stage-Vendor heatmap
   - Error code distribution
   - Pattern analysis

3. **Predictions**
   - Real-time failure probability
   - Risk assessment
   - Preventive recommendations

4. **Cost Impact**
   - Vendor-wise cost analysis
   - Total impact calculation
   - ROI projections

## Performance Metrics
- Model Accuracy: ~91%
- Early Detection Rate: Up to 70%
- Cost Reduction Potential: 15-30%

## Business Benefits
1. **Operational Excellence**
   - Reduced defect rate
   - Proactive quality management
   - Optimized production flow

2. **Cost Optimization**
   - Minimized rework costs
   - Reduced downtime
   - Better resource allocation

3. **Strategic Insights**
   - Vendor performance benchmarking
   - Stage-wise quality metrics
   - Data-driven decision making

## Future Enhancements
1. **Phase 2: GenAI Integration**
   - LLM-powered root cause analysis
   - Natural language insights
   - Contextual recommendations

2. **Phase 3: Edge Deployment**
   - Real-time edge processing
   - IoT sensor integration
   - Low-latency predictions

3. **Phase 4: Enterprise Scale**
   - Multi-plant deployment
   - Cross-facility analytics
   - Enterprise-wide insights

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- TCS AI Hackathon platform
- Manufacturing quality management best practices
- Open-source ML/AI community

## Contact
- Developer: Sravanthi NC
- Project: AI-Powered Defect Prediction & Root Cause Analysis System
- Repository: [GitHub](https://github.com/SravanthiNC/AI-Powered-Defect-Prediction-Root-Cause-Analysis-System-)