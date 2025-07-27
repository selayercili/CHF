#!/bin/bash
# run_pipeline.sh
# Automated pipeline script for CHF prediction project with SMOTE comparison support

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python package
check_python_package() {
    python -c "import $1" 2>/dev/null
}

# Parse command line arguments
SKIP_DOWNLOAD=false
SKIP_TRAIN=false
SKIP_TEST=false
SKIP_EVAL=false
MODELS=""
DATA_TYPE="both"  # New: default to both for comparison

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-test)
            SKIP_TEST=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --models)
            shift
            MODELS="$1"
            shift
            ;;
        --data-type)
            shift
            DATA_TYPE="$1"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --skip-download    Skip data download step"
            echo "  --skip-train       Skip model training step"
            echo "  --skip-test        Skip model testing step"
            echo "  --skip-eval        Skip evaluation step"
            echo "  --models MODELS    Train specific models (space-separated)"
            echo "  --data-type TYPE   Data type: 'smote', 'regular', or 'both' (default: both)"
            echo "  --help             Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --models \"xgboost lightgbm\" --data-type both"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to run pipeline for a specific data type
run_pipeline_for_data_type() {
    local data_type=$1
    local suffix=""
    
    if [ "$data_type" = "smote" ]; then
        suffix="_smote"
    elif [ "$data_type" = "regular" ]; then
        suffix="_regular"
    fi
    
    echo ""
    echo "============================================"
    echo "     Running Pipeline for ${data_type^^} Data"
    echo "============================================"
    echo ""
    
    # Create data-type specific directories
    print_status "Creating directories for $data_type data..."
    mkdir -p logs$suffix
    mkdir -p weights$suffix
    mkdir -p results$suffix
    mkdir -p reports$suffix/{figures,metrics}
    print_success "Directories created for $data_type data"
    
    # Step 2: Train models
    if [ "$SKIP_TRAIN" = false ]; then
        echo ""
        print_status "Training models on $data_type data..."
        
        if [ -n "$MODELS" ]; then
            python scripts/train.py --models $MODELS --data-type $data_type
        else
            python scripts/train.py --data-type $data_type
        fi
        
        if [ $? -eq 0 ]; then
            print_success "Model training completed for $data_type data"
        else
            print_error "Model training failed for $data_type data"
            return 1
        fi
    else
        print_warning "Skipping model training step"
    fi
    
    # Step 3: Test models
    if [ "$SKIP_TEST" = false ]; then
        echo ""
        print_status "Testing models trained on $data_type data..."
        
        if [ -n "$MODELS" ]; then
            python scripts/test.py --models $MODELS --data-type $data_type
        else
            python scripts/test.py --data-type $data_type
        fi
        
        if [ $? -eq 0 ]; then
            print_success "Model testing completed for $data_type data"
        else
            print_error "Model testing failed for $data_type data"
            return 1
        fi
    else
        print_warning "Skipping model testing step"
    fi
    
    return 0
}

# Banner
echo "============================================"
echo "     CHF Prediction Pipeline Runner"
echo "      Data Type: ${DATA_TYPE^^}"
echo "============================================"
echo ""

# Check Python
print_status "Checking Python installation..."
if ! command_exists python; then
    print_error "Python not found! Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
print_success "Python $PYTHON_VERSION found"

# Check for virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    print_warning "No virtual environment active"
    
    # Check if venv exists
    if [ -d "venv" ]; then
        print_status "Activating existing virtual environment..."
        source venv/bin/activate || source venv/Scripts/activate
        print_success "Virtual environment activated"
    else
        print_status "Creating virtual environment..."
        python -m venv venv
        source venv/bin/activate || source venv/Scripts/activate
        print_success "Virtual environment created and activated"
        
        # Upgrade pip
        print_status "Upgrading pip..."
        pip install --upgrade pip
    fi
else
    print_success "Virtual environment is active: $VIRTUAL_ENV"
fi

# Install dependencies
print_status "Checking dependencies..."
if ! check_python_package "pandas"; then
    print_status "Installing requirements..."
    pip install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        print_success "Dependencies installed successfully"
    else
        print_error "Failed to install dependencies"
        exit 1
    fi
else
    print_success "Dependencies already installed"
fi

# Create base directories
print_status "Creating base project directories..."
mkdir -p data/{raw,processed}
mkdir -p logs
mkdir -p reports
print_success "Base directories created"

# Step 1: Download and organize data (only needed once)
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo ""
    print_status "Step 1: Downloading and organizing data..."
    python scripts/download_and_organize.py
    
    if [ $? -eq 0 ]; then
        print_success "Data pipeline completed"
    else
        print_error "Data pipeline failed"
        exit 1
    fi
else
    print_warning "Skipping data download step"
fi

# Run pipeline based on data type
if [ "$DATA_TYPE" = "both" ]; then
    # Run for both data types
    run_pipeline_for_data_type "regular"
    REGULAR_RESULT=$?
    
    run_pipeline_for_data_type "smote"
    SMOTE_RESULT=$?
    
    if [ $REGULAR_RESULT -ne 0 ] || [ $SMOTE_RESULT -ne 0 ]; then
        print_error "One or more pipelines failed"
        exit 1
    fi
    
    # Run comparative evaluation
    if [ "$SKIP_EVAL" = false ]; then
        echo ""
        print_status "Generating comparative evaluation reports..."
        python scripts/evaluate.py --comparison-mode --data-type both
        
        if [ $? -eq 0 ]; then
            print_success "Comparative evaluation completed"
        else
            print_error "Comparative evaluation failed"
            exit 1
        fi
    fi
else
    # Run for single data type
    run_pipeline_for_data_type "$DATA_TYPE"
    
    if [ $? -ne 0 ]; then
        print_error "Pipeline failed"
        exit 1
    fi
    
    # Run standard evaluation
    if [ "$SKIP_EVAL" = false ]; then
        echo ""
        print_status "Generating evaluation reports..."
        python scripts/evaluate.py --data-type "$DATA_TYPE"
        
        if [ $? -eq 0 ]; then
            print_success "Evaluation completed"
        else
            print_error "Evaluation failed"
            exit 1
        fi
    fi
fi

# Summary
echo ""
echo "============================================"
echo "          Pipeline Complete!"
echo "============================================"
echo ""
print_success "All steps completed successfully"
echo ""

if [ "$DATA_TYPE" = "both" ]; then
    echo "Results available in:"
    echo "  Regular data:"
    echo "    - Model weights: weights_regular/"
    echo "    - Test results: results_regular/"
    echo "    - Evaluation reports: reports_regular/"
    echo ""
    echo "  SMOTE data:"
    echo "    - Model weights: weights_smote/"
    echo "    - Test results: results_smote/"
    echo "    - Evaluation reports: reports_smote/"
    echo ""
    echo "  Comparative analysis: reports/comparison/"
else
    SUFFIX=""
    if [ "$DATA_TYPE" = "smote" ]; then
        SUFFIX="_smote"
    elif [ "$DATA_TYPE" = "regular" ]; then
        SUFFIX="_regular"
    fi
    
    echo "Results available in:"
    echo "  - Model weights: weights$SUFFIX/"
    echo "  - Test results: results$SUFFIX/"
    echo "  - Evaluation reports: reports$SUFFIX/"
fi
echo ""

# Open report if available
if [ "$DATA_TYPE" = "both" ] && [ -f "reports/comparison/comparison_report.md" ]; then
    print_status "Comparison report available at: reports/comparison/comparison_report.md"
    
    # Try to open in default browser/editor
    if command_exists xdg-open; then
        xdg-open reports/comparison/comparison_report.md 2>/dev/null &
    elif command_exists open; then
        open reports/comparison/comparison_report.md 2>/dev/null &
    fi
fi