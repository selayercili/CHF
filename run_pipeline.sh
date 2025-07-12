#!/bin/bash
# run_pipeline.sh
# Automated pipeline script for CHF prediction project

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

# Banner
echo "============================================"
echo "     CHF Prediction Pipeline Runner"
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

# Create necessary directories
print_status "Creating project directories..."
mkdir -p data/{raw,processed}
mkdir -p logs
mkdir -p weights
mkdir -p results
mkdir -p reports/{figures,metrics}
print_success "Directories created"

# Parse command line arguments
SKIP_DOWNLOAD=false
SKIP_TRAIN=false
SKIP_TEST=false
SKIP_EVAL=false
MODELS=""

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
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --skip-download    Skip data download step"
            echo "  --skip-train       Skip model training step"
            echo "  --skip-test        Skip model testing step"
            echo "  --skip-eval        Skip evaluation step"
            echo "  --models MODELS    Train specific models (space-separated)"
            echo "  --help             Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --models \"xgboost lightgbm\""
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Step 1: Download and organize data
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo ""
    print_status "Step 1/4: Downloading and organizing data..."
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

# Step 2: Train models
if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    print_status "Step 2/4: Training models..."
    
    if [ -n "$MODELS" ]; then
        python scripts/train.py --models $MODELS
    else
        python scripts/train.py
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Model training completed"
    else
        print_error "Model training failed"
        exit 1
    fi
else
    print_warning "Skipping model training step"
fi

# Step 3: Test models
if [ "$SKIP_TEST" = false ]; then
    echo ""
    print_status "Step 3/4: Testing models..."
    
    if [ -n "$MODELS" ]; then
        python scripts/test.py --models $MODELS
    else
        python scripts/test.py
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Model testing completed"
    else
        print_error "Model testing failed"
        exit 1
    fi
else
    print_warning "Skipping model testing step"
fi

# Step 4: Generate evaluation reports
if [ "$SKIP_EVAL" = false ]; then
    echo ""
    print_status "Step 4/4: Generating evaluation reports..."
    python scripts/evaluate.py
    
    if [ $? -eq 0 ]; then
        print_success "Evaluation completed"
    else
        print_error "Evaluation failed"
        exit 1
    fi
else
    print_warning "Skipping evaluation step"
fi

# Summary
echo ""
echo "============================================"
echo "          Pipeline Complete!"
echo "============================================"
echo ""
print_success "All steps completed successfully"
echo ""
echo "Results available in:"
echo "  - Model weights: weights/"
echo "  - Test results: results/"
echo "  - Evaluation reports: reports/"
echo ""

# Open report if available
if [ -f "reports/evaluation_report.md" ]; then
    print_status "Evaluation report available at: reports/evaluation_report.md"
    
    # Try to open in default browser/editor
    if command_exists xdg-open; then
        xdg-open reports/evaluation_report.md 2>/dev/null &
    elif command_exists open; then
        open reports/evaluation_report.md 2>/dev/null &
    fi
fi