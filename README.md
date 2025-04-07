# Time Value of Money Calculator App

This repository contains a Streamlit application that helps users perform and visualize time value of money (TVM) calculations. It provides an interactive interface for exploring different financial scenarios and understanding TVM concepts.

## Features

- **Single Payment Calculations**: Calculate present value from future value and vice versa
- **Uniform Series (Annuity) Calculations**: Analyze equal periodic payments
- **Gradient Series Calculations**: Evaluate cash flows that increase by a constant amount
- **Custom Cash Flow Analysis**: Create your own cash flow pattern and calculate NPV and IRR
- **Dynamic Visualizations**: Interactive charts update in real-time as you change parameters
- **Comprehensive Formula Reference**: Access all TVM formulas for reference

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/tvm-calculator.git
   cd tvm-calculator
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Use the tabs to choose the type of calculation you want to perform:
   - Single Payment
   - Uniform Series (Annuity)
   - Gradient Series
   - Custom Cash Flow

4. Enter your parameters and view the results and visualizations in real-time

## Examples

### Single Payment Calculation
- Calculate how much a $1,000 investment will be worth in 5 years at 5% interest
- Determine how much you need to invest today to have $10,000 in 10 years

### Annuity Calculation
- Find the present value of $100 monthly payments for 5 years
- Calculate how much you need to save monthly to reach $50,000 in 10 years

### Gradient Series
- Evaluate cash flows that increase by a fixed amount each period
- Compare gradient series with equivalent uniform series

### Custom Cash Flow Analysis
- Input your own pattern of cash inflows and outflows
- Calculate NPV and IRR for investment analysis

## File Structure

- `app.py`: Main Streamlit application
- `requirements.txt`: Python dependencies
- `README.md`: This file
- `LICENSE`: License information

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributors

- [Your Name](https://github.com/yourusername)

## Acknowledgments

- Streamlit for the excellent framework for building data apps
- The finance and engineering economics community for TVM concepts
