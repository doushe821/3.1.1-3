import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.odr import ODR, RealData, Model
import re

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot data with errors and linear fit')
    parser.add_argument('filename', help='Input CSV filename')
    parser.add_argument('error_flags', help='Error flags (11, 10, 01, 00)')
    parser.add_argument('xlabel', help='X axis label')
    parser.add_argument('ylabel', help='Y axis label')
    parser.add_argument('title', help='Plot title')
    parser.add_argument('output', help='Output PNG filename')
    return parser.parse_args()

def load_data(filename, error_flags):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    data = []
    for line in lines:
        numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
        
        # Updated condition: accept 2, 3, or 4 columns based on error_flags
        if error_flags == '11' and len(numbers) != 4:
            raise ValueError(f"Expected 4 columns for error flags '11', got {len(numbers)}: {line}")
        elif error_flags in ['10', '01'] and len(numbers) != 3:
            raise ValueError(f"Expected 3 columns for error flags '{error_flags}', got {len(numbers)}: {line}")
        elif error_flags == '00' and len(numbers) != 2:
            raise ValueError(f"Expected 2 columns for error flags '00', got {len(numbers)}: {line}")
        
        numbers = [float(num) for num in numbers]
        data.append(numbers)
    
    df = pd.DataFrame(data)
    
    # Now assign columns based on error_flags
    if error_flags == '11':
        df.columns = ['x', 'x_err', 'y', 'y_err']
    elif error_flags == '10':
        df.columns = ['x', 'x_err', 'y']
        df['y_err'] = 0.0
    elif error_flags == '01':
        df.columns = ['x', 'y', 'y_err'] 
        df['x_err'] = 0.0
    elif error_flags == '00':
        df.columns = ['x', 'y']
        df['x_err'] = 0.0
        df['y_err'] = 0.0
    else:
        raise ValueError("Invalid error flags. Use: 11, 10, 01, or 00")
    
    return df

def linear_func(p, x):
    return p[0] * x + p[1]
def fit_linear(x, y, x_err, y_err):
    # Replace zero errors with a very small value to avoid division by zero
    # but only if ALL errors are zero (otherwise preserve actual error values)
    x_err_safe = x_err.copy()
    y_err_safe = y_err.copy()
    
    # If all X errors are exactly zero, set them to a small relative value
    if np.all(x_err == 0):
        x_err_safe = np.full_like(x_err, 1e-10 * np.std(x))
    
    # If all Y errors are exactly zero, set them to a small relative value  
    if np.all(y_err == 0):
        y_err_safe = np.full_like(y_err, 1e-10 * np.std(y))
    
    # If both errors are zero, use ordinary least squares
    if np.all(x_err == 0) and np.all(y_err == 0):
        A = np.vstack([x, np.ones(len(x))]).T
        coeffs, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
        
        # Estimate errors from residuals
        if len(residuals) > 0:
            y_pred = coeffs[0] * x + coeffs[1]
            mse = np.sum((y - y_pred)**2) / (len(x) - 2)
            cov = mse * np.linalg.inv(A.T @ A)
            errors = np.sqrt(np.diag(cov))
        else:
            errors = np.array([0.0, 0.0])
        return coeffs, errors
    else:
        # Use ODR with safe error values
        try:
            data = RealData(x, y, sx=x_err_safe, sy=y_err_safe)
            model = Model(linear_func)
            odr = ODR(data, model, beta0=[1.0, 1.0])
            output = odr.run()
            return output.beta, output.sd_beta
        except Exception as e:
            # Fallback to ordinary least squares if ODR fails
            print(f"ODR failed, using ordinary least squares: {e}")
            A = np.vstack([x, np.ones(len(x))]).T
            coeffs, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
            
            if len(residuals) > 0:
                y_pred = coeffs[0] * x + coeffs[1]
                mse = np.sum((y - y_pred)**2) / (len(x) - 2)
                cov = mse * np.linalg.inv(A.T @ A)
                errors = np.sqrt(np.diag(cov))
            else:
                errors = np.array([0.0, 0.0])
            return coeffs, errors

def main():
    args = parse_arguments()
    
    try:
        df = load_data(args.filename, args.error_flags)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Отключаем LaTeX полностью
    plt.rcParams['text.usetex'] = False
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data with error bars
    if args.error_flags in ['11', '10'] and not df['x_err'].isna().all() and np.any(df['x_err'] > 0):
        x_err = df['x_err']
    else:
        x_err = None
        
    if args.error_flags in ['11', '01'] and not df['y_err'].isna().all() and np.any(df['y_err'] > 0):
        y_err = df['y_err']
    else:
        y_err = None
    
    ax.errorbar(df['x'], df['y'], xerr=x_err, yerr=y_err, 
                fmt='o', capsize=3, label='', markersize=4)
    
    # Perform linear fit
    try:
        coeffs, errors = fit_linear(df['x'].values, df['y'].values, 
                                   df['x_err'].values, df['y_err'].values)
        
        # Plot fit line
        x_fit = np.linspace(df['x'].min(), df['x'].max(), 100)
        y_fit = linear_func(coeffs, x_fit)
        ax.plot(x_fit, y_fit, 'r-', label='', linewidth=2)
        
        # Add equation text - используем обычный текст вместо LaTeX
        equation = f'y = ({coeffs[0]:.3f} ± {errors[0]:.3f})x + ({coeffs[1]:.3f} ± {errors[1]:.3f})'
        
        ax.text(0.05, 0.95, equation, transform=ax.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
    except Exception as e:
        print(f"Error during linear fit: {e}")
        # Если аппроксимация не удалась, просто рисуем точки
        pass
    
    # Устанавливаем подписи осей и заголовок
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.set_title(args.title)
    
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    try:
        plt.tight_layout()
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {args.output}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    plt.close()
    
    # Print results to terminal
    try:
        print("Linear fit results:")
        print(f"Slope: {coeffs[0]:.6f} ± {errors[0]:.6f}")
        print(f"Intercept: {coeffs[1]:.6f} ± {errors[1]:.6f}")
        print(f"Correlation coefficient: {np.corrcoef(df['x'], df['y'])[0,1]:.3f}")
    except:
        print("Could not calculate fit results")

if __name__ == '__main__':
    main()