import subprocess

scripts = [
    'scrape_data.py',
    'process_data.py',
    'correlation_plots.py',
    'feature_importance.py',
    'linear_regression.py',
    'knn_regression.py',
    'random_forest_regression.py',
    'compare_models.py',
    'test_knn_rf_models.py',
    'analyze_knn_rf_models.py'
]

# inspired by: https://docs.python.org/3/library/subprocess.html
def run_script(script_name):
    try:
        print(f"Running {script_name}...")
        result = subprocess.run(['python', script_name], check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"{script_name} finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}:")
        print(e.stderr)
        raise

def main():
    for script in scripts:
        run_script(script)

if __name__ == "__main__":
    main()
