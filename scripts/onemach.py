import pandas as pd

def filter_and_save_csv(input_path, output_path):
    df = pd.read_csv(input_path)

    filtered_df = df[df['MACH'] == 0.15]

    filtered_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_path = 'data/s3v4.csv'
    output_path = 'data/s3v4_mach15.csv'
    filter_and_save_csv(input_path, output_path)
