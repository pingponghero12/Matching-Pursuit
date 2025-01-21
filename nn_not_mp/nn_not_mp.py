import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

def main():
    print("dupa")
    data_path = 'data/s3v4_mach15.csv'
    df = pd.read_csv(data_path)
    print("dupa")
    mach_unique = df['MACH'].unique()
    re_unique = df['RE'].unique()
    print(f"Unique Mach values in dataset: {mach_unique}")
    print(f"Unique RE values in dataset: {re_unique}")

    input_cols = ['ALPHA', 'BETA', 'DEFL_1_1', 'DEFL_1_2', 'DEFL_1_3', 'DEFL_1_4']
    output_cols = ['FX', 'FY', 'FZ', 'TX', 'TY', 'TZ']

    df = df.dropna(subset=input_cols + output_cols)

    X = df[input_cols].values
    y = df[output_cols].values

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='swish', input_shape=(len(input_cols),)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='swish'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='swish'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='swish'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='swish'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(len(output_cols), activation='linear')  # 6 outputs: Fx, Fy, Fz, Tx, Ty, Tz
    ])

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    model.fit(X, y, epochs=200, batch_size=32, verbose=0)

    predictions = model.predict(X)
    correlations = []
    for i, col in enumerate(output_cols):
        corr = np.corrcoef(y[:, i], predictions[:, i])[0, 1]
        correlations.append(corr)
        print(f"Correlation for {col}: {corr:.4f}")

    alpha_min, alpha_max = df['ALPHA'].min(), df['ALPHA'].max()
    beta_min, beta_max = df['BETA'].min(), df['BETA'].max()
    alpha_space = np.linspace(alpha_min, alpha_max, 100)
    beta_space = np.linspace(beta_min, beta_max, 100)

    defl_means = df[['DEFL_1_1', 'DEFL_1_2', 'DEFL_1_3', 'DEFL_1_4']].mean().values

    alpha_mesh, beta_mesh = np.meshgrid(alpha_space, beta_space)
    grid_points = []
    for i in range(100):
        for j in range(100):
            grid_points.append([alpha_mesh[i, j], beta_mesh[i, j]] + list(defl_means))

    grid_points = np.array(grid_points)
    grid_preds = model.predict(grid_points)  # shape (10000, 6) -> (100 * 100, 6)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, col in enumerate(output_cols):
        z_values = grid_preds[:, idx]
        Z = z_values.reshape(100, 100)

        ax = axes[idx]
        c = ax.contourf(alpha_mesh, beta_mesh, Z, levels=50, cmap='viridis')
        ax.set_title(f'{col} Heatmap')
        ax.set_xlabel('ALPHA')
        ax.set_ylabel('BETA')
        fig.colorbar(c, ax=ax)

    plt.tight_layout()

    img_dir = 'img'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    output_path = os.path.join(img_dir, 'model_heatmaps.png')
    plt.savefig(output_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    print("dupa")
    main()
