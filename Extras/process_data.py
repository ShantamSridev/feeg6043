import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('fixed3.csv')

# Calculate Euclidean error
df['Euclidean_Error'] = np.sqrt(df['Error_Northings']**2 + df['Error_Eastings']**2)

# Create the plot
plt.figure(figsize=(12, 6))

# Plot all error metrics
plt.plot(df['Time'], df['Error_Northings'], 'b-', label='Error Northings')
plt.plot(df['Time'], df['Error_Eastings'], 'r-', label='Error Eastings')
plt.plot(df['Time'], df['Euclidean_Error'], 'g-', label='Absolute Error')

# Customize the plot
plt.title('Position Errors Over Time')
plt.xlabel('Time')
plt.ylabel('Error (meters)')
plt.grid(True)
plt.legend()

# Save the plot
plt.savefig('error_plot.png')
plt.close()

# Calculate and print statistics
print("Error Statistics:")
print("\nNorthings Error:")
print(f"Mean Absolute Error: {np.mean(np.abs(df['Error_Northings'])):.6f}")
print(f"Max Error: {df['Error_Northings'].max():.6f}")
print(f"Min Error: {df['Error_Northings'].min():.6f}")
print(f"Standard Deviation: {df['Error_Northings'].std():.6f}")

print("\nEastings Error:")
print(f"Mean Absolute Error: {np.mean(np.abs(df['Error_Eastings'])):.6f}")
print(f"Max Error: {df['Error_Eastings'].max():.6f}")
print(f"Min Error: {df['Error_Eastings'].min():.6f}")
print(f"Standard Deviation: {df['Error_Eastings'].std():.6f}")

print("\nEuclidean Error:")
print(f"Mean Error: {np.mean(df['Euclidean_Error']):.6f}")
print(f"Max Error: {df['Euclidean_Error'].max():.6f}")
print(f"Min Error: {df['Euclidean_Error'].min():.6f}")
print(f"Standard Deviation: {df['Euclidean_Error'].std():.6f}")

# Save processed data with Euclidean error
df.to_csv('processed_data_with_euclidean.csv', index=False) 