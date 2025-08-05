# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display

# --------------------------------------------------------------
# Configuration
# --------------------------------------------------------------
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams['figure.figsize'] = (20, 5)
mpl.rcParams['figure.dpi'] = 100

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# --------------------------------------------------------------
# Helper: Plot sensor data
# --------------------------------------------------------------
def plot_sensor_data(subset, sensor_columns, ylabel, title, cmap):
    fig, ax = plt.subplots()
    subset[sensor_columns].plot(ax=ax, cmap=cmap, linewidth=2, alpha=0.85)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Samples")
    ax.set_title(title)
    ax.legend(title="Sensor")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# --------------------------------------------------------------
# Plot single column for set = 1
# --------------------------------------------------------------
set_df = df[df['set'] == 1]
plt.figure()
plt.plot(set_df['acc_y'].reset_index(drop=True))
plt.title("acc_y for set 1")
plt.show()

# --------------------------------------------------------------
# Plot acc_y for each exercise label
# --------------------------------------------------------------
for label in df['label'].unique():
    subset = df[df['label'] == label]
    plt.figure()
    plt.plot(subset['acc_y'].reset_index(drop=True), label=label)
    plt.legend()
    plt.title(f"{label} - acc_y")
    plt.show()

# Plot first 100 samples of acc_y for each label
for label in df['label'].unique():
    subset = df[df['label'] == label][:100]
    plt.figure()
    plt.plot(subset['acc_y'].reset_index(drop=True), label=label)
    plt.legend()
    plt.title(f"{label} - acc_y (first 100 samples)")
    plt.show()

# --------------------------------------------------------------
# Compare medium vs. heavy sets (Squat - Participant A)
# --------------------------------------------------------------
squat_df = df[(df['label'] == 'squat') & (df['participant'] == 'A')].reset_index()
fig, ax = plt.subplots()
squat_df.groupby('category')['acc_y'].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("Samples")
ax.legend(title="Category")
plt.title("Squat - Participant A: Medium vs. Heavy")
plt.show()

# --------------------------------------------------------------
# Compare acc_y between participants for bench press
# --------------------------------------------------------------
bench_df = df[df['label'] == 'bench'].sort_values('participant').reset_index()
fig, ax = plt.subplots()
bench_df.groupby('participant')['acc_y'].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("Samples")
ax.legend(title="Participant")
plt.title("Bench Press - acc_y by Participant")
plt.show()

# --------------------------------------------------------------
# Plot acc/gyr sensors for each label and participant
# --------------------------------------------------------------
labels = df['label'].unique()
participants = df['participant'].unique()

for label in labels:
    for participant in participants:
        subset = df[(df['label'] == label) & (df['participant'] == participant)].reset_index()
        if subset.empty:
            continue

        # Accelerometer
        plot_sensor_data(
            subset,
            sensor_columns=['acc_y', 'acc_x', 'acc_z'],
            ylabel="Acceleration",
            title=f"{label.title()} - {participant} (Accelerometer)",
            cmap='viridis'
        )

        # Gyroscope
        plot_sensor_data(
            subset,
            sensor_columns=['gyr_y', 'gyr_x', 'gyr_z'],
            ylabel="Gyroscope",
            title=f"{label.title()} - {participant} (Gyroscope)",
            cmap='plasma'
        )

# --------------------------------------------------------------
# Combined acc and gyr plots in one figure and save
# --------------------------------------------------------------
output_dir = "../../reports/figures/"

for label in labels:
    for participant in participants:
        subset = df[(df['label'] == label) & (df['participant'] == participant)].reset_index()
        if subset.empty:
            continue

        fig, axes = plt.subplots(2, 1, figsize=(22, 10), sharex=True, gridspec_kw={'hspace': 0.3})
        fig.patch.set_facecolor('#f7f7f7')

        # Accelerometer plot
        subset[['acc_y', 'acc_x', 'acc_z']].plot(
            ax=axes[0], linewidth=2, alpha=0.85, cmap='viridis')
        axes[0].set_ylabel("Acceleration", fontsize=14)
        axes[0].legend(title='Sensor', fontsize=12, title_fontsize=13, loc='upper right',
                       frameon=True, facecolor='white')
        axes[0].set_title(f"{label.title()} - {participant} (Accelerometer)", fontsize=16, fontweight='bold', color='#333')
        axes[0].grid(True, linestyle='--', alpha=0.5)

        # Gyroscope plot
        subset[['gyr_y', 'gyr_x', 'gyr_z']].plot(
            ax=axes[1], linewidth=2, alpha=0.85, cmap='plasma')
        axes[1].set_ylabel("Gyroscope", fontsize=14)
        axes[1].legend(title='Sensor', fontsize=12, title_fontsize=13, loc='upper right',
                       frameon=True, facecolor='white')
        axes[1].set_title(f"{label.title()} - {participant} (Gyroscope)", fontsize=16, fontweight='bold', color='#333')
        axes[1].set_xlabel("Samples", fontsize=14)
        axes[1].grid(True, linestyle='--', alpha=0.5)

        # Super title
        fig.suptitle(
            f"Sensor Data for {label.title()} - Participant {participant}",
            fontsize=18, fontweight='bold', color='#222', y=1.02
        )

        # Save and show
        plt.savefig(f"{output_dir}{label}_{participant}_combined.png", dpi=300, bbox_inches='tight')
        plt.tight_layout()
        plt.show()
