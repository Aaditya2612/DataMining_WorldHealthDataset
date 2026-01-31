import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ----------------------------
# 1. LOAD DATA
# ----------------------------
df = pd.read_csv("world_health_data.csv")

# Fill missing numeric values with mean
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Fill missing non-numeric values with mode
non_num_cols = df.select_dtypes(exclude=[np.number]).columns
df[non_num_cols] = df[non_num_cols].fillna(df[non_num_cols].mode().iloc[0])

# ----------------------------
# 2. PREDICTIVE MODEL
# ----------------------------
features = ['maternal_mortality', 'infant_mortality', 'neonatal_mortality',
            'under_5_mortality', 'prev_hiv', 'inci_tuberc', 'prev_undernourishment']
target_life = 'life_expect'
target_health = 'health_exp'

X = df[features]
y_life = df[target_life]
y_health = df[target_health]

X_train, X_test, y_train, y_train_health = train_test_split(X, y_life, test_size=0.25, random_state=42)
rf_life = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
rf_life.fit(X_train, y_train)

rf_health = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
rf_health.fit(X, y_health)

# ----------------------------
# 3. TKINTER GUI
# ----------------------------
root = tk.Tk()
root.title("🌍 Global Health Predictor & Visualizer")
root.geometry("1200x850")
root.configure(bg='#111111')

# Tab style
style = ttk.Style()
style.theme_use('clam')
style.configure('TNotebook.Tab', font=('Arial', 13, 'bold'), padding=[25, 12])

notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True)

# ----------------------------
# Scrollable frame 
# ----------------------------
def create_scrollable_tab(tab):
    canvas = tk.Canvas(tab, bg='#111111', highlightthickness=0)
    scrollbar = tk.Scrollbar(tab, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg='#111111')
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    return scrollable_frame

# ----------------------------
# Input Tab
# ----------------------------
tab_input = tk.Frame(notebook, bg='#111111')
notebook.add(tab_input, text="Predict Metrics")

tk.Label(tab_input, text="Enter Health Indicators", font=('Arial', 16, 'bold'), fg='white', bg='#111111').pack(pady=15)

entries = {}
for col in features:
    tk.Label(tab_input, text=col.replace('_', ' ').title(), fg='white', bg='#111111', font=('Arial', 12)).pack(pady=5)
    entry = tk.Entry(tab_input, font=('Arial', 12))
    entry.pack(pady=5, ipadx=30)
    entries[col] = entry

def predict_now():
    try:
        input_data = [float(entries[col].get()) for col in features]
        life_pred = rf_life.predict([input_data])[0]
        health_pred = rf_health.predict([input_data])[0]
        r2_life = r2_score(y_train, rf_life.predict(X_train))
        messagebox.showinfo(
            "Predictions",
            f"Predicted Life Expectancy: {life_pred:.2f} years\n"
            f"Predicted Health Expenditure: {health_pred:.2f} % of GDP\n"
            f"Life Expectancy Model R²: {r2_life:.3f}"
        )
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

tk.Button(tab_input, text="Predict", command=predict_now,
          bg='teal', fg='white', font=('Arial', 14)).pack(pady=20)

# ----------------------------
# Overview Tab
# ----------------------------
tab_overview = tk.Frame(notebook, bg='#111111')
notebook.add(tab_overview, text="Overview")

frame_overview = create_scrollable_tab(tab_overview)

# Helper function to format axes for dark background
def format_ax_dark(ax):
    ax.set_facecolor('#111111')
    ax.figure.set_facecolor('#111111')
    ax.title.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    if ax.get_legend() is not None:
        leg = ax.get_legend()
        for text in leg.get_texts():
            text.set_color('white')
        leg.get_frame().set_facecolor('#111111')
        leg.get_frame().set_edgecolor('white')

# Life Expectancy Distribution
fig1, ax1 = plt.subplots(figsize=(10,5))
fig1.patch.set_facecolor('#111111')
sns.histplot(df['life_expect'], bins=30, kde=True, color='cyan', ax=ax1)
ax1.set_title('Global Life Expectancy Distribution')
ax1.set_xlabel('Life Expectancy (Years)')
ax1.set_ylabel('Density')
format_ax_dark(ax1)
canvas1 = FigureCanvasTkAgg(fig1, master=frame_overview)
canvas1.draw()
canvas1.get_tk_widget().pack(pady=25)

# Under-5 Mortality Trend
fig2, ax2 = plt.subplots(figsize=(10,5))
fig2.patch.set_facecolor('#111111')
under5 = df.groupby('year')['under_5_mortality'].mean()
smoothed = gaussian_filter1d(under5.values, sigma=2)
ax2.plot(under5.index, under5.values, marker='o', color='orange', label='Under-5 Mortality')
ax2.plot(under5.index, smoothed, linestyle='--', color='cyan', label='Trend (smoothed)')
ax2.set_title('Global Under-5 Mortality Over Time')
ax2.set_xlabel('Year')
ax2.set_ylabel('Under-5 Mortality (per 1000)')
ax2.legend()
format_ax_dark(ax2)
canvas2 = FigureCanvasTkAgg(fig2, master=frame_overview)
canvas2.draw()
canvas2.get_tk_widget().pack(pady=25)

# Correlation Heatmap
fig3, ax3 = plt.subplots(figsize=(10,7))
fig3.patch.set_facecolor('#111111')
corr_matrix = df[features + ['life_expect']].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax3,
            cbar_kws={'label':'Correlation'}, annot_kws={'color':'white'})
ax3.set_title('Correlation of Health Indicators')
format_ax_dark(ax3)
canvas3 = FigureCanvasTkAgg(fig3, master=frame_overview)
canvas3.draw()
canvas3.get_tk_widget().pack(pady=25)

# ----------------------------
# Mortality Tab
# ----------------------------
tab_mortality = tk.Frame(notebook, bg='#111111')
notebook.add(tab_mortality, text="Mortality")

frame_mortality = create_scrollable_tab(tab_mortality)

# Infant vs Maternal Mortality
fig4, ax4 = plt.subplots(figsize=(10,5))
fig4.patch.set_facecolor('#111111')
ax4.scatter(df['maternal_mortality'], df['infant_mortality'], color='orange', alpha=0.6)
sns.regplot(x='maternal_mortality', y='infant_mortality', data=df, scatter=False, ax=ax4, line_kws={"color": "cyan", "linestyle": "--"})
ax4.set_title('Infant vs Maternal Mortality')
ax4.set_xlabel('Maternal Mortality')
ax4.set_ylabel('Infant Mortality')
format_ax_dark(ax4)
canvas4 = FigureCanvasTkAgg(fig4, master=frame_mortality)
canvas4.draw()
canvas4.get_tk_widget().pack(pady=25)

# Neonatal vs Under-5 Mortality
fig5, ax5 = plt.subplots(figsize=(10,5))
fig5.patch.set_facecolor('#111111')
ax5.scatter(df['neonatal_mortality'], df['under_5_mortality'], color='purple', alpha=0.6)
sns.regplot(x='neonatal_mortality', y='under_5_mortality', data=df, scatter=False, ax=ax5, line_kws={"color":"cyan","linestyle":"--"})
ax5.set_title('Neonatal vs Under-5 Mortality')
ax5.set_xlabel('Neonatal Mortality')
ax5.set_ylabel('Under-5 Mortality')
format_ax_dark(ax5)
canvas5 = FigureCanvasTkAgg(fig5, master=frame_mortality)
canvas5.draw()
canvas5.get_tk_widget().pack(pady=25)

# HIV Prevalence vs Life Expectancy
fig6, ax6 = plt.subplots(figsize=(10,5))
fig6.patch.set_facecolor('#111111')
ax6.scatter(df['prev_hiv'], df['life_expect'], color='red', alpha=0.6)
sns.regplot(x='prev_hiv', y='life_expect', data=df, scatter=False, ax=ax6, line_kws={"color":"cyan","linestyle":"--"})
ax6.set_title('HIV Prevalence vs Life Expectancy')
ax6.set_xlabel('HIV Prevalence (%)')
ax6.set_ylabel('Life Expectancy')
format_ax_dark(ax6)
canvas6 = FigureCanvasTkAgg(fig6, master=frame_mortality)
canvas6.draw()
canvas6.get_tk_widget().pack(pady=25)

# Under-nourishment over time
fig7, ax7 = plt.subplots(figsize=(10,5))
fig7.patch.set_facecolor('#111111')
undernourished = df.groupby('year')['prev_undernourishment'].mean()
smoothed_und = gaussian_filter1d(undernourished.values, sigma=2)
ax7.plot(undernourished.index, undernourished.values, color='orange', marker='o', label='Under-nourishment')
ax7.plot(undernourished.index, smoothed_und, linestyle='--', color='cyan', label='Trend (smoothed)')
ax7.set_title('Global Undernourishment Over Time')
ax7.set_xlabel('Year')
ax7.set_ylabel('Percent Undernourished')
ax7.legend()
format_ax_dark(ax7)
canvas7 = FigureCanvasTkAgg(fig7, master=frame_mortality)
canvas7.draw()
canvas7.get_tk_widget().pack(pady=25)

# TB incidence vs Life Expectancy
fig8, ax8 = plt.subplots(figsize=(10,5))
fig8.patch.set_facecolor('#111111')
ax8.scatter(df['inci_tuberc'], df['life_expect'], color='green', alpha=0.6)
sns.regplot(x='inci_tuberc', y='life_expect', data=df, scatter=False, ax=ax8, line_kws={"color":"cyan","linestyle":"--"})
ax8.set_title('TB Incidence vs Life Expectancy')
ax8.set_xlabel('TB Incidence')
ax8.set_ylabel('Life Expectancy')
format_ax_dark(ax8)
canvas8 = FigureCanvasTkAgg(fig8, master=frame_mortality)
canvas8.draw()
canvas8.get_tk_widget().pack(pady=25)

root.mainloop()
