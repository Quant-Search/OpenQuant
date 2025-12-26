"""
OpenQuant Launcher - Simple GUI for non-technical users
Double-click to start the trading dashboard
"""
import subprocess
import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import threading
import webbrowser

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
VENV_PYTHON = SCRIPT_DIR / ".venv" / "Scripts" / "python.exe"
DASHBOARD_SCRIPT = SCRIPT_DIR / "robot" / "dashboard.py"


class OpenQuantLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("OpenQuant Trading Robot")
        self.root.geometry("400x300")
        self.root.resizable(False, False)
        
        # Center window
        self.root.eval('tk::PlaceWindow . center')
        
        self.process = None
        self.setup_ui()
        
    def setup_ui(self):
        # Title
        title = tk.Label(
            self.root, 
            text="🤖 OpenQuant Trading Robot",
            font=("Segoe UI", 16, "bold")
        )
        title.pack(pady=20)
        
        # Status
        self.status_var = tk.StringVar(value="Status: Stopped")
        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Segoe UI", 10)
        )
        self.status_label.pack(pady=10)
        
        # Progress bar (indeterminate when running)
        self.progress = ttk.Progressbar(
            self.root, 
            mode='indeterminate',
            length=300
        )
        self.progress.pack(pady=10)
        
        # Buttons frame
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=20)
        
        self.start_btn = tk.Button(
            btn_frame,
            text="▶ Start Dashboard",
            command=self.start_dashboard,
            font=("Segoe UI", 12),
            bg="#4CAF50",
            fg="white",
            width=15,
            height=2
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(
            btn_frame,
            text="⏹ Stop",
            command=self.stop_dashboard,
            font=("Segoe UI", 12),
            bg="#f44336",
            fg="white",
            width=15,
            height=2,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Open browser button
        self.browser_btn = tk.Button(
            self.root,
            text="🌐 Open Dashboard in Browser",
            command=lambda: webbrowser.open("http://localhost:8501"),
            font=("Segoe UI", 10),
            state=tk.DISABLED
        )
        self.browser_btn.pack(pady=10)
        
        # Info
        info = tk.Label(
            self.root,
            text="Dashboard URL: http://localhost:8501",
            font=("Segoe UI", 9),
            fg="gray"
        )
        info.pack(side=tk.BOTTOM, pady=10)
        
    def start_dashboard(self):
        if not VENV_PYTHON.exists():
            messagebox.showerror(
                "Error", 
                f"Python not found at:\n{VENV_PYTHON}\n\nPlease run setup first."
            )
            return
            
        def run():
            try:
                self.process = subprocess.Popen(
                    [str(VENV_PYTHON), "-m", "streamlit", "run", 
                     str(DASHBOARD_SCRIPT), "--server.port", "8501"],
                    cwd=str(SCRIPT_DIR),
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                self.root.after(0, self.on_started)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
                
        self.status_var.set("Status: Starting...")
        self.start_btn.config(state=tk.DISABLED)
        threading.Thread(target=run, daemon=True).start()
        
    def on_started(self):
        self.status_var.set("Status: ✅ Running")
        self.progress.start(10)
        self.stop_btn.config(state=tk.NORMAL)
        self.browser_btn.config(state=tk.NORMAL)
        # Auto-open browser after 3 seconds
        self.root.after(3000, lambda: webbrowser.open("http://localhost:8501"))
        
    def stop_dashboard(self):
        if self.process:
            self.process.terminate()
            self.process = None
        self.status_var.set("Status: Stopped")
        self.progress.stop()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.browser_btn.config(state=tk.DISABLED)
        
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()
        
    def on_close(self):
        self.stop_dashboard()
        self.root.destroy()


if __name__ == "__main__":
    app = OpenQuantLauncher()
    app.run()

