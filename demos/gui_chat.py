"""Simple GUI chat interface for Tantra-LLM."""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import os

from demos.demo_minimal import build_demo


class TantraGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tantra-LLM Chat")
        self.root.geometry("800x600")
        
        # Build brain (in background to avoid blocking)
        self.brain = None
        self.brain_ready = False
        self.status_label = tk.Label(root, text="Loading brain...", fg="orange")
        self.status_label.pack(pady=5)
        
        # Load brain in background
        threading.Thread(target=self._load_brain, daemon=True).start()
        
        # Chat area
        self.chat_area = scrolledtext.ScrolledText(
            root,
            wrap=tk.WORD,
            width=80,
            height=30,
            font=("Arial", 10),
            state=tk.DISABLED
        )
        self.chat_area.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Configure tags (use standard color names)
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.tag_config("title", font=("Arial", 12, "bold"))
        self.chat_area.tag_config("ready", foreground="green")
        self.chat_area.tag_config("error", foreground="red")
        self.chat_area.tag_config("user", font=("Arial", 10, "bold"))
        self.chat_area.tag_config("bot", font=("Arial", 10))
        self.chat_area.config(state=tk.DISABLED)
        
        # Input frame
        input_frame = ttk.Frame(root)
        input_frame.pack(pady=10, padx=10, fill=tk.X)
        
        self.input_entry = tk.Entry(input_frame, font=("Arial", 12))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.input_entry.bind("<Return>", lambda e: self.send_message())
        
        self.send_button = tk.Button(
            input_frame,
            text="Send",
            command=self.send_message,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 11, "bold"),
            width=10
        )
        self.send_button.pack(side=tk.RIGHT)
        
        # Add welcome message
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, "Tantra-LLM Chat\n", "title")
        self.chat_area.insert(tk.END, "=" * 50 + "\n")
        self.chat_area.insert(tk.END, "Loading brain, please wait...\n\n")
        self.chat_area.config(state=tk.DISABLED)
        
    def _load_brain(self):
        try:
            # Set env vars
            os.environ["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            os.environ["TANTRA_LV_DIR"] = os.environ.get("TANTRA_LV_DIR", "D:\\models\\longvita-16k")
            os.environ["TANTRA_SPB"] = os.environ.get("TANTRA_SPB", "microsoft/DialoGPT-medium")
            
            self.brain = build_demo()
            self.brain_ready = True
            self.status_label.config(text="Ready ✅", fg="green")
            self.root.after(0, self._show_ready_message)
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", fg="red")
            self.root.after(0, lambda: self.chat_area.insert(tk.END, f"Error loading brain: {e}\n", "error"))
    
    def _show_ready_message(self):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, "\nBrain loaded! You can start chatting.\n\n", "ready")
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.see(tk.END)
    
    def send_message(self):
        if not self.brain_ready:
            return
        
        user_input = self.input_entry.get().strip()
        if not user_input:
            return
        
        self.input_entry.delete(0, tk.END)
        self.add_message("You", user_input, "user")
        
        # Process in background thread to avoid blocking
        threading.Thread(target=self._get_response, args=(user_input,), daemon=True).start()
    
    def _get_response(self, user_input):
        try:
            response = self.brain.step(text=user_input)
            self.root.after(0, lambda: self.add_message("Tantra", response, "bot"))
        except Exception as e:
            self.root.after(0, lambda: self.add_message("System", f"Error: {e}", "error"))
    
    def add_message(self, sender, message, tag):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, f"{sender}: ", tag if tag != "error" else "error")
        self.chat_area.insert(tk.END, f"{message}\n\n")
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.see(tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = TantraGUI(root)
    root.mainloop()

