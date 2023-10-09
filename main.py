import os
import tkinter as tk
from tkinter import Text, scrolledtext
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import os
import tkinter as tk
from tkinter import ttk, Text, scrolledtext, simpledialog, messagebox
from datetime import datetime

# Load the trained model and tokenizer
model = BartForConditionalGeneration.from_pretrained('Text_Summarizer',from_tf=True)
tokenizer = BartTokenizer.from_pretrained('Text_Summarizer',from_tf=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# Function to summarize text
def summarize_text():
    input_text = text_input.get("1.0", "end-1c")

    # Replace paragraph breaks with spaces
    input_text = input_text.replace('\n', ' ')

    inputs = tokenizer(input_text, return_tensors="pt", max_length=2048, truncation=True).to(device)

    # Calculate the desired summary length as 30% of the input length
    input_length = inputs["input_ids"].shape[1]
    desired_output_length = int(0.3 * input_length)

    summary_ids = model.generate(inputs["input_ids"],
                                 max_length=desired_output_length,
                                 min_length=int(0.25 * input_length),
                                 length_penalty=1.0,
                                 num_beams=2,  # Changed from 1 to 2
                                 early_stopping=True,
                                 do_sample=True,  # Enable sampling
                                 temperature=4.0,
                                 top_k=50,
                                 top_p=0.95,
                                 no_repeat_ngram_size=4,
                                 eos_token_id=tokenizer.eos_token_id)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # Display the summary
    summarized_text.config(state=tk.NORMAL)
    summarized_text.delete("1.0", tk.END)
    summarized_text.insert(tk.END, summary + "\n\n")
    summarized_text.config(state=tk.DISABLED)

    # Save the summary to history with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open("summarization_history.txt", "a", encoding="utf-8") as f:
        f.write(f"Timestamp: {timestamp}\n{summary}\n\n")

    # Update the history display and dropdown
    update_history_display()
    update_history_dropdown()


# Function to clear history
def clear_history():
    open("summarization_history.txt", "w").close()
    update_history_display()


# Function to update history display
def update_history_display():
    history_text.config(state=tk.NORMAL)
    history_text.delete("1.0", tk.END)
    if os.path.exists("summarization_history.txt"):
        with open("summarization_history.txt", "r", encoding="utf-8") as f:
            history = f.read()
            history_text.insert(tk.END, history)
    history_text.config(state=tk.DISABLED)

def update_history_dropdown():
    history_dates.clear()
    if os.path.exists("summarization_history.txt"):
        with open("summarization_history.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Timestamp:"):
                    history_dates.append(line.strip().split(": ")[1])
    date_dropdown['values'] = history_dates


# Updated function to clear a specific summarization
def delete_selected_summary():
    selected_date = date_var.get()
    if not selected_date:
        return

    # Read the history and then rewrite it without the selected summary
    with open("summarization_history.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open("summarization_history.txt", "w", encoding="utf-8") as f:
        is_selected_date = False
        for line in lines:
            if line.startswith(f"Timestamp: {selected_date}"):
                is_selected_date = True
                continue
            if is_selected_date and not line.startswith("Timestamp:"):
                continue
            elif is_selected_date and line.startswith("Timestamp:"):
                is_selected_date = False
                continue

            f.write(line)

    update_history_display()
    update_history_dropdown()

def load_selected_summary():
    selected_date = date_var.get()
    if not selected_date:
        return
    with open("summarization_history.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        is_selected_date = False
        summary = ""
        for line in lines:
            if line.startswith(f"Timestamp: {selected_date}"):
                is_selected_date = True
            if is_selected_date and not line.startswith("Timestamp:"):
                summary += line
            elif is_selected_date and line.startswith("Timestamp:"):
                break
    history_text.config(state=tk.NORMAL)
    history_text.delete("1.0", tk.END)
    history_text.insert(tk.END, summary)
    history_text.config(state=tk.DISABLED)
def new_summary():
    text_input.delete("1.0", tk.END)
    summarized_text.delete("1.0", tk.END)
root = tk.Tk()
root.title("Texty  -  Text Summarizer App")

# Create a title label for the app name
title_label = ttk.Label(root, text="Texty  -  Text Summarizer App", font=("Helvetica", 18, "bold"))
title_label.pack(pady=(10, 20))  # Adjust pady as needed

# Create a label for the input textr
input_label = ttk.Label(root, text="Input Text:")
input_label.pack()

# Input text widget
text_input = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=10, font=("Helvetica", 12))
text_input.pack(padx=10, pady=10)

# Create a frame for buttons and labels
button_frame = ttk.Frame(root)
button_frame.pack(pady=10)

# New Summary button
new_summary_button = ttk.Button(button_frame, text="New Summary", command=new_summary)
new_summary_button.grid(row=0, column=0, padx=10)

# Summarize button
summarize_button = ttk.Button(button_frame, text="Summarize", command=summarize_text)
summarize_button.grid(row=0, column=1, padx=10)

# Clear History button
clear_history_button = ttk.Button(button_frame, text="Clear History", command=clear_history)
clear_history_button.grid(row=0, column=2, padx=10)

# History date dropdown
history_dates = []
date_var = tk.StringVar()
date_dropdown = ttk.Combobox(button_frame, textvariable=date_var, values=history_dates, font=("Helvetica", 12))
date_dropdown.grid(row=0, column=3, padx=10)
date_dropdown.bind("<<ComboboxSelected>>", lambda e: load_selected_summary())

# Create a label for the output text
output_label = ttk.Label(root, text="Summarized Text:")
output_label.pack(pady=(10, 0))

# Display summarized text
summarized_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=10, font=("Helvetica", 12))
summarized_text.pack(padx=10, pady=10)

# Create a label for the history
history_label = ttk.Label(root, text="Summary History:")
history_label.pack(pady=(10, 0))

# History display
history_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=10, font=("Helvetica", 12))
history_text.pack(padx=10, pady=10)

# Adding an option to delete a specific summarization from history
delete_history_button = ttk.Button(root, text="Delete Selected Summary", command=delete_selected_summary)
delete_history_button.pack(pady=10)

# GUI Customization for colors and other features
text_input.configure(bg='#D3D3D3', fg='#000000')
summarized_text.configure(bg='#D3D3D3', fg='#000000')
history_text.configure(bg='#D3D3D3', fg='#000000')

# Making the window resizable
root.geometry("1000x800")  # Set an initial size, can be adjusted
root.minsize(800, 600)  # Set a minimum size

root.mainloop()
 