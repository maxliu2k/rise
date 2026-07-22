"""A pop-up progress bar (tkinter — stdlib, no install).

Guarded so it NEVER breaks a headless/background run: if no display is available, the
factory returns a no-op stub and training proceeds silently. Used by train.py's --progress
flag.

Standalone demo:
    python -m instrument_robustness.progress_popup
"""


class ProgressPopup:
    def __init__(self, title="Progress", total=100, width=360):
        import tkinter as tk
        from tkinter import ttk
        self.total = total
        self.root = tk.Tk()
        self.root.title(title)
        self.root.resizable(False, False)
        frm = ttk.Frame(self.root, padding=16)
        frm.pack(fill="both", expand=True)
        self.label = ttk.Label(frm, text="0%")
        self.label.pack(anchor="w", pady=(0, 6))
        self.bar = ttk.Progressbar(frm, length=width, mode="determinate", maximum=total)
        self.bar.pack()
        self.root.update()

    def update(self, value, text=None):
        self.bar["value"] = value
        pct = 100 * value / self.total if self.total else 0
        self.label.config(text=text or f"{pct:.0f}%")
        self.root.update()

    def close(self):
        self.root.destroy()


class _NullPopup:
    """Stand-in when no GUI is available — every method is a no-op."""
    def update(self, *a, **k): pass
    def close(self): pass


def make_popup(title="Progress", total=100):
    """Return a ProgressPopup, or a silent no-op stub if a window can't be opened
    (headless CI, background job, no DISPLAY). Training must not depend on the GUI."""
    try:
        return ProgressPopup(title, total)
    except Exception as e:
        print(f"(progress popup unavailable: {type(e).__name__}; continuing without it)")
        return _NullPopup()


if __name__ == "__main__":
    import time
    bar = make_popup("Demo", total=100)
    for i in range(1, 101):
        time.sleep(0.03)
        bar.update(i)
    bar.close()
