"""Graphical renderer for Assignment 3 (optional; run.py offers the same renders).

    python gui.py

Choose a scene and the settings, press Render, and watch the image improve as
samples accumulate. The render runs in a background thread, so the window
stays responsive and Stop ends it after the current pass.
"""

import base64
import io
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run import configure_numba, probe_tasks  # noqa: E402

configure_numba(False)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import scene as sc  # noqa: E402
import student  # noqa: E402

VIEW = 512  # longest side of the displayed image in screen pixels


class App:
    def __init__(self, root):
        self.root = root
        self.root.title('Assignment 3: Path Tracer')
        self.events = queue.Queue()
        self.stop_flag = threading.Event()
        self.worker = None
        self.radiance = None
        self.photo = None

        panel = ttk.Frame(root, padding=10)
        panel.grid(row=0, column=0, sticky='ns')
        self.vars = {
            'scene': tk.StringVar(value='cornell'),
            'size': tk.IntVar(value=256),
            'spp': tk.IntVar(value=64),
            'max_depth': tk.IntVar(value=50),
            'seed': tk.IntVar(value=1),
            'mesh_level': tk.IntVar(value=4),
            'bvh': tk.BooleanVar(value=True),
            'roulette': tk.BooleanVar(value=True),
        }
        row = 0
        ttk.Label(panel, text='Scene').grid(row=row, column=0, sticky='w')
        ttk.Combobox(panel, textvariable=self.vars['scene'], values=sc.SCENES, state='readonly',
                     width=12).grid(row=row, column=1, sticky='ew')
        for key, label in (('size', 'Height (px)'), ('spp', 'Samples per pixel'), ('max_depth', 'Max depth'),
                           ('seed', 'Seed'), ('mesh_level', 'Mesh level')):
            row += 1
            ttk.Label(panel, text=label).grid(row=row, column=0, sticky='w', pady=2)
            ttk.Spinbox(panel, textvariable=self.vars[key], from_=0, to=100000, width=10).grid(
                row=row, column=1, sticky='ew', pady=2)
        row += 1
        ttk.Checkbutton(panel, text='Use BVH', variable=self.vars['bvh']).grid(row=row, column=0, columnspan=2, sticky='w')
        row += 1
        ttk.Checkbutton(panel, text='Russian roulette', variable=self.vars['roulette']).grid(
            row=row, column=0, columnspan=2, sticky='w')
        row += 1
        buttons = ttk.Frame(panel)
        buttons.grid(row=row, column=0, columnspan=2, pady=(10, 4), sticky='ew')
        self.render_button = ttk.Button(buttons, text='Render', command=self.start)
        self.render_button.pack(side='left', expand=True, fill='x')
        self.stop_button = ttk.Button(buttons, text='Stop', command=self.stop_flag.set, state='disabled')
        self.stop_button.pack(side='left', expand=True, fill='x')
        row += 1
        self.save_button = ttk.Button(panel, text='Save PNG...', command=self.save, state='disabled')
        self.save_button.grid(row=row, column=0, columnspan=2, sticky='ew')
        row += 1
        self.progress = ttk.Progressbar(panel, length=200, mode='determinate')
        self.progress.grid(row=row, column=0, columnspan=2, pady=(10, 4), sticky='ew')
        row += 1
        self.status = ttk.Label(panel, text='Ready.', wraplength=220, justify='left')
        self.status.grid(row=row, column=0, columnspan=2, sticky='w')

        self.blank = tk.PhotoImage(width=VIEW, height=VIEW)
        self.canvas = tk.Label(root, image=self.blank, background='#202020', borderwidth=0)
        self.canvas.grid(row=0, column=1, padx=10, pady=10)
        self.root.after(100, self.poll)

    def settings(self):
        try:
            return {k: v.get() for k, v in self.vars.items()}
        except tk.TclError:
            return None

    def start(self):
        s = self.settings()
        if s is None or s['size'] < 1 or s['spp'] < 1 or s['max_depth'] < 1:
            self.status.config(text='Height, samples per pixel, and max depth must be positive integers.')
            return
        self.stop_flag.clear()
        self.render_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.save_button.config(state='disabled')
        self.progress.config(value=0, maximum=s['spp'])
        self.status.config(text='Preparing the scene and compiling (the first render takes longer)...')
        self.worker = threading.Thread(target=self.render, args=(s,), daemon=True)
        self.worker.start()

    def render(self, s):
        """Background thread: render in passes and send the running mean to the window."""
        try:
            missing, sdf_done, bvh_done = probe_tasks()
            if missing:
                self.events.put(('error', f'Incomplete student implementation: {missing}.'))
                return
            world, (cam, half_h) = sc.load_scene(s['scene'], s['mesh_level'])
            notes = []
            if not sdf_done:
                world = sc.without_sdf(world)
                notes.append('Task 5 is not implemented yet: the SDF object is left out.')
            if s['bvh']:
                if not bvh_done:
                    self.events.put(('error', 'Incomplete student implementation: Task 4: build_bvh. '
                                              'Untick "Use BVH" until Task 4 is done.'))
                    return
                world = sc.with_bvh(world)
            student.trace_path(cam[0].copy(), -cam[3], world, np.ones(1, np.uint64), 1, True)
            height = s['size']
            width = round(height * (4 / 3 if s['scene'] in ('spheres', 'materials') else 1))
            total = np.zeros((height, width, 3))
            done = 0
            batch = 1
            t0 = time.perf_counter()
            while done < s['spp'] and not self.stop_flag.is_set():
                n = min(batch, s['spp'] - done)
                pass_seed = s['seed'] * 1_000_003 + done
                total += n * sc.render(world, cam, half_h, width, height, n, s['max_depth'], pass_seed,
                                       s['roulette'])
                done += n
                batch = min(batch * 2, 16)
                image = total / done
                self.events.put(('image', image, done, time.perf_counter() - t0, notes))
            self.events.put(('done', done, time.perf_counter() - t0))
        except NotImplementedError as error:
            self.events.put(('error', f'Incomplete student implementation: {error}.'))
        except Exception as error:  # show any failure of student code in the window
            self.events.put(('error', f'{type(error).__name__}: {error}\nRun run.py with --no-jit for a full traceback.'))

    def poll(self):
        latest = None
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event[0] == 'image':
                latest = event
            else:
                if latest:
                    self.show(latest)
                    latest = None
                self.finish(event)
        if latest:
            self.show(latest)
        self.root.after(100, self.poll)

    def show(self, event):
        _, image, done, elapsed, notes = event
        self.radiance = image
        pil = Image.fromarray(sc.tone_map(image))
        scale = VIEW / max(pil.size)
        if scale != 1:
            pil = pil.resize((round(pil.width * scale), round(pil.height * scale)),
                             Image.NEAREST if scale > 1 else Image.LANCZOS)
        buffer = io.BytesIO()
        pil.save(buffer, format='PNG')
        self.photo = tk.PhotoImage(data=base64.b64encode(buffer.getvalue()))
        self.canvas.config(image=self.photo)
        self.progress.config(value=done)
        text = f'{done} samples per pixel, {elapsed:.1f} s, mean radiance {np.nanmean(image):.4f}'
        self.status.config(text='\n'.join([text] + notes))

    def finish(self, event):
        self.render_button.config(state='normal')
        self.stop_button.config(state='disabled')
        if event[0] == 'error':
            self.status.config(text=event[1])
            return
        _, done, elapsed = event
        self.save_button.config(state='normal' if self.radiance is not None else 'disabled')
        self.status.config(text=self.status.cget('text') + f'\nFinished: {done} samples per pixel in {elapsed:.1f} s.')

    def save(self):
        if self.radiance is None:
            return
        path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG image', '*.png')])
        if path:
            sc.save_png(self.radiance, path)
            self.status.config(text=self.status.cget('text') + f'\nSaved {path}')


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
