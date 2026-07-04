"""
gui/app.py
===========
GUI para LF Ambisonics Toolkit — CustomTkinter + Matplotlib

Uso (desde la raíz del proyecto):
    python gui/app.py

Dependencias:
    pip install customtkinter matplotlib
"""

import sys
import os
from pathlib import Path

# Path setup — funciona corriendo desde la raíz o desde gui/

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from src.spatial_metrics import (
    analyze_directionality,
    plot_directional_panel,
)
import matplotlib.pyplot as plt




import threading
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import scipy.io.wavfile as _wav

from src.acoustic_core import (
    load_aformat_mono_files,
    load_aformat_multichannel,
    align_aformat_channels,
    fine_align_channels,
    aformat_to_bformat,
    detect_onset_noise_floor,
    octave_band_filter,
    compute_LF_band,
    export_bformat_wav,
    BFormatSignals,
    _normalize_wav,
    OCTAVE_BANDS_HZ,
)

# ── Apariencia ────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

SIDEBAR_WIDTH = 290
WINDOW_SIZE = "1150x700"
BAND_LABELS = [str(f) for f in OCTAVE_BANDS_HZ]


class LFApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("LF Ambisonics Toolkit")
        self.geometry(WINDOW_SIZE)
        self.minsize(950, 600)

        # Estado de la aplicación
        self.signals = None
        self.fs = None
        self.bformat = None
        self.bformat_input = None   # B-format cargado directamente (sin conversión A→B)
        self.import_mode = ctk.StringVar(value="multichannel")
        self.bformat_order = ctk.StringVar(value="WYZX")
        self.processing = False

        self._build_layout()
        self._update_import_ui()
        self._update_align_ui()

    # ── Layout principal ──────────────────────────────────────────────────────
    def _build_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_main()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=SIDEBAR_WIDTH, corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_propagate(False)
        sb.grid_columnconfigure(0, weight=1)
        sb.grid_columnconfigure(1, weight=0)
        sb.grid_rowconfigure(98, weight=1)  # spacer

        r = 0

        # Título
        ctk.CTkLabel(sb, text="LF Ambisonics\nToolkit",
                     font=ctk.CTkFont(size=17, weight="bold")).grid(
            row=r, column=0, columnspan=2, padx=20, pady=(20, 4), sticky="w")
        r += 1
        ctk.CTkLabel(sb, text="ISO 3382-1 · A-format pipeline",
                     font=ctk.CTkFont(size=11), text_color="gray").grid(
            row=r, column=0, columnspan=2, padx=20, pady=(0, 14), sticky="w")
        r += 1

        # ── Importar RIR ──────────────────────────────────────────────────────
        self._section_label(sb, r, "IMPORTAR RIR")
        r += 1

        ctk.CTkRadioButton(sb, text="Archivo multicanal (4ch)",
                           variable=self.import_mode, value="multichannel",
                           command=self._update_import_ui).grid(
            row=r, column=0, columnspan=2, padx=20, pady=2, sticky="w")
        r += 1

        ctk.CTkRadioButton(sb, text="4 archivos mono independientes",
                           variable=self.import_mode, value="mono",
                           command=self._update_import_ui).grid(
            row=r, column=0, columnspan=2, padx=20, pady=2, sticky="w")
        r += 1

        ctk.CTkRadioButton(sb, text="B-format (4ch — comparación)",
                           variable=self.import_mode, value="bformat",
                           command=self._update_import_ui).grid(
            row=r, column=0, columnspan=2, padx=20, pady=2, sticky="w")
        r += 1

        # Selector de orden de canales B-format (visible solo cuando mode=bformat)
        self.frm_bformat_order = ctk.CTkFrame(sb, fg_color="transparent")
        self.frm_bformat_order.grid(row=r, column=0, columnspan=2, padx=(32, 20), pady=(2, 0), sticky="ew")
        self.frm_bformat_order.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self.frm_bformat_order, text="Orden canales:",
                     font=ctk.CTkFont(size=11)).grid(row=0, column=0, sticky="w")
        ctk.CTkOptionMenu(self.frm_bformat_order,
                          variable=self.bformat_order,
                          values=["WYZX", "WXYZ"],
                          width=80,
                          font=ctk.CTkFont(size=11)).grid(row=0, column=1, padx=(6, 0), sticky="e")
        r += 1

        self.btn_load = ctk.CTkButton(sb, text="Seleccionar archivo",
                                      command=self._load_files)
        self.btn_load.grid(row=r, column=0, columnspan=2, padx=20, pady=(8, 4), sticky="ew")
        r += 1

        self.lbl_status = ctk.CTkLabel(sb, text="Sin archivos cargados",
                                       font=ctk.CTkFont(size=11),
                                       text_color="gray", wraplength=250,
                                       justify="left")
        self.lbl_status.grid(row=r, column=0, columnspan=2, padx=20, pady=(2, 10), sticky="w")
        r += 1

        # ── Parámetros ────────────────────────────────────────────────────────
        self._section_label(sb, r, "PARÁMETROS DE PROCESAMIENTO")
        r += 1

        self.var_noise_window = ctk.StringVar(value="50")
        r = self._param_row(sb, r, "Ventana de ruido (ms)", self.var_noise_window,
                            tooltip="Pre-roll de silencio para estimar el piso de ruido.\n"
                                    "Usar 50 ms para archivos del DAW, 5 ms para archivos\n"
                                    "con onset cerca del inicio.")

        self.var_threshold = ctk.StringVar(value="15")
        r = self._param_row(sb, r, "Umbral onset (dB)", self.var_threshold,
                            tooltip="Nivel sobre el piso de ruido para detectar el inicio.\n"
                                    "Típico: 15 dB. Usar 20 dB si hay pre-ringing.")

        self.var_t1 = ctk.StringVar(value="5")
        r = self._param_row(sb, r, "t₁ integración (ms)", self.var_t1,
                            tooltip="Inicio de la ventana del numerador. ISO 3382-1: 5 ms.")

        self.var_t2 = ctk.StringVar(value="80")
        r = self._param_row(sb, r, "t₂ integración (ms)", self.var_t2,
                            tooltip="Fin de la ventana de integración. ISO 3382-1: 80 ms.")

        # Alineación agregado gpt
        self.var_align = ctk.BooleanVar(value=True)
        self.chk_align = ctk.CTkCheckBox(
            sb,
            text="Alineación temporal de canales",
            variable=self.var_align,
            command=self._update_align_ui
        )
        self.chk_align.grid(
            row=r,
            column=0,
            columnspan=2,
            padx=20,
            pady=(10,2),
            sticky="w"
        )
        r += 1

        self.var_radius = ctk.StringVar(value="5.0")
        self.lbl_radius = ctk.CTkLabel(sb, text="Radio búsqueda (ms):",
                                        font=ctk.CTkFont(size=12))
        self.lbl_radius.grid(row=r, column=0, padx=(32, 4), pady=2, sticky="w")
        self.ent_radius = ctk.CTkEntry(sb, textvariable=self.var_radius, width=65)
        self.ent_radius.grid(row=r, column=1, padx=(0, 20), pady=2, sticky="e")
        r += 1

        # Spacer
        ctk.CTkLabel(sb, text="").grid(row=98, column=0)

        # ── Botones de acción ─────────────────────────────────────────────────
        self.btn_export = ctk.CTkButton(sb, text="Exportar B-format (.wav)",
                                         command=self._export_bformat,
                                         fg_color="transparent",
                                         border_width=1,
                                         state="disabled")
        self.btn_export.grid(row=99, column=0, columnspan=2, padx=20, pady=(8, 4), sticky="ew")

        self.btn_process = ctk.CTkButton(sb, text="▶   Procesar",
                                          command=self._start_processing,
                                          font=ctk.CTkFont(size=14, weight="bold"),
                                          height=46,
                                          state="disabled")
        self.btn_process.grid(row=101, column=0, columnspan=2, padx=20, pady=(4, 20), sticky="ew")
        self.btn_direction = ctk.CTkButton(
                sb,
                text="Directional Analysis",
                command=self.show_directionality,
                state="disabled",
        )
        #agregado gpt        
        self.btn_direction.grid(
            row=100,
            column=0,
            columnspan=2,
            padx=20,
            pady=(4,20),
            sticky="ew"
        )
    def _section_label(self, parent, row, text):
        ctk.CTkLabel(parent, text=text,
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color="gray").grid(
            row=row, column=0, columnspan=2, padx=20, pady=(12, 4), sticky="w")

    def _param_row(self, parent, row, label, var, tooltip=None):
        ctk.CTkLabel(parent, text=f"{label}:",
                     font=ctk.CTkFont(size=12)).grid(
            row=row, column=0, padx=20, pady=2, sticky="w")
        ctk.CTkEntry(parent, textvariable=var, width=65).grid(
            row=row, column=1, padx=(0, 20), pady=2, sticky="e")
        return row + 1

    # ── Panel principal ───────────────────────────────────────────────────────
    def _build_main(self):
        main = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(main)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.tabview.add("Resultados")
        self.tabview.add("Instrucciones")

        self._build_results_tab()
        self._build_instructions_tab()

    def _build_results_tab(self):
        tab = self.tabview.tab("Resultados")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        self.lbl_result_status = ctk.CTkLabel(
            tab, text="Cargá una RIR y presioná ▶ Procesar para ver los resultados.",
            font=ctk.CTkFont(size=12), text_color="gray")
        self.lbl_result_status.grid(row=0, column=0, pady=(8, 4), padx=10, sticky="w")

        content = ctk.CTkFrame(tab, fg_color="transparent")
        content.grid(row=1, column=0, sticky="nsew")
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        # Tabla
        table_frame = ctk.CTkFrame(content, width=210)
        table_frame.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        table_frame.grid_propagate(False)
        self._build_table(table_frame)

        # Plot
        plot_frame = ctk.CTkFrame(content)
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.grid_columnconfigure(0, weight=1)
        plot_frame.grid_rowconfigure(0, weight=1)
        self._build_plot(plot_frame)

    def _build_table(self, parent):
        # Encabezados
        for col, (txt, w) in enumerate([("Banda (Hz)", 110), ("LF", 75)]):
            ctk.CTkLabel(parent, text=txt,
                         font=ctk.CTkFont(size=12, weight="bold"),
                         width=w).grid(row=0, column=col, padx=4, pady=(14, 6))

        self.table_labels = {}
        for i, fc in enumerate(OCTAVE_BANDS_HZ):
            ctk.CTkLabel(parent, text=str(fc),
                         font=ctk.CTkFont(size=12), width=110).grid(
                row=i + 1, column=0, padx=4, pady=3)
            lbl = ctk.CTkLabel(parent, text="—",
                               font=ctk.CTkFont(size=12), width=75)
            lbl.grid(row=i + 1, column=1, padx=4, pady=3)
            self.table_labels[fc] = lbl

        # Separador visual (línea vacía)
        ctk.CTkLabel(parent, text="", height=2).grid(
            row=len(OCTAVE_BANDS_HZ) + 1, column=0, columnspan=2)

        ctk.CTkLabel(parent, text="Media",
                     font=ctk.CTkFont(size=12, weight="bold"), width=110).grid(
            row=len(OCTAVE_BANDS_HZ) + 2, column=0, padx=4, pady=(4, 14))
        self.lbl_mean = ctk.CTkLabel(parent, text="—",
                                      font=ctk.CTkFont(size=12, weight="bold"), width=75)
        self.lbl_mean.grid(row=len(OCTAVE_BANDS_HZ) + 2, column=1, padx=4, pady=(4, 14))

    def _build_plot(self, parent):
        self.fig = Figure(figsize=(5, 3.8), dpi=100, facecolor="#2b2b2b")
        self.ax = self.fig.add_subplot(111)
        self._draw_empty_plot()

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    def _draw_empty_plot(self):
        ax = self.ax
        ax.set_facecolor("#1e1e1e")
        self.fig.patch.set_facecolor("#2b2b2b")
        ax.set_xlabel("Frecuencia (Hz)", color="#cccccc", fontsize=11)
        ax.set_ylabel("LF", color="#cccccc", fontsize=11)
        ax.set_title("Lateral Fraction por banda de octava", color="white", fontsize=12)
        ax.set_xscale("log")
        ax.set_xticks(OCTAVE_BANDS_HZ)
        ax.set_xticklabels(BAND_LABELS, color="#cccccc", fontsize=10)
        ax.tick_params(axis="y", colors="#cccccc")
        for spine in ax.spines.values():
            spine.set_color("#444444")
        ax.axhline(y=0.2, color="#555555", linestyle="--", linewidth=0.8, alpha=0.6,
                   label="LF = 0.2 (ref.)")
        ax.set_ylim(0, 1.4)
        ax.set_xlim(80, 6000)
        ax.legend(facecolor="#3a3a3a", labelcolor="#cccccc", fontsize=9,
                  framealpha=0.6)
        self.fig.tight_layout(pad=1.5)

    def _build_instructions_tab(self):
        tab = self.tabview.tab("Instrucciones")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        box = ctk.CTkTextbox(tab, font=ctk.CTkFont(size=12), wrap="word")
        box.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        box.insert("1.0", INSTRUCTIONS_TEXT)
        box.configure(state="disabled")

    # ── UI updates ────────────────────────────────────────────────────────────
    def _update_import_ui(self):
        mode = self.import_mode.get()
        if mode == "multichannel":
            self.btn_load.configure(text="Seleccionar archivo (4ch)")
            self.frm_bformat_order.grid_remove()
            self._set_align_widgets_state("normal")
        elif mode == "mono":
            self.btn_load.configure(text="Seleccionar 4 archivos mono")
            self.frm_bformat_order.grid_remove()
            self._set_align_widgets_state("normal")
        else:  # bformat
            self.btn_load.configure(text="Seleccionar B-format (4ch)")
            self.frm_bformat_order.grid()
            # Alineación no aplica a B-format
            self._set_align_widgets_state("disabled")

    def _set_align_widgets_state(self, state):
        self.chk_align.configure(state=state)
        self.ent_radius.configure(state=state)
    
        self.lbl_radius.configure(
            text_color="gray" if state=="disabled" else "white"
        )
    def _update_align_ui(self):
        if self.import_mode.get() == "bformat":
            return
    
        enabled = self.var_align.get()
    
        self.ent_radius.configure(
            state="normal" if enabled else "disabled"
        )
    
        self.lbl_radius.configure(
            text_color="white" if enabled else "gray"
        )

    # ── Carga de archivos ─────────────────────────────────────────────────────
    def _load_files(self):
        mode = self.import_mode.get()
        if mode == "multichannel":
            path = filedialog.askopenfilename(
                title="Seleccionar archivo A-format (4 canales)",
                filetypes=[("WAV files", "*.wav *.WAV")])
            if not path:
                return
            try:
                signals, fs = load_aformat_multichannel(
                    path, channel_order=["LF", "RF", "LB", "RB"])
                self._validate_and_store(signals, fs, Path(path).name)
            except Exception as e:
                self.lbl_status.configure(text=f"Error: {e}", text_color="red")
        elif mode == "mono":
            keys = ["LF", "RF", "LB", "RB"]
            labels = ["LF — Front Left", "RF — Front Right",
                      "LB — Back Left", "RB — Back Right"]
            paths = {}
            for key, label in zip(keys, labels):
                p = filedialog.askopenfilename(
                    title=f"Canal {label}",
                    filetypes=[("WAV files", "*.wav *.WAV")])
                if not p:
                    self.lbl_status.configure(text="Carga cancelada.", text_color="orange")
                    return
                paths[key] = p
            try:
                signals, fs = load_aformat_mono_files(paths)
                self._validate_and_store(signals, fs, "4 archivos mono")
            except Exception as e:
                self.lbl_status.configure(text=f"Error: {e}", text_color="red")
        else:  # bformat
            self._load_bformat_file()

    def _load_bformat_file(self):
        path = filedialog.askopenfilename(
            title="Seleccionar B-format (4 canales)",
            filetypes=[("WAV files", "*.wav *.WAV")])
        if not path:
            return
        try:
            fs, raw = _wav.read(path)
            if raw.ndim != 2 or raw.shape[1] != 4:
                raise ValueError(
                    f"El archivo debe tener exactamente 4 canales (tiene {raw.shape[1] if raw.ndim == 2 else 1}).")
            data = _normalize_wav(raw)
            order = self.bformat_order.get()   # "WYZX" o "WXYZ"
            ch = {letter: data[:, i] for i, letter in enumerate(order)}
            bf = BFormatSignals(
                W=ch['W'], X=ch['X'], Y=ch['Y'], Z=ch['Z'], fs=fs)
            self.bformat_input = bf
            self.signals = None
            self.fs = fs
            self.bformat = None
            dur_ms = len(bf.W) / fs * 1000
            self.lbl_status.configure(
                text=f"✓ B-format cargado ({order})\n{fs} Hz · {dur_ms:.0f} ms",
                text_color="#4caf50")
            self.btn_process.configure(state="normal")
            self.btn_direction.configure(state="disabled")
            self.btn_export.configure(state="disabled")
        except Exception as e:
            self.lbl_status.configure(text=f"Error: {e}", text_color="red")

    def _validate_and_store(self, signals, fs, name):
        lengths = [len(s) for s in signals.values()]
        if len(set(lengths)) > 1:
            diff_ms = (max(lengths) - min(lengths)) / fs * 1000
            msg = (f"✓ {name}\n{fs} Hz · canales con Δ longitud = {diff_ms:.0f} ms\n"
                   f"Activar alineación de canales.")
            color = "orange"
        else:
            dur_ms = lengths[0] / fs * 1000
            msg = f"✓ {name}\n{fs} Hz · {dur_ms:.0f} ms"
            color = "#4caf50"
#agregado gpt
        self.btn_direction.configure(state="disabled")

        self.signals = signals
        self.fs = fs
        self.bformat = None
        self.bformat_input = None   # no es B-format directo
        self.lbl_status.configure(text=msg, text_color=color)
        self.btn_process.configure(state="normal")
        self.btn_export.configure(state="disabled")

    # ── Procesamiento ─────────────────────────────────────────────────────────
    def _start_processing(self):
        if self.signals is None and self.bformat_input is None:
            return
        if self.processing:
            return
        try:
            params = dict(
                noise_window=float(self.var_noise_window.get()),
                threshold=float(self.var_threshold.get()),
                t1=float(self.var_t1.get()),
                t2=float(self.var_t2.get()),
                radius=float(self.var_radius.get()),
                do_align=self.var_align.get(),
            )
        except ValueError:
            messagebox.showerror("Parámetros inválidos",
                                 "Revisá que todos los parámetros sean números válidos.")
            return

        self.processing = True
        self.btn_process.configure(state="disabled", text="Procesando…")
        self.lbl_result_status.configure(text="Procesando…", text_color="orange")

        threading.Thread(target=self._run_pipeline, kwargs=params, daemon=True).start()

    def _run_pipeline(self, noise_window, threshold, t1, t2, radius, do_align):
        try:
            fs = self.fs

            # ── Modo B-format directo (sin conversión A→B) ────────────────────
            if self.bformat_input is not None:
                bformat = self.bformat_input
                self.bformat = bformat #agregado gpt
                onset = detect_onset_noise_floor(
                    bformat.W, fs,
                    noise_window_ms=noise_window,
                    threshold_db=threshold)

            # ── Modo A-format → B-format ──────────────────────────────────────
            else:
                signals = self.signals
                if do_align:
                    aligned, onsets, common_onset = align_aformat_channels(
                        signals, fs,
                        noise_window_ms=noise_window,
                        threshold_db=threshold)
                    aligned_fine, _ = fine_align_channels(
                        aligned, fs, common_onset,
                        reference_channel="LF",
                        search_radius_ms=radius,
                        correlation_window_ms=30.0)
                    bformat = aformat_to_bformat(aligned_fine, fs)
                    onset = common_onset
                else:
                    bformat = aformat_to_bformat(signals, fs)
                    onset = detect_onset_noise_floor(
                        bformat.W, fs,
                        noise_window_ms=noise_window,
                        threshold_db=threshold)
                self.bformat = bformat

            lf_vals = {}
            for fc in OCTAVE_BANDS_HZ:
                Wf = octave_band_filter(bformat.W, fc, fs)
                Yf = octave_band_filter(bformat.Y, fc, fs)
                lf_vals[fc] = compute_LF_band(Wf, Yf, fs,
                                               onset=onset, t1_ms=t1, t2_ms=t2)

            mean_lf = float(np.nanmean(list(lf_vals.values())))
            onset_ms = onset / fs * 1000

            self.after(0, self._update_results, lf_vals, mean_lf, onset_ms)

        except Exception as e:
            self.after(0, self._show_error, str(e))

    def _update_results(self, lf_vals, mean_lf, onset_ms):
        # Tabla
        for fc, lf in lf_vals.items():
            if lf > 1.0:
                color = "#f44336"   # rojo
            elif lf > 0.5:
                color = "#ff9800"   # naranja
            else:
                color = "white"
            self.table_labels[fc].configure(text=f"{lf:.3f}", text_color=color)
        self.lbl_mean.configure(text=f"{mean_lf:.3f}")

        # Plot
        self.ax.clear()
        self._draw_empty_plot()
        freqs = list(lf_vals.keys())
        vals = list(lf_vals.values())
        label = "LF (B-format importado)" if self.bformat_input is not None else "LF pipeline (A→B)"
        self.ax.plot(freqs, vals, "o-", color="#4da6ff", linewidth=2,
                     markersize=8, label=label, zorder=3)

        # Zona de referencia LF típico (0.05–0.35)
        self.ax.axhspan(0.05, 0.35, alpha=0.08, color="#4da6ff", label="Rango típico (0.05–0.35)")
        self.ax.axhline(y=1.0, color="#f44336", linestyle="--",
                        linewidth=0.9, alpha=0.7, label="LF = 1 (límite físico)")

        self.ax.legend(facecolor="#3a3a3a", labelcolor="#cccccc", fontsize=9, framealpha=0.7)
        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()
 
#agregado gpt
        self.btn_direction.configure(state="normal")

        # Estado
        self.lbl_result_status.configure(
            text=f"✓ Procesado   |   Onset: {onset_ms:.1f} ms   |   LF medio: {mean_lf:.3f}",
            text_color="#4caf50")
        self.btn_process.configure(state="normal", text="▶   Procesar")
        # Exportar solo si el input fue A-format (no tiene sentido re-exportar un B-format)
        if self.bformat_input is None:
            self.btn_export.configure(state="normal")
        self.processing = False

    def _show_error(self, msg):
        self.lbl_result_status.configure(text=f"Error: {msg}", text_color="#f44336")
        self.btn_process.configure(state="normal", text="▶   Procesar")
        self.processing = False
        self.btn_direction.configure(state="disabled") #agregado gpt
        messagebox.showerror("Error de procesamiento", msg)

    # ── Exportar B-format ─────────────────────────────────────────────────────
    def _export_bformat(self):
        if self.bformat is None:
            return
        path = filedialog.asksaveasfilename(
            title="Exportar B-format",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")])
        if not path:
            return
        try:
            export_bformat_wav(self.bformat, path, layout="interleaved", order="WYZX")
            messagebox.showinfo("Exportado", f"B-format guardado en:\n{path}")
        except Exception as e:
            messagebox.showerror("Error al exportar", str(e))

#agregado gpt
    def show_directionality(self):
    
        if self.bformat is None:
            messagebox.showwarning(
                "Spatial Metrics",
                "Run the LF analysis first."
            )
            return
    
        try:
            direction = analyze_directionality(
                self.bformat,
                window_ms=1.0,
            )
            plt.close("all")
            plot_directional_panel(
                self.bformat,
                direction,
            )
    
            plt.show()
    
        except Exception as e:
            messagebox.showerror(
                "Spatial Metrics",
                str(e)
            )

# ── Texto de instrucciones ─────────────────────────────────────────────────────
INSTRUCTIONS_TEXT = """\
LF AMBISONICS TOOLKIT — Guía de uso
=====================================

FLUJO DE TRABAJO
─────────────────
1. Seleccioná el modo de importación:
   • Archivo multicanal (4ch): un único WAV de 4 canales en orden LF / RF / LB / RB.
   • 4 archivos mono: seleccionás cada canal por separado (LF, RF, LB, RB).

2. Cargá los archivos con "Seleccionar archivo(s)".
   La aplicación informa la frecuencia de muestreo y la duración detectada.
   Si los canales tienen distinta longitud, se avisa para activar la alineación.

3. Ajustá los parámetros según tu caso (ver más abajo).

4. Presioná ▶ Procesar. El pipeline ejecuta:
   a) Alineación temporal en 2 etapas (si está activada):
      - Etapa gruesa: detecta el onset por canal y recorta el silencio previo.
      - Etapa fina: alineación a nivel de muestra por correlación cruzada.
   b) Conversión A-format → B-format (matriz tetraédrica estándar).
   c) Detección de onset sobre el canal W del B-format.
   d) Filtrado por bandas de octava (125–4000 Hz, Butterworth orden 3, fase cero).
   e) Cálculo de LF por banda según ISO 3382-1:
        Numerador   = ∫[onset+5ms, onset+80ms] Y²(t) dt
        Denominador = ∫[onset,     onset+80ms] W²(t) dt

5. Los resultados se muestran en la tabla y en el gráfico.
   Podés ajustar parámetros y volver a procesar sin recargar el archivo.

6. Exportá el B-format (botón inferior) si necesitás comparar con EASERA u otro software.


PARÁMETROS
───────────
• Ventana de ruido (ms)
  Duración del pre-roll de silencio al inicio del archivo, usada para estimar el
  piso de ruido antes del onset.
  → Usar 50 ms para archivos exportados del DAW (caso Usina del Arte).
  → Usar 5 ms si el onset está muy cerca del inicio del archivo (caso Pori, Catedral).

• Umbral onset (dB)
  Nivel sobre el piso de ruido a partir del cual se considera que comenzó la RIR.
  → Típico: 15 dB.
  → Usar 20 dB si hay pre-ringing antes del sonido directo (ej.: J12 Usina del Arte).

• t₁ integración (ms)
  Inicio de la ventana de integración del numerador (canal Y).
  ISO 3382-1: 5 ms (excluye el sonido directo, que no contribuye a la energía lateral).

• t₂ integración (ms)
  Fin de la ventana de integración. ISO 3382-1: 80 ms.

• Alineación temporal de canales
  Activa el pipeline de alineación en 2 etapas para corregir desfases entre cápsulas.
  → SIEMPRE activar para archivos exportados del DAW por pistas separadas
    (caso Usina del Arte — SP200).
  → Desactivar para archivos multicanal ya sincronizados (caso Catedral, archivos
    de repositorios externos como Pori).

• Radio de búsqueda (ms)
  Ventana de búsqueda para la alineación fina por correlación cruzada.
  → Valor típico: 5 ms. Aumentar si los desfases entre canales son mayores.


INTERPRETACIÓN DE RESULTADOS
──────────────────────────────
• Rango típico de LF: 0.05 – 0.35 (sombreado azul en el gráfico).
• LF > 0.5  → naranja (poco probable físicamente; revisar parámetros o alineación).
• LF > 1.0  → rojo (físicamente imposible; revisar onset, alineación y archivos).
• La banda de 4 kHz frecuentemente supera 1.0 sin corrección frecuencial del
  fabricante. Esto no es un bug: es una limitación conocida de la conversión
  A→B frecuencia-independiente. Reportar las bandas 125–2000 Hz como primarias.


COMPARACIÓN CON SOFTWARE DE REFERENCIA (EASERA, DIRAC, etc.)
──────────────────────────────────────────────────────────────
1. Procesá la RIR con los parámetros deseados.
2. Exportá el B-format como WAV 4 canales (orden W, X, Y, Z).
3. Importá ese archivo en el software de referencia y comparar los valores de LF.
   Acuerdo esperado: < 10% en bandas 125–2000 Hz.


LIMITACIONES CONOCIDAS
───────────────────────
• Sin corrección frecuencial del fabricante (EQ propietaria del SoundField MKV/SP200).
  Efecto más notable en 125–500 Hz y en 4 kHz.
• La alineación usa detección de onset por piso de ruido; si el pre-roll es
  muy corto, reducir el parámetro "Ventana de ruido" a 5–10 ms.
"""


if __name__ == "__main__":
    app = LFApp()
    app.mainloop()
