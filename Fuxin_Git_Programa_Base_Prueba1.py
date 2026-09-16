# ================================================================
# FUXING CLI / CLOUD EDITION
# ================================================================
# Derivada de Completo_GUI prueba8-2(2).py.
# Mantiene los motores de histórico, features, ML, redes, optimización,
# Monte Carlo, backtesting y predicción. Se elimina únicamente la GUI.
# La ejecución diaria y semanal comparten histórico + preparación +
# entrenamiento y después ejecutan sus cálculos específicos en paralelo.
# ================================================================
# ================================================================
# BONOLOTO AI RESEARCH LAB - MAXIMUM EDITION
# ================================================================
# Versión revisada y reforzada sobre el programa original.
#
# Objetivos principales:
#   1) Histórico desde 1988 y actualización incremental.
#   2) Validación, deduplicación y auditoría del dataset.
#   3) Feature engineering sin fuga de información.
#   4) Modelos estadísticos + opcionalmente PyTorch.
#   5) Backtesting walk-forward antes de confiar en cualquier regla.
#   6) Generación de combinaciones mediante ranking y Monte Carlo.
#   7) Penalización de combinaciones excesivamente populares.
#   8) Interfaz Tkinter conservando el flujo original.
#
# IMPORTANTE:
# La Bonoloto es un juego aleatorio. Este programa NO puede conocer
# la combinación futura ni convertir una señal histórica en certeza.
# Su función es investigación estadística y selección sistemática.
# ================================================================

import os
import re
import csv
import json
import math
import time
import pickle
import random
import hashlib
import threading
import traceback
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


# ----------------------------------------------------------------
# Dependencias opcionales
# ----------------------------------------------------------------
TORCH_AVAILABLE = False
SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    torch = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import log_loss, roc_auc_score
    ULTIMATE_SKLEARN = True
    SKLEARN_AVAILABLE = True
except Exception:
    ULTIMATE_SKLEARN = False

ULTIMATE_SEED = 20260901

# ================================================================
# CONFIGURACIÓN
# ================================================================

MAX_NUM = 49
NUMBERS_PER_DRAW = 6
START_YEAR = 1988  # Histórico completo: desde 1988
SEQ_LEN = 60

CURRENT_YEAR = datetime.now().year

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "bonoloto_data")
MODEL_DIR = os.path.join(BASE_DIR, "bonoloto_models")
CACHE_FILE = os.path.join(DATA_DIR, "bonoloto_history.pkl")
CSV_FILE = os.path.join(DATA_DIR, "bonoloto_history.csv")
AUDIT_FILE = os.path.join(DATA_DIR, "dataset_audit.json")
BACKTEST_FILE = os.path.join(DATA_DIR, "backtest_results.json")
PRED_FILE = os.path.join(BASE_DIR, "predicciones_avanzadas.txt")
LOG_FILE = os.path.join(BASE_DIR, "execution_log.txt")
PREDICTION_STATE_FILE = os.path.join(BASE_DIR, "fuxing_prediction_state.json")
WEEKLY_STATE_FILE = os.path.join(BASE_DIR, "fuxing_weekly_state.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = "cpu"
if TORCH_AVAILABLE and torch.cuda.is_available():
    DEVICE = "cuda"

# Parámetros conservadores por defecto.
# No se afirma que sean "óptimos" hasta que el backtest lo demuestre.
OPTIMAL_PARAMS = {
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "epochs": 35,
    "batch_size": 128,
    "dropout": 0.15,
    "patience": 7,
}

MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5,
    "jun": 6, "jul": 7, "ago": 8, "sep": 9, "oct": 10,
    "nov": 11, "dic": 12,
}

lista_logs = None
ventana = None
result_var = None
FEATURE_SIZE = 0
CURRENT_APP = None


# ================================================================
# LOGGING
# ================================================================

# Archivos de estado/modelos de la edición MAXIMUM (compatibles con el original)
ULTIMATE_MODEL_DIR = MODEL_DIR
ULTIMATE_STATE_FILE = os.path.join(ULTIMATE_MODEL_DIR, "ultimate_training_state.json")
ULTIMATE_SK_FILE = os.path.join(ULTIMATE_MODEL_DIR, "sklearn_models.pkl")
ULTIMATE_TORCH_FILE = os.path.join(ULTIMATE_MODEL_DIR, "bonoloto_transformer.pt")
os.makedirs(ULTIMATE_MODEL_DIR, exist_ok=True)

def _safe_json_write(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)

class AdvancedLogger:
    def __init__(self):
        self.log_buffer = []
        self.start_time = datetime.now()

    def log(self, msg, level="INFO"):
        timestamp = datetime.now()
        elapsed = (timestamp - self.start_time).total_seconds()
        text = f"[{timestamp:%H:%M:%S}] [{level}] {msg} (Δ{elapsed:.1f}s)"
        self.log_buffer.append(text)
        try:
            if lista_logs is not None:
                lista_logs.after(0, self._ui_log, text)
            else:
                print(text)
        except Exception:
            print(text)

    def _ui_log(self, text):
        try:
            tag = "log_info"
            upper = text.upper()
            if "[ERROR]" in upper:
                tag = "log_error"
            elif "[WARNING]" in upper:
                tag = "log_warning"
            elif "[NETWORK]" in upper or "RED / SCRAPING" in upper or "CONEXIÓN ESTABLECIDA" in upper:
                tag = "log_data"
            elif "AÑO " in upper or "SORTEOS" in upper or "HISTÓRICO" in upper:
                tag = "log_data"
            elif "ENTREN" in upper or "MODELO" in upper or "PYTORCH" in upper or "SCIKIT" in upper:
                tag = "log_model"
            elif "OPTIM" in upper or "BACKTEST" in upper:
                tag = "log_opt"
            elif "PREDIC" in upper or "COMBINACION" in upper or "COMBINACIONES" in upper:
                tag = "log_prediction"
            elif "AUDITOR" in upper or "ESTADÍST" in upper or "ENTROP" in upper:
                tag = "log_audit"
            lista_logs.insert(tk.END, text + "\n", tag)
            lista_logs.see(tk.END)
        except Exception:
            pass

    def save_logs(self):
        try:
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(self.log_buffer))
        except Exception:
            pass


logger = AdvancedLogger()
log = logger.log


# ================================================================
# UTILIDADES BÁSICAS
# ================================================================

def parse_fecha_es(texto):
    if not texto:
        return None
    try:
        txt = texto.lower().strip()
        txt = re.sub(
            r"^(lun|mar|mié|mie|jue|vie|sáb|sab|dom)[,.\s]+",
            "",
            txt,
        )
        txt = re.sub(r"\s+", " ", txt)
        partes = txt.split()
        if len(partes) != 3:
            return None
        dia = int(re.sub(r"\D", "", partes[0]))
        mes = MESES.get(partes[1].strip("."))
        anio = int(partes[2])
        if mes is None:
            return None
        return datetime(anio, mes, dia)
    except Exception:
        return None


def validate_draw(draw):
    if draw is None or len(draw) != 6:
        return False
    try:
        nums = [int(x) for x in draw]
    except Exception:
        return False
    return len(set(nums)) == 6 and all(1 <= x <= 49 for x in nums)


def normalize_draw(draw):
    return tuple(sorted(int(x) for x in draw))


def one_hot_draw(draw):
    v = np.zeros(MAX_NUM, dtype=np.float32)
    for n in draw:
        if 1 <= n <= MAX_NUM:
            v[n - 1] = 1.0
    return v


def even_count(draw):
    return sum(x % 2 == 0 for x in draw)


def odd_count(draw):
    return 6 - even_count(draw)


def low_count(draw):
    return sum(x <= 24 for x in draw)


def high_count(draw):
    return 6 - low_count(draw)


def draw_sum(draw):
    return int(sum(draw))


def draw_range(draw):
    return max(draw) - min(draw)


def calculate_global_frequency(data):
    freq = np.zeros(MAX_NUM, dtype=np.float64)
    for _, draw in data:
        for n in draw:
            if 1 <= n <= MAX_NUM:
                freq[n - 1] += 1
    return freq


def calculate_entropy_feature(freq):
    s = float(np.sum(freq))
    if s <= 0:
        return 0.0
    p = freq / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def chi_square_test(freq):
    total = float(np.sum(freq))
    if total <= 0:
        return 0.0
    expected = total / MAX_NUM
    return float(np.sum((freq - expected) ** 2 / expected))


def safe_zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - np.mean(x)) / (np.std(x) + 1e-12)


def softmax(x, temperature=1.0):
    x = np.asarray(x, dtype=float)
    temperature = max(float(temperature), 1e-6)
    z = (x - np.max(x)) / temperature
    e = np.exp(np.clip(z, -60, 60))
    return e / (e.sum() + 1e-12)


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


# ================================================================
# DATA SCRAPER ROBUSTO
# ================================================================

class DataScraper:
    """
    Descarga el histórico desde 1988 y lo mantiene localmente.

    Fuente primaria: páginas anuales de elgordo.com.
    La estructura se valida antes de aceptar cada registro.

    El caché ya no bloquea la actualización: se conserva el histórico
    anterior y se incorporan los años que falten o hayan cambiado.
    """

    BASE_URL = "https://www.elgordo.com/es/resultados/bonoloto--a%C3%B1o-{year}"
    # Fuente secundaria de respaldo. Tiene tablas anuales completas desde 1988.
    SECONDARY_BASE_URL = "https://lotocrack.com/historico-de-resultados/bonoloto/resultados-{year}/"

    def __init__(self, start_year=START_YEAR, end_year=None):
        self.start_year = int(start_year)
        self.end_year = int(end_year or datetime.now().year)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept-Language": "es-ES,es;q=0.9",
        })

    def _request(self, url, attempts=4):
        last_error = None
        for attempt in range(attempts):
            try:
                log(
                    f"RED / SCRAPING: conectando con {url} "
                    f"(intento {attempt + 1}/{attempts})",
                    "NETWORK"
                )
                r = self.session.get(url, timeout=35)
                if r.status_code == 200 and r.text:
                    log(
                        f"RED / SCRAPING: conexión establecida — {url} "
                        f"(HTTP {r.status_code})",
                        "NETWORK"
                    )
                    return r
                last_error = RuntimeError(f"HTTP {r.status_code}")
                log(
                    f"RED / SCRAPING: respuesta no válida de {url} "
                    f"(HTTP {r.status_code})",
                    "WARNING"
                )
            except Exception as e:
                last_error = e
                log(
                    f"RED / SCRAPING: no se pudo conectar con {url}: {e}",
                    "WARNING"
                )
            time.sleep(1.5 * (attempt + 1))
        raise last_error or RuntimeError("Error HTTP")

    def scrape_with_cache(self, force_refresh=False, progress_callback=None):
        old = []
        if os.path.exists(CACHE_FILE) and not force_refresh:
            try:
                with open(CACHE_FILE, "rb") as f:
                    old = pickle.load(f)
                old = self._clean_data(old)
                log(f"Caché local: {len(old)} sorteos válidos")
            except Exception as e:
                log(f"Caché no utilizable: {e}", "WARNING")
                old = []

        # IMPORTANTE: no basta con comprobar si un año existe en caché.
        # Una caché puede contener un año pero estar incompleta (esto era
        # precisamente lo que estaba ocurriendo: 8.612 sorteos con todos
        # los años presentes, pero cientos de sorteos ausentes).
        #
        # Por ello el botón 1 vuelve a contrastar TODOS los años 1988->actual.
        # Los datos ya existentes se conservan y se deduplican al fusionar.
        # Así nunca volvemos a quedarnos únicamente con 2026.
        years = list(range(self.start_year, self.end_year + 1))

        fresh = []
        if years:
            log(f"Contrastando histórico COMPLETO: {years[0]}-{years[-1]} ({len(years)} años)")
            with ThreadPoolExecutor(max_workers=min(8, len(years))) as ex:
                futures = {ex.submit(self.scrape_year, y): y for y in years}
                completed = 0
                total = len(futures)
                for fut in as_completed(futures):
                    year = futures[fut]
                    completed += 1
                    try:
                        rows = fut.result()
                        fresh.extend(rows)
                        log(f"Año {year}: {len(rows)} sorteos válidos ({completed}/{total})")
                        if progress_callback:
                            progress_callback(completed, total, year, len(rows), None)
                    except Exception as e:
                        log(f"Año {year}: {e}", "WARNING")
                        if progress_callback:
                            progress_callback(completed, total, year, 0, str(e))

        merged = self._merge(old, fresh)

        # Si el año actual no devuelve nada, conservamos el caché.
        # Nunca se sustituye silenciosamente un histórico bueno por uno vacío.
        if len(merged) < len(old):
            log("La actualización produciría menos datos; se conserva el caché.", "WARNING")
            merged = old

        self._save(merged)
        self._audit_coverage(merged)
        return merged

    def scrape_all_years(self):
        return self.scrape_with_cache(force_refresh=True)

    def scrape_year(self, year):
        # NUEVA FUENTE PRINCIPAL: LotoCrack publica páginas anuales con
        # fecha completa + seis números ganadores + C/R. Es mucho más
        # estable para el histórico 1988-actual que depender de selectores
        # CSS de una sola web.
        try:
            lotocrack_rows = self._scrape_year_lotocrack(year)
            if len(lotocrack_rows) >= 100:
                return lotocrack_rows
            if lotocrack_rows:
                log(
                    f"Año {year}: LotoCrack devolvió sólo "
                    f"{len(lotocrack_rows)} registros; se prueba fuente alternativa",
                    "WARNING"
                )
        except Exception as e:
            log(f"Año {year}: LotoCrack no disponible: {e}; usando fuente alternativa", "WARNING")

        url = self.BASE_URL.format(year=year)
        r = self._request(url)
        soup = BeautifulSoup(r.text, "html.parser")
        resultados = []

        filas = soup.select("div.pseudo-table-row")
        if not filas:
            # Fallback genérico para cambios menores de HTML.
            filas = soup.select(
                "[class*='history-results'], "
                "tr, .result, .results-row"
            )

        # Parser de respaldo MUY robusto: la web publica cada sorteo como
        # combinación seguida de su fecha. No dependemos de nombres CSS.
        text_lines = [
            re.sub(r"\s+", " ", x.strip())
            for x in soup.get_text("\n", strip=True).splitlines()
            if x.strip()
        ]
        combo_re = re.compile(
            r"^((?:\d{1,2}-){5}\d{1,2})(?:-C\d{1,2})?(?:\s+-?R\d{1,2})?$",
            re.IGNORECASE,
        )
        fallback_rows = []
        for i, line in enumerate(text_lines):
            m = combo_re.match(line)
            if not m:
                continue
            nums = [int(x) for x in m.group(1).split("-")]
            if not validate_draw(nums):
                continue
            fecha = None
            for nxt in text_lines[i + 1:i + 6]:
                fecha = parse_fecha_es(nxt)
                if fecha:
                    break
            if fecha and fecha.year == int(year):
                fallback_rows.append((fecha, sorted(nums)))

        # Si el parser estructural no encuentra nada, usamos exclusivamente
        # el parser de texto. Si encuentra algo, combinamos ambos y deduplicamos.
        if fallback_rows:
            resultados.extend(fallback_rows)

        for fila in filas:
            try:
                fecha_el = (
                    fila.select_one(".history-results-date")
                    or fila.select_one("[class*='results-date']")
                )
                comb_el = (
                    fila.select_one(".history-results-combi")
                    or fila.select_one("[class*='results-combi']")
                )

                if not fecha_el or not comb_el:
                    continue

                fecha = parse_fecha_es(fecha_el.get_text(" ", strip=True))
                if not fecha:
                    continue

                # Se prefieren spans del bloque de combinación.
                texts = [
                    x.get_text(strip=True)
                    for x in comb_el.select("span")
                ]
                nums = []
                for txt in texts:
                    if re.fullmatch(r"\d{1,2}", txt):
                        n = int(txt)
                        if 1 <= n <= 49:
                            nums.append(n)

                # Fallback: extraer números de 1-49 del texto, limitando
                # a la primera combinación de seis.
                if len(nums) < 6:
                    raw = comb_el.get_text(" ", strip=True)
                    candidates = [
                        int(x) for x in re.findall(r"(?<!\d)\d{1,2}(?!\d)", raw)
                        if 1 <= int(x) <= 49
                    ]
                    nums = candidates[:6]

                nums = sorted(set(nums))
                if validate_draw(nums):
                    resultados.append((fecha, nums))
            except Exception:
                continue

        resultados = self._clean_data(resultados)

        # Si la fuente primaria devuelve pocos/ningún registro, usamos una
        # segunda fuente independiente con tablas anuales. También la usamos
        # cuando la primaria parece incompleta.
        # Para años antiguos esperamos al menos ~150 sorteos; para años
        # modernos el histórico suele superar ampliamente esa cifra.
        if len(resultados) < 150:
            try:
                secondary = self._scrape_year_secondary(year)
                if len(secondary) > len(resultados):
                    log(
                        f"Año {year}: fuente secundaria aporta "
                        f"{len(secondary)} sorteos frente a {len(resultados)} de la primaria"
                    )
                    resultados = secondary
                elif secondary:
                    # Si ambas contienen datos, combinamos por si una fuente
                    # tiene alguna fecha que falte en la otra.
                    resultados = self._clean_data(resultados + secondary)
            except Exception as e:
                log(f"Año {year}: respaldo histórico no disponible: {e}", "WARNING")

        return self._clean_data(resultados)

    def _scrape_year_lotocrack(self, year):
        """Extrae el histórico anual desde LotoCrack.

        Formato observado: una línea con la fecha completa y, justo después,
        una línea del tipo ``1 22 34 36 37 45 C-32 R--``.
        Se toman exclusivamente los seis primeros números.
        """
        url = self.SECONDARY_BASE_URL.format(year=int(year))
        r = self._request(url)
        soup = BeautifulSoup(r.text, "html.parser")
        lines = [
            re.sub(r"\s+", " ", x.strip())
            for x in soup.get_text("\n", strip=True).splitlines()
            if x.strip()
        ]

        months = {
            "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
            "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
            "septiembre": 9, "setiembre": 9, "octubre": 10,
            "noviembre": 11, "diciembre": 12,
        }
        date_re = re.compile(
            r"\b(\d{1,2})\s+de\s+([A-Za-zÁÉÍÓÚáéíóúñÑ]+)\s+de\s+(\d{4})\b",
            re.IGNORECASE,
        )
        combo_re = re.compile(
            r"^\s*(\d{1,2}(?:\s+\d{1,2}){5})\s+C-\d{1,2}\s+R",
            re.IGNORECASE,
        )

        rows = []
        for i, line in enumerate(lines):
            dm = date_re.search(line)
            if not dm:
                continue
            try:
                day = int(dm.group(1))
                month = months.get(dm.group(2).lower())
                y = int(dm.group(3))
                if month is None or y != int(year):
                    continue
                fecha = datetime(y, month, day)
            except Exception:
                continue

            # La combinación aparece inmediatamente después de la fecha,
            # aunque admitimos hasta 3 líneas de margen por cambios HTML.
            for nxt in lines[i + 1:i + 4]:
                cm = combo_re.match(nxt)
                if not cm:
                    continue
                nums = [int(x) for x in re.findall(r"\d{1,2}", cm.group(1))]
                if validate_draw(nums):
                    rows.append((fecha, sorted(nums)))
                break

        rows = self._clean_data(rows)
        log(f"Año {year}: LotoCrack -> {len(rows)} sorteos válidos")
        return rows

    def _scrape_year_secondary(self, year):
        """Alias de compatibilidad para el antiguo nombre del respaldo."""
        return self._scrape_year_lotocrack(year)

    def _clean_data(self, data):
        unique = {}
        for row in data or []:
            try:
                fecha, draw = row
                fecha = fecha if isinstance(fecha, datetime) else pd.to_datetime(fecha).to_pydatetime()
                draw = list(normalize_draw(draw))
                if validate_draw(draw):
                    unique[(fecha.date().isoformat(), tuple(draw))] = (fecha, draw)
            except Exception:
                continue
        return sorted(unique.values(), key=lambda x: x[0])

    def _merge(self, a, b):
        return self._clean_data(list(a or []) + list(b or []))

    def _save(self, data):
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(data, f)

        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["fecha", "n1", "n2", "n3", "n4", "n5", "n6"])
            for fecha, draw in data:
                w.writerow([fecha.strftime("%Y-%m-%d"), *draw])

    def _audit_coverage(self, data):
        if not data:
            return
        years = Counter(d.year for d, _ in data)
        audit = {
            "generated": datetime.now().isoformat(),
            "total_draws": len(data),
            "min_date": data[0][0].strftime("%Y-%m-%d"),
            "max_date": data[-1][0].strftime("%Y-%m-%d"),
            "years": dict(sorted(years.items())),
            "expected_start": START_YEAR,
            "expected_end": self.end_year,
        }
        with open(AUDIT_FILE, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2, ensure_ascii=False)

        log(
            f"Histórico final: {len(data):,} sorteos | "
            f"{data[0][0]:%d/%m/%Y} -> {data[-1][0]:%d/%m/%Y}"
        )


# ================================================================
# AUDITORÍA ESTADÍSTICA
# ================================================================

class LotteryStatistics:
    def __init__(self, data):
        self.data = data
        self.draws = [tuple(d) for _, d in data]

    def frequency(self, window=None):
        rows = self.draws[-window:] if window else self.draws
        freq = np.zeros(MAX_NUM, dtype=float)
        for draw in rows:
            for n in draw:
                freq[n - 1] += 1
        return freq

    def normalized_frequency(self, window=None):
        f = self.frequency(window)
        return f / (f.sum() + 1e-12)

    def absence(self):
        result = np.zeros(MAX_NUM, dtype=int)
        for n in range(1, MAX_NUM + 1):
            gap = 0
            for draw in reversed(self.draws):
                if n in draw:
                    break
                gap += 1
            result[n - 1] = gap
        return result

    def pair_frequency(self, window=None):
        rows = self.draws[-window:] if window else self.draws
        c = Counter()
        for draw in rows:
            for pair in combinations(draw, 2):
                c[pair] += 1
        return c

    def triple_frequency(self, window=None):
        rows = self.draws[-window:] if window else self.draws
        c = Counter()
        for draw in rows:
            for tri in combinations(draw, 3):
                c[tri] += 1
        return c

    def sums(self, window=None):
        rows = self.draws[-window:] if window else self.draws
        return np.array([sum(d) for d in rows], dtype=float)

    def parity_counts(self, window=None):
        rows = self.draws[-window:] if window else self.draws
        return Counter(sum(n % 2 == 0 for n in d) for d in rows)

    def low_high_counts(self, window=None):
        rows = self.draws[-window:] if window else self.draws
        return Counter(sum(n <= 24 for n in d) for d in rows)

    def consecutive_counts(self, window=None):
        rows = self.draws[-window:] if window else self.draws
        out = []
        for d in rows:
            out.append(sum(d[i + 1] == d[i] + 1 for i in range(5)))
        return np.array(out, dtype=int)

    def decade_counts(self, window=None):
        rows = self.draws[-window:] if window else self.draws
        c = Counter()
        for d in rows:
            for n in d:
                c[min((n - 1) // 10, 4)] += 1
        return c

    def summary(self):
        freq = self.frequency()
        return {
            "total_draws": len(self.draws),
            "min_date": self.data[0][0].isoformat() if self.data else None,
            "max_date": self.data[-1][0].isoformat() if self.data else None,
            "chi_square": chi_square_test(freq),
            "entropy": calculate_entropy_feature(freq),
            "mean_sum": float(np.mean(self.sums())) if self.draws else 0,
            "std_sum": float(np.std(self.sums())) if self.draws else 0,
            "mean_range": float(np.mean([
                draw_range(d) for d in self.draws
            ])) if self.draws else 0,
        }


def randomness_audit(data):
    stats_obj = LotteryStatistics(data)
    f = stats_obj.frequency()
    log(
        f"Auditoría: sorteos={len(data)}, "
        f"chi2={chi_square_test(f):.3f}, "
        f"entropía={calculate_entropy_feature(f):.5f}"
    )

    z = safe_zscore(f)
    outliers = [i + 1 for i, x in enumerate(z) if abs(x) >= 2.5]
    if outliers:
        log(f"Números con desviación histórica >=2.5σ: {outliers}", "INFO")

    return {
        "frequency": f,
        "zscore": z,
        "outliers": outliers,
    }


def dataset_audit(data):
    errors = []
    seen = set()
    for fecha, draw in data:
        key = (fecha.date().isoformat(), tuple(draw))
        if not validate_draw(draw):
            errors.append((fecha, draw, "invalid_draw"))
        if key in seen:
            errors.append((fecha, draw, "duplicate"))
        seen.add(key)

    dates = [x[0] for x in data]
    report = {
        "records": len(data),
        "errors": len(errors),
        "duplicates_or_invalid": errors[:100],
        "min_date": min(dates).isoformat() if dates else None,
        "max_date": max(dates).isoformat() if dates else None,
    }

    with open(AUDIT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    log(f"Auditoría dataset: {len(data)} registros, {len(errors)} incidencias")
    return report


# ================================================================
# FEATURE ENGINEERING
# ================================================================

class FeatureEngineeringExtreme:
    @staticmethod
    def create_cyclical_features(fecha):
        day_of_year = fecha.timetuple().tm_yday
        return np.array([
            np.sin(2 * np.pi * day_of_year / 365.25),
            np.cos(2 * np.pi * day_of_year / 365.25),
            np.sin(2 * np.pi * fecha.weekday() / 7),
            np.cos(2 * np.pi * fecha.weekday() / 7),
        ], dtype=np.float32)

    @staticmethod
    def rolling_number_features(draws, index, windows=(10, 25, 50, 100, 250)):
        """
        Features calculadas exclusivamente con sorteos anteriores al índice.
        """
        out = []
        for w in windows:
            rows = draws[max(0, index - w):index]
            freq = np.zeros(MAX_NUM, dtype=float)
            for d in rows:
                for n in d:
                    freq[n - 1] += 1
            denom = max(1, len(rows))
            # frecuencia por sorteo, no por total de casillas
            out.extend((freq / denom).tolist())
        return np.asarray(out, dtype=np.float32)

    @staticmethod
    def recency_features(draws, index):
        gaps = np.full(MAX_NUM, index + 1, dtype=float)
        rows = draws[:index]
        for back, d in enumerate(reversed(rows), 1):
            for n in d:
                if gaps[n - 1] > index + 1:
                    gaps[n - 1] = back
        # compresión logarítmica para que un gap enorme no domine
        return np.log1p(gaps).astype(np.float32)

    @staticmethod
    def pair_strength_features(draws, index, window=250):
        rows = draws[max(0, index - window):index]
        pair = Counter()
        for d in rows:
            for a, b in combinations(d, 2):
                pair[(a, b)] += 1

        f = np.zeros(MAX_NUM, dtype=float)
        for (a, b), count in pair.items():
            f[a - 1] += count
            f[b - 1] += count
        if rows:
            f /= len(rows)
        return f.astype(np.float32)

    @staticmethod
    def number_features(data, index):
        draws = [d for _, d in data]
        fecha = data[index][0]
        current = draws[index]

        f10 = FeatureEngineeringExtreme.rolling_number_features(
            draws, index, (10, 25, 50, 100, 250)
        )
        rec = FeatureEngineeringExtreme.recency_features(draws, index)
        pair = FeatureEngineeringExtreme.pair_strength_features(
            draws, index, 250
        )

        # Estadísticas recientes de forma que no miren el futuro.
        recent = draws[max(0, index - 50):index]
        if recent:
            sums = np.array([sum(d) for d in recent], dtype=float)
            ranges = np.array([draw_range(d) for d in recent], dtype=float)
            parity = np.array([even_count(d) for d in recent], dtype=float)
            low = np.array([low_count(d) for d in recent], dtype=float)
            stats = np.array([
                np.mean(sums), np.std(sums),
                np.mean(ranges), np.std(ranges),
                np.mean(parity), np.mean(low),
            ], dtype=np.float32)
        else:
            stats = np.zeros(6, dtype=np.float32)

        cyc = FeatureEngineeringExtreme.create_cyclical_features(fecha)

        # Frecuencia histórica hasta index.
        all_freq = np.zeros(MAX_NUM, dtype=float)
        for d in draws[:index]:
            for n in d:
                all_freq[n - 1] += 1
        total = all_freq.sum()
        freq = all_freq / total if total else all_freq

        return np.concatenate([
            f10,
            rec,
            pair,
            freq.astype(np.float32),
            stats,
            cyc,
        ]).astype(np.float32)


def build_dataset(data, seq_len=SEQ_LEN):
    global FEATURE_SIZE

    if len(data) <= seq_len + 20:
        raise ValueError(
            f"No hay suficientes sorteos para construir el dataset: {len(data)}"
        )

    draws = [tuple(d) for _, d in data]
    X_seq, X_feat, Y = [], [], []

    # El índice i representa el último sorteo conocido.
    # El target es i+1. Nunca se incorpora el target a los features.
    for i in range(seq_len - 1, len(draws) - 1):
        seq = [
            one_hot_draw(draws[k])
            for k in range(i - seq_len + 1, i + 1)
        ]
        features = FeatureEngineeringExtreme.number_features(data, i)

        X_seq.append(seq)
        X_feat.append(features)
        Y.append(one_hot_draw(draws[i + 1]))

    X_seq = np.asarray(X_seq, dtype=np.float32)
    X_feat = np.asarray(X_feat, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    FEATURE_SIZE = X_feat.shape[1]

    log(
        f"Dataset: X_seq={X_seq.shape}, "
        f"X_feat={X_feat.shape}, Y={Y.shape}"
    )

    return (
        X_seq,
        X_feat,
        Y,
    )


# ================================================================
# MODELO ESTADÍSTICO: NÚCLEO PRINCIPAL
# ================================================================

class StatisticalEngine:
    """
    Motor principal. Se apoya en señales históricas con varias escalas.

    Las señales no se suman ciegamente: cada una se normaliza y puede
    comprobarse posteriormente mediante backtesting.
    """

    def __init__(self, data):
        self.data = data
        self.stats = LotteryStatistics(data)

    def score_numbers(self):
        f_full = self.stats.normalized_frequency()
        f10 = self.stats.normalized_frequency(10)
        f25 = self.stats.normalized_frequency(25)
        f50 = self.stats.normalized_frequency(50)
        f100 = self.stats.normalized_frequency(100)
        f250 = self.stats.normalized_frequency(250)

        absence = self.stats.absence().astype(float)
        absence_z = safe_zscore(absence)

        # Coocurrencia individual: cuántos pares históricos acompaña cada n.
        pair = self.stats.pair_frequency(250)
        pair_score = np.zeros(MAX_NUM, dtype=float)
        for (a, b), c in pair.items():
            pair_score[a - 1] += c
            pair_score[b - 1] += c
        pair_score = pair_score / (pair_score.sum() + 1e-12)

        # Tendencia reciente contra largo plazo.
        trend = (
            0.05 * safe_zscore(f_full) +
            0.10 * safe_zscore(f100) +
            0.15 * safe_zscore(f50) +
            0.25 * safe_zscore(f25) +
            0.25 * safe_zscore(f10)
        )

        # La ausencia se usa como señal débil, no como "debe salir".
        # Evitamos la falacia del jugador: un número atrasado no tiene
        # mayor probabilidad física por estar atrasado.
        recency_signal = np.tanh(absence_z / 2.5) * 0.03

        pair_signal = safe_zscore(pair_score) * 0.05

        raw = trend + recency_signal + pair_signal
        return raw, {
            "full": f_full,
            "10": f10,
            "25": f25,
            "50": f50,
            "100": f100,
            "250": f250,
            "absence": absence,
            "trend": trend,
        }

    def probability_vector(self):
        raw, details = self.score_numbers()
        p = softmax(raw, temperature=1.25)
        return p, details


# ================================================================
# MODELO OPCIONAL PYTORCH
# ================================================================

if TORCH_AVAILABLE:

    class UltraTransformer(nn.Module):
        """
        Transformer pequeño y regularizado.
        La versión original era desproporcionadamente grande para el
        tamaño del dataset. Aquí se prioriza generalización.
        """
        def __init__(self, d_model=128, nhead=8, layers=3, dropout=0.15):
            super().__init__()
            self.proj = nn.Linear(MAX_NUM, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 3,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=layers
            )
            self.norm = nn.LayerNorm(d_model)
            self.out = nn.Linear(d_model, MAX_NUM)

        def forward(self, x):
            x = self.proj(x)
            x = self.encoder(x)
            x = self.norm(x[:, -1])
            return self.out(x)

    class BayesianFeatureNetwork(nn.Module):
        def __init__(self, feature_size, dropout=0.15):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(feature_size, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, MAX_NUM),
            )

        def forward(self, x):
            return self.net(x)

    class UltraBonolotoNet(nn.Module):
        def __init__(self, feature_size):
            super().__init__()
            self.transformer = UltraTransformer()
            self.features = BayesianFeatureNetwork(feature_size)
            self.fusion = nn.Sequential(
                nn.Linear(MAX_NUM * 2, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(256, MAX_NUM),
            )

        def forward(self, seq, feat):
            a = self.transformer(seq)
            b = self.features(feat)
            return self.fusion(torch.cat([a, b], dim=1))


# ================================================================
# MODELO ML OPCIONAL
# ================================================================

class MLNumberModel:
    """
    Modelo por número. Aprende P(n pertenece al próximo sorteo).

    Se entrena solo con observaciones históricas disponibles antes del
    target correspondiente.
    """

    def __init__(self):
        self.models = [None] * MAX_NUM
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_matrix = None

    def fit(self, X, Y):
        if not SKLEARN_AVAILABLE:
            return False

        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        # Un único scaler global para evitar 49 transformaciones distintas.
        self.feature_matrix = self.scaler.fit_transform(X)

        for n in range(MAX_NUM):
            y = Y[:, n].astype(int)
            # Si una clase no existe en un split pequeño, no se puede ajustar.
            if len(np.unique(y)) < 2:
                self.models[n] = None
                continue

            model = LogisticRegression(
                C=0.15,
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
            )
            model.fit(self.feature_matrix, y)
            self.models[n] = model

        return True

    def predict(self, X):
        if not SKLEARN_AVAILABLE or self.scaler is None:
            return None

        Xs = self.scaler.transform(np.asarray(X, dtype=np.float32))
        p = np.zeros(MAX_NUM, dtype=float)

        for n, model in enumerate(self.models):
            if model is None:
                p[n] = 0.5
            else:
                p[n] = model.predict_proba(Xs)[0, 1]

        # Ajuste para que la masa total corresponda a seis posiciones.
        p = p / (p.sum() + 1e-12) * NUMBERS_PER_DRAW
        p = np.clip(p, 1e-9, 1.0)
        return p


# ================================================================
# GENERADOR DE COMBINACIONES
# ================================================================

class CombinationEngine:
    def __init__(self, data, number_probs):
        self.data = data
        self.number_probs = np.asarray(number_probs, dtype=float)
        self.number_probs = self.number_probs / (
            self.number_probs.sum() + 1e-12
        )
        self.stats = LotteryStatistics(data)
        self.pair_freq = self.stats.pair_frequency(500)
        self.triple_freq = self.stats.triple_frequency(500)

        self.historical_combos = {
            tuple(sorted(d)) for d in self.stats.draws
        }

    def structural_score(self, draw):
        draw = tuple(sorted(draw))

        even = even_count(draw)
        low = low_count(draw)
        total = draw_sum(draw)
        consecutive = sum(
            draw[i + 1] == draw[i] + 1 for i in range(5)
        )
        spread = draw_range(draw)

        # Penalizaciones suaves. No se descarta una combinación por ser
        # "rara", porque las combinaciones raras siguen siendo posibles.
        score = 0.0

        # Distribuciones observadas históricamente.
        parity_dist = self.stats.parity_counts(500)
        low_dist = self.stats.low_high_counts(500)
        sum_values = self.stats.sums(500)

        score += math.log1p(parity_dist.get(even, 0))
        score += math.log1p(low_dist.get(low, 0))

        if len(sum_values):
            zsum = abs((total - np.mean(sum_values)) /
                       (np.std(sum_values) + 1e-9))
            score -= min(zsum, 5.0) * 0.12

        if consecutive > 2:
            score -= 0.25 * (consecutive - 2)

        if spread < 15:
            score -= 0.20
        if spread > 48:
            score -= 0.10

        # Evita patrones extremadamente evidentes, pero no elimina
        # secuencias automáticamente.
        if len({n % 2 for n in draw}) == 1:
            score -= 0.15

        # Coocurrencia: señal débil.
        for a, b in combinations(draw, 2):
            score += math.log1p(self.pair_freq.get((a, b), 0)) * 0.015

        return score

    def popularity_penalty(self, draw):
        """
        Evita combinaciones muy obvias para reducir potencialmente el
        riesgo de compartir premio. Esto NO aumenta la probabilidad de
        que salgan los números.
        """
        score = 0.0
        draw = tuple(sorted(draw))

        # Muchas fechas de cumpleaños: exceso de números <=31.
        n_le_31 = sum(n <= 31 for n in draw)
        if n_le_31 >= 5:
            score -= 0.12
        if n_le_31 == 6:
            score -= 0.10

        # Secuencia completa o patrón visual.
        if all(draw[i] + 1 == draw[i + 1] for i in range(5)):
            score -= 0.30

        # Todos terminan igual/modulo sencillo.
        if len({n % 5 for n in draw}) <= 2:
            score -= 0.08

        return score

    def combo_score(self, draw):
        s = 0.0
        for n in draw:
            s += math.log(self.number_probs[n - 1] + 1e-12)

        s += self.structural_score(draw)
        s += self.popularity_penalty(draw)

        # Si ya apareció, se aplica una penalización muy pequeña.
        # No se afirma que repetir sea imposible.
        if draw in self.historical_combos:
            s -= 0.04

        return float(s)

    def generate(self, n=10, iterations=120000, seed=42):
        rng = np.random.default_rng(seed)
        results = {}

        weights = self.number_probs.copy()
        weights = weights / weights.sum()

        for _ in range(iterations):
            draw = tuple(sorted(
                rng.choice(
                    np.arange(1, MAX_NUM + 1),
                    size=6,
                    replace=False,
                    p=weights,
                )
            ))
            score = self.combo_score(draw)
            old = results.get(draw)
            if old is None or score > old:
                results[draw] = score

        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)

        # Diversificación: no devolver diez combinaciones casi idénticas.
        selected = []
        for draw, score in ranked:
            if all(
                len(set(draw) & set(other)) <= 5
                for other in selected
            ):
                selected.append(draw)
            if len(selected) >= n:
                break

        return selected


# ================================================================
# MONTE CARLO RÁPIDO
# ================================================================

class UltraMonteCarlo:
    def __init__(self, n_iterations=100000, seed=12345):
        self.n_iterations = int(n_iterations)
        self.seed = seed

    def sample_with_constraints(self, probs):
        rng = np.random.default_rng(self.seed)
        probs = np.asarray(probs, dtype=float)
        probs = np.clip(probs, 1e-12, None)
        probs /= probs.sum()

        freq = np.zeros(MAX_NUM, dtype=float)
        valid = 0

        # En lugar de generar listas Python innecesarias, se usa un bucle
        # compacto porque cada muestra es sin reemplazo.
        for _ in range(self.n_iterations):
            draw = tuple(sorted(
                rng.choice(
                    np.arange(1, MAX_NUM + 1),
                    size=6,
                    replace=False,
                    p=probs,
                )
            ))

            if self._check_constraints(draw):
                valid += 1
                for n in draw:
                    freq[n - 1] += 1

        if valid == 0:
            return probs

        return freq / valid

    @staticmethod
    def _check_constraints(draw):
        even = even_count(draw)
        low = low_count(draw)
        total = draw_sum(draw)
        consecutive = sum(draw[i + 1] == draw[i] + 1 for i in range(5))
        spread = draw_range(draw)

        # Rangos amplios, no reglas rígidas.
        if not 1 <= even <= 5:
            return False
        if not 1 <= low <= 5:
            return False
        if not 45 <= total <= 255:
            return False
        if consecutive > 3:
            return False
        if spread < 10:
            return False

        return True


# ================================================================
# BACKTESTING WALK-FORWARD
# ================================================================

class Backtester:
    """
    Es la pieza más importante del sistema.

    Para cada punto del histórico:
      - usa únicamente sorteos anteriores;
      - construye probabilidades;
      - selecciona top-N números;
      - mide cuántos de los seis futuros aparecen.

    Así evitamos evaluar una estrategia mirando el futuro.
    """

    def __init__(self, data):
        self.data = data

    def run(
        self,
        start=500,
        windows=(10, 25, 50, 100, 250),
        step=1,
        top_k=12,
    ):
        if len(self.data) <= start + 1:
            return {}

        hits = []
        six_hit = 0
        five_plus = 0
        four_plus = 0

        for i in range(start, len(self.data) - 1, step):
            train = self.data[:i]
            actual = set(self.data[i][1])

            engine = StatisticalEngine(train)
            p, _ = engine.probability_vector()
            top = set(np.argsort(p)[-top_k:] + 1)

            h = len(actual & top)
            hits.append(h)

            if h == 6:
                six_hit += 1
            if h >= 5:
                five_plus += 1
            if h >= 4:
                four_plus += 1

        if not hits:
            return {}

        arr = np.asarray(hits, dtype=float)
        result = {
            "tests": int(len(arr)),
            "mean_hits_top12": float(arr.mean()),
            "median_hits_top12": float(np.median(arr)),
            "p_ge_3": float(np.mean(arr >= 3)),
            "p_ge_4": float(np.mean(arr >= 4)),
            "p_ge_5": float(np.mean(arr >= 5)),
            "p_6": float(np.mean(arr == 6)),
            "six_hit_count": int(six_hit),
            "five_plus_count": int(five_plus),
            "four_plus_count": int(four_plus),
        }

        # Baseline aleatorio: hipergeométrica exacta para 12 de 49.
        # Se calcula la esperanza de aciertos.
        expected_random = NUMBERS_PER_DRAW * top_k / MAX_NUM
        result["random_expected_hits_top12"] = float(expected_random)
        result["lift_over_random"] = float(
            result["mean_hits_top12"] / expected_random
        )

        with open(BACKTEST_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return result


# ================================================================
# PREDICTOR PRINCIPAL
# ================================================================

class WeeklyStrategyEngine:
    """Motor específico para una única combinación mantenida durante siete días.

    No intenta conocer sorteos futuros. Construye un pronóstico robusto usando:
      1) ventanas estadísticas de varias longitudes;
      2) analogías históricas del estado actual;
      3) estabilidad de cada número entre horizontes;
      4) coocurrencias históricas de pares y triples;
      5) simulación Monte Carlo y puntuación estructural;
      6) backtesting walk-forward de la estrategia semanal.

    Todo se calcula exclusivamente con información disponible hasta el momento
    de la ejecución. La salida es una selección robusta, no una garantía.
    """
    WINDOWS = (10, 25, 50, 100, 250, 500, 1000)

    def __init__(self, data, progress_callback=None, model_probabilities=None):
        self.data = sorted(data, key=lambda x: x[0])
        self.progress_callback = progress_callback
        self.model_probabilities = (
            np.asarray(model_probabilities, dtype=float)
            if model_probabilities is not None else None
        )
        self._pair_cache = None
        self._combo_engine = None

    def _progress(self, pct, msg):
        if self.progress_callback:
            try:
                self.progress_callback(float(pct), msg)
            except Exception:
                pass

    @staticmethod
    def _freq_vector(rows):
        f = np.zeros(MAX_NUM, dtype=float)
        for _, draw in rows:
            for n in draw:
                f[n - 1] += 1.0
        s = f.sum()
        return f / s if s else np.full(MAX_NUM, 1.0 / MAX_NUM)

    @staticmethod
    def _signature(data, end_index):
        draws = data[:end_index]
        parts = []
        for w in (10, 25, 50, 100, 250):
            rows = draws[-w:]
            parts.append(WeeklyStrategyEngine._freq_vector(rows))
        # Señales compactas de estructura reciente.
        rows = draws[-50:]
        if rows:
            sums = np.asarray([draw_sum(d) for _, d in rows], dtype=float)
            ranges = np.asarray([draw_range(d) for _, d in rows], dtype=float)
            parity = np.asarray([even_count(d) for _, d in rows], dtype=float)
            low = np.asarray([low_count(d) for _, d in rows], dtype=float)
            extra = np.asarray([
                sums.mean(), sums.std(), ranges.mean(), ranges.std(),
                parity.mean(), low.mean()
            ], dtype=float)
        else:
            extra = np.zeros(6, dtype=float)
        return np.concatenate(parts + [extra])

    def _current_components(self):
        data = self.data
        components = []
        # Varias escalas, con mayor peso en lo reciente pero sin ignorar
        # el comportamiento de largo plazo.
        weights = {10: 0.16, 25: 0.19, 50: 0.18, 100: 0.15,
                   250: 0.12, 500: 0.10, 1000: 0.10}
        for w, weight in weights.items():
            rows = data[-min(w, len(data)):]
            p = self._freq_vector(rows)
            # Mezcla con la distribución global para reducir sobreajuste.
            global_p = self._freq_vector(data)
            p = 0.88 * p + 0.12 * global_p
            components.append((weight, p))
        p = sum(w * x for w, x in components)
        p /= p.sum() + 1e-12
        return p, components

    def _analog_probability(self, max_anchors=420):
        """Busca estados históricos parecidos y observa sus siete sorteos posteriores."""
        n = len(self.data)
        if n < 800:
            return np.full(MAX_NUM, 1.0 / MAX_NUM), 0.0, 0

        current = self._signature(self.data, n)
        # Anchors separados de la actualidad para que siempre dispongan de 7 sorteos futuros.
        lo = 500
        hi = n - 8
        if hi <= lo:
            return np.full(MAX_NUM, 1.0 / MAX_NUM), 0.0, 0
        candidates = list(range(lo, hi, max(1, (hi - lo) // max_anchors)))
        if len(candidates) > max_anchors:
            candidates = candidates[-max_anchors:]

        scored = []
        scale = np.std(current) + 1e-9
        for idx, anchor in enumerate(candidates):
            sig = self._signature(self.data, anchor)
            dist = float(np.mean(((sig - current) / scale) ** 2))
            scored.append((dist, anchor))
        scored.sort(key=lambda x: x[0])
        chosen = scored[:min(55, len(scored))]

        out = np.zeros(MAX_NUM, dtype=float)
        total_w = 0.0
        for rank, (dist, anchor) in enumerate(chosen):
            # Combina similitud y rango del vecino.
            w = math.exp(-min(dist, 25.0) / 4.0) / (1.0 + 0.015 * rank)
            future = self.data[anchor:anchor + 7]
            for _, draw in future:
                for nmb in draw:
                    out[nmb - 1] += w
            total_w += 7.0 * w
        if total_w <= 0:
            return np.full(MAX_NUM, 1.0 / MAX_NUM), 0.0, 0
        out /= total_w
        best_dist = float(chosen[0][0]) if chosen else 0.0
        reliability = math.exp(-min(best_dist, 20.0) / 5.0)
        return out, reliability, len(chosen)

    def _pair_triple_scores(self):
        rows = self.data[-1000:]
        pair = Counter()
        triple = Counter()
        for _, draw in rows:
            for p in combinations(draw, 2):
                pair[p] += 1
            for tri in combinations(draw, 3):
                triple[tri] += 1
        pair_score = np.zeros(MAX_NUM, dtype=float)
        triple_score = np.zeros(MAX_NUM, dtype=float)
        for (a, b), c in pair.items():
            pair_score[a - 1] += c
            pair_score[b - 1] += c
        for tri, c in triple.items():
            for n in tri:
                triple_score[n - 1] += c
        pair_score = pair_score / (pair_score.sum() + 1e-12)
        triple_score = triple_score / (triple_score.sum() + 1e-12)
        return pair_score, triple_score, pair, triple

    def _robust_number_vector(self):
        base, components = self._current_components()
        self._progress(8, "Semanal: calculando siete escalas temporales…")
        analog, analog_rel, analog_count = self._analog_probability()
        self._progress(30, f"Semanal: analogías históricas evaluadas ({analog_count} estados)…")

        # Estabilidad: número que permanece fuerte al cambiar la ventana.
        matrix = np.vstack([p for _, p in components])
        mean_p = np.mean(matrix, axis=0)
        variability = np.std(matrix, axis=0)
        stability = mean_p / (1.0 + 10.0 * variability)
        stability = stability / (stability.sum() + 1e-12)

        pair_score, triple_score, pair, triple = self._pair_triple_scores()
        self._progress(42, "Semanal: coocurrencias de pares y triples…")

        # Si hay analogía fiable, le damos peso relevante; si no, no inventamos
        # confianza y el motor vuelve hacia las señales estadísticas.
        wa = 0.24 * analog_rel
        wb = 0.28
        ws = 0.22
        wp = 0.12
        wt = 0.06
        wg = max(0.0, 1.0 - (wa + wb + ws + wp + wt))
        global_p = self._freq_vector(self.data)

        robust = (wg * global_p + wb * base + wa * analog +
                  ws * stability + wp * pair_score + wt * triple_score)
        robust = np.clip(robust, 1e-12, None)
        robust /= robust.sum()

        # Si ya existe un ensemble entrenado en Fuxing, se incorpora como
        # una señal adicional de contraste, sin permitir que domine al
        # análisis semanal específico.
        if self.model_probabilities is not None and len(self.model_probabilities) == MAX_NUM:
            mp = np.clip(self.model_probabilities, 1e-12, None)
            mp /= mp.sum()
            robust = 0.85 * robust + 0.15 * mp
            robust /= robust.sum()
            self._progress(50, "Semanal: contrastando la señal semanal con el ensemble Fuxing…")

        return robust, {
            "base": base, "analog": analog, "stability": stability,
            "pair": pair_score, "triple": triple_score,
            "analog_reliability": analog_rel, "analog_count": analog_count,
            "pair_count": len(pair), "triple_count": len(triple),
            "component_std": variability,
        }

    def _score_combo(self, draw, probs, details):
        """Puntuación rápida para explorar muchas combinaciones sin recalcular todo el histórico."""
        if self._pair_cache is None:
            self._pair_cache = self._pair_triple_scores()
        _, _, pair, triple = self._pair_cache
        score = 0.0
        stab = details["stability"]
        analog = details["analog"]
        for n in draw:
            score += math.log(probs[n - 1] + 1e-12)
            score += 0.55 * math.log(stab[n - 1] + 1e-12)
            score += 0.35 * math.log(analog[n - 1] + 1e-12)
        for a, b in combinations(draw, 2):
            score += 0.012 * math.log1p(pair.get((a, b), 0))
        for tri in combinations(draw, 3):
            score += 0.006 * math.log1p(triple.get(tri, 0))
        ev = even_count(draw); low = low_count(draw); total = draw_sum(draw)
        cons = sum(draw[i + 1] == draw[i] + 1 for i in range(5)); spread = draw_range(draw)
        if ev in (0, 6): score -= 0.35
        if low in (0, 6): score -= 0.30
        if total < 75 or total > 225: score -= 0.10
        if cons > 2: score -= 0.18 * (cons - 2)
        if spread < 15: score -= 0.18
        if sum(n <= 31 for n in draw) >= 5: score -= 0.08
        return float(score)

    def _monte_carlo_candidates(self, probs, details, iterations=140000):
        self._progress(55, f"Semanal: simulación Monte Carlo robusta ({iterations:,} iteraciones)…")
        rng = np.random.default_rng(ULTIMATE_SEED + 701)
        probs = np.asarray(probs, dtype=float)
        probs = np.clip(probs, 1e-12, None)
        probs /= probs.sum()
        top = np.argsort(probs)[::-1]
        # Se reserva el universo principal a los 18 más fuertes, pero se mantiene
        # exploración en los 49 para no convertir el filtro en una regla rígida.
        explore = np.arange(1, MAX_NUM + 1)
        results = {}
        keep = max(30000, min(int(iterations), 160000))
        for i in range(keep):
            draw = tuple(sorted(rng.choice(explore, 6, replace=False, p=probs)))
            if not UltraMonteCarlo._check_constraints(draw):
                continue
            score = self._score_combo(draw, probs, details)
            old = results.get(draw)
            if old is None or score > old:
                results[draw] = score
        if not results:
            return tuple(sorted((top[:6] + 1).tolist()))
        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return ranked[0][0]

    def _deterministic_best(self, probs, details):
        """Búsqueda exhaustiva sobre los 12 números principales: barata y estable."""
        top = (np.argsort(probs)[::-1][:12] + 1).tolist()
        best = None
        best_score = -float("inf")
        for draw in combinations(top, 6):
            if not UltraMonteCarlo._check_constraints(draw):
                continue
            score = self._score_combo(draw, probs, details)
            if score > best_score:
                best_score, best = score, draw
        return tuple(best) if best else tuple(sorted(top[:6]))

    def backtest(self, max_weeks=100):
        """Backtesting semanal walk-forward: lunes hipotético, 7 sorteos futuros."""
        n = len(self.data)
        if n < 900:
            return {"tests": 0, "reason": "histórico insuficiente"}
        # Anclamos en fechas reales y usamos puntos separados por 7 sorteos.
        end = n - 8
        start = max(800, end - max_weeks * 7)
        anchors = list(range(start, end, 7))[-max_weeks:]
        hits = []
        best_hits = []
        exact = 0
        if not anchors:
            return {"tests": 0}
        self._progress(68, f"Semanal: backtesting walk-forward ({len(anchors)} semanas)…")
        for pos, anchor in enumerate(anchors):
            train = self.data[:anchor]
            engine = WeeklyStrategyEngine(train)
            p, det = engine._robust_number_vector()
            combo = engine._deterministic_best(p, det)
            future = self.data[anchor:anchor + 7]
            week_hits = [len(set(combo) & set(draw)) for _, draw in future]
            hits.extend(week_hits)
            best_hits.append(max(week_hits))
            if max(week_hits) == 6:
                exact += 1
            if pos % max(1, len(anchors)//10) == 0:
                self._progress(68 + 15 * (pos / max(1, len(anchors))),
                               f"Semanal: backtesting {pos+1}/{len(anchors)} semanas…")

        arr = np.asarray(hits, dtype=float)
        bh = np.asarray(best_hits, dtype=float)
        expected_single = NUMBERS_PER_DRAW * 6 / MAX_NUM
        result = {
            "tests": int(len(anchors)),
            "draws_evaluated": int(len(arr)),
            "mean_hits_per_draw": float(arr.mean()) if len(arr) else 0.0,
            "median_hits_per_draw": float(np.median(arr)) if len(arr) else 0.0,
            "p_ge_2": float(np.mean(arr >= 2)) if len(arr) else 0.0,
            "p_ge_3": float(np.mean(arr >= 3)) if len(arr) else 0.0,
            "p_ge_4": float(np.mean(arr >= 4)) if len(arr) else 0.0,
            "mean_best_hits_week": float(bh.mean()) if len(bh) else 0.0,
            "p_best_ge_3": float(np.mean(bh >= 3)) if len(bh) else 0.0,
            "p_best_ge_4": float(np.mean(bh >= 4)) if len(bh) else 0.0,
            "six_hit_weeks": int(exact),
            "random_expected_hits_per_draw": float(expected_single),
        }
        result["lift_over_random"] = (result["mean_hits_per_draw"] / expected_single
                                       if expected_single else 0.0)
        return result

    def predict(self, mc_iterations=140000, run_backtest=True):
        if len(self.data) < 1000:
            raise ValueError("Se necesitan al menos 1000 sorteos para la modalidad semanal robusta.")
        self._progress(3, "Semanal: iniciando análisis multiescala…")
        probs, details = self._robust_number_vector()
        self._progress(52, "Semanal: vector robusto consolidado…")
        combo = self._monte_carlo_candidates(probs, details, mc_iterations)
        # Segundo criterio determinista para evitar que una única semilla domine.
        deterministic = self._deterministic_best(probs, details)
        score_mc = self._score_combo(combo, probs, details)
        score_det = self._score_combo(deterministic, probs, details)
        if score_det > score_mc:
            combo = deterministic
        self._progress(88, "Semanal: validando estabilidad de la combinación final…")

        scores = np.asarray([probs[n - 1] for n in combo], dtype=float)
        concentration = float(np.mean(scores) / (np.mean(np.sort(probs)[-6:]) + 1e-12))
        stability_values = [details["stability"][n - 1] for n in combo]
        stability_score = float(np.mean(stability_values) / (np.max(details["stability"]) + 1e-12))
        backtest = self.backtest(max_weeks=60) if run_backtest else {"tests": 0}
        self._progress(98, "Semanal: análisis y backtesting finalizados.")

        result = {
            "combination": tuple(sorted(int(x) for x in combo)),
            "probabilities": probs,
            "details": details,
            "stability_score": max(0.0, min(1.0, stability_score)),
            "concentration_score": max(0.0, min(1.0, concentration)),
            "backtest": backtest,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "period_days": 7,
        }
        return result




# ================================================================
# INTERFAZ GRÁFICA
# ================================================================


# ---------------------------------------------------------------------
# Predictor definitivo: reemplaza el predictor anterior manteniendo la UI.
# ---------------------------------------------------------------------

class UltraPredictor:
    def __init__(self):
        self.data = None
        self.X_seq = None
        self.X_feat = None
        self.Y = None
        self.stat_engine = None
        self.ml_model = UltimateSklearnEnsemble() if ULTIMATE_SKLEARN else None
        self.neural = UltimateNeuralTrainer()
        self.torch_model = None
        self.last_result = None
        self.training_report = {}
        self.weights = np.array([0.55, 0.15, 0.20, 0.10], dtype=float)
        self._trained_signature = None

    def prepare_data(self, data):
        if not data:
            raise ValueError("Dataset vacío")

        self.data = sorted(data, key=lambda x: x[0])
        dataset_audit(self.data)
        randomness_audit(self.data)

        self.X_seq, self.X_feat, self.Y = build_dataset(self.data)
        self.stat_engine = StatisticalEngine(self.data)

        log(
            f"Preparación completa: {len(self.data):,} sorteos; "
            f"{len(self.X_seq):,} muestras supervisadas; "
            f"{self.X_feat.shape[1]} features."
        )
        return self.X_seq, self.X_feat, self.Y

    def optimize_hyperparameters(self):
        if self.data is None:
            raise ValueError("Primero prepara los datos")

        # Validación temporal reservada para seleccionar los pesos.
        n = len(self.X_feat)
        cut = int(n * 0.82)
        val_end = int(n * 0.94)
        if cut < 300 or val_end <= cut:
            return {"status": "insufficient_validation_data"}

        # Modelo estadístico para cada punto de validación.
        # No se utiliza aquí para entrenar la red; solo para medir el
        # componente estadístico fuera de muestra.
        stat_rows = []
        for j in range(cut, val_end):
            # j de build_dataset corresponde aproximadamente al instante
            # del último sorteo conocido; se reconstruye el train correcto.
            data_index = SEQ_LEN - 1 + j
            if data_index >= len(self.data) - 1:
                break
            p, _ = StatisticalEngine(self.data[:data_index + 1]).probability_vector()
            stat_rows.append(p)

        if not stat_rows:
            return {"status": "no_validation_rows"}

        y_val = self.Y[cut:cut + len(stat_rows)]
        stat_val = np.asarray(stat_rows, dtype=float)

        # Si los modelos ya están entrenados, se incluyen.
        comps = [stat_val, None, None, None]
        if self.ml_model is not None and self.ml_model.fitted:
            a, b = self.ml_model.predict_components(self.X_feat[cut:cut + len(stat_rows)])
            comps[1], comps[2] = a, b

        if self.neural.fitted:
            npred = self.neural.predict(
                self.X_seq[cut:cut + len(stat_rows)],
                self.X_feat[cut:cut + len(stat_rows)],
            )
            comps[3] = npred

        optimizer = UltimateWeightOptimizer()
        self.weights = optimizer.optimize(y_val, comps)
        self.training_report["weight_optimization"] = optimizer.report

        log(
            "Pesos del ensemble optimizados por validación temporal: "
            + ", ".join(f"{x:.2f}" for x in self.weights)
        )
        return optimizer.report

    def train_ensemble(self, n_models=5):
        if self.data is None:
            raise ValueError("Primero prepara los datos")

        log("=== ENTRENAMIENTO REAL DEL ENSEMBLE ===")
        started = time.time()

        n = len(self.X_feat)
        split = int(n * 0.84)
        val_end = int(n * 0.92)

        Xtr, Xv = self.X_feat[:split], self.X_feat[split:val_end]
        Ytr, Yv = self.Y[:split], self.Y[split:val_end]

        report = {
            "timestamp": datetime.now().isoformat(),
            "samples": int(n),
            "features": int(self.X_feat.shape[1]),
            "sklearn": bool(ULTIMATE_SKLEARN),
            "pytorch": bool(TORCH_AVAILABLE),
        }

        # ---------------- Scikit-learn ----------------
        if self.ml_model is not None:
            log("Entrenando Scikit-learn: Logistic Regression + HistGradientBoosting...")
            sk_report = self.ml_model.fit(Xtr, Ytr, Xv, Yv)
            report["sklearn_report"] = sk_report
            if sk_report.get("val_brier") is not None:
                log(
                    f"Scikit-learn validación: Brier={sk_report['val_brier']:.6f}, "
                    f"hits@12={sk_report['val_hits_top12']:.4f}"
                )
        else:
            log("Scikit-learn no está instalado.", "WARNING")

        # ---------------- Red neuronal ----------------
        if TORCH_AVAILABLE:
            log("Entrenando red neuronal Transformer + MLP...")
            nn_report = self.neural.fit(self.X_seq, self.X_feat, self.Y)
            report["neural_report"] = nn_report
            self.torch_model = self.neural.model
        else:
            log("PyTorch no está instalado.", "WARNING")

        # Optimización posterior de pesos.
        try:
            self.optimize_hyperparameters()
        except Exception as exc:
            log(f"No se pudieron optimizar pesos: {exc}", "WARNING")

        # Persistencia.
        self._trained_signature = hashlib.sha256(
            f"{self.data[0][0]}|{self.data[-1][0]}|{len(self.data)}".encode()
        ).hexdigest()

        if TORCH_AVAILABLE and self.neural.fitted:
            try:
                torch.save(
                    {
                        "state_dict": self.neural.model.state_dict(),
                        "feature_size": int(self.X_feat.shape[1]),
                        "scaler": self.neural.feature_scaler,
                        "signature": self._trained_signature,
                    },
                    ULTIMATE_TORCH_FILE,
                )
            except Exception as exc:
                log(f"No se pudo guardar la red neuronal: {exc}", "WARNING")

        report["weights"] = self.weights.tolist()
        report["elapsed_seconds"] = time.time() - started
        self.training_report = report
        _safe_json_write(ULTIMATE_STATE_FILE, report)

        log(
            f"=== ENTRENAMIENTO TERMINADO en "
            f"{report['elapsed_seconds']:.1f}s ==="
        )
        return self

    def _stat_current(self):
        return self.stat_engine.probability_vector()[0]

    def _current_components(self):
        seq, feat = UltimateFeatureBuilder.future_sample(self.data)

        stat = self._stat_current()
        stat = _normalize_mass(stat)[None, :]

        logistic = gradient = neural = None

        if self.ml_model is not None and self.ml_model.fitted:
            logistic, gradient = self.ml_model.predict_components(feat)

        if self.neural.fitted:
            neural = self.neural.predict(seq, feat)

        return (
            stat,
            logistic,
            gradient,
            neural,
        )

    def predict_ultra(self, n_mc_iterations=150000, n_combinations=10):
        if self.data is None:
            raise ValueError("Primero prepara los datos")

        # Si todavía no se ha entrenado, entrenamos automáticamente.
        # Así el botón de predicción nunca utiliza una "IA de mentira".
        if not self._is_trained():
            log("No existe entrenamiento válido para este histórico. Entrenando automáticamente...")
            self.train_ensemble()

        components = self._current_components()

        rows = []
        for p in components:
            if p is not None:
                rows.append(_normalize_rows(p)[0])

        if not rows:
            raise RuntimeError("No se pudo obtener ningún componente predictivo.")

        # Si los pesos no corresponden a componentes disponibles, redistribuir
        # proporcionalmente sin inventar información.
        base_weights = self.weights.copy()
        available = np.array([p is not None for p in components], dtype=bool)
        base_weights[~available] = 0.0
        if base_weights.sum() <= 0:
            base_weights[available] = 1.0
        base_weights /= base_weights.sum()

        combined = np.zeros(MAX_NUM, dtype=float)
        for w, p in zip(base_weights, components):
            if p is not None:
                combined += w * _normalize_rows(p)[0]
        combined = _normalize_mass(combined)

        log(
            "Pesos activos en predicción final: "
            + ", ".join(f"{x:.3f}" for x in base_weights)
        )

        # Monte Carlo como muestreo de combinaciones, no como prueba de
        # que una combinación "tenga que salir".
        mc = UltraMonteCarlo(
            n_iterations=min(max(20000, int(n_mc_iterations)), 500000),
            seed=ULTIMATE_SEED,
        )
        mc_p6 = mc.sample_with_constraints(combined)
        mc_p6 = _normalize_mass(mc_p6)

        final_p = _normalize_mass(0.85 * combined + 0.15 * mc_p6)

        generator = CombinationEngine(self.data, final_p)
        combos = generator.generate(
            n=max(1, int(n_combinations)),
            iterations=min(max(50000, int(n_mc_iterations)), 400000),
            seed=ULTIMATE_SEED + 7,
        )

        matrix = np.vstack([
            _normalize_rows(p)[0] for p in components if p is not None
        ])
        uncertainty = np.std(matrix, axis=0)
        agreement = 1.0 - float(np.mean(uncertainty))
        agreement = float(np.clip(agreement, 0.0, 1.0))

        result = {
            "combinations": combos,
            "probabilities": final_p,
            "statistical_probability": components[0][0],
            "logistic_probability": (
                components[1][0] if components[1] is not None else None
            ),
            "gradient_probability": (
                components[2][0] if components[2] is not None else None
            ),
            "neural_probability": (
                components[3][0] if components[3] is not None else None
            ),
            "mc_probability": mc_p6,
            "uncertainty": uncertainty,
            "confidence": agreement,
            "weights": base_weights,
            "top_numbers": np.argsort(final_p)[::-1] + 1,
            "training_report": self.training_report,
            "model_agreement": agreement,
            "warning": (
                "La consistencia del ensemble no es la probabilidad "
                "matemática de acertar 6/6."
            ),
        }

        self.last_result = result
        return result

    def _is_trained(self):
        return bool(
            (self.ml_model is not None and self.ml_model.fitted)
            or self.neural.fitted
        )


# =====================================================================
# BONOLOTO AI RESEARCH LAB - MAXIMUM COMPUTE EXTENSION
# =====================================================================
# Capa adicional: NO elimina las capas anteriores.
#
# Añade, cuando las librerías están disponibles:
#   - ExtraTrees
#   - RandomForest
#   - XGBoost (opcional)
#   - ensemble de varias redes neuronales con semillas distintas
#   - calibración/mezcla robusta de componentes
#   - más capacidad de CPU/GPU sin cambiar la interfaz original
#
# IMPORTANTE:
# Más capacidad computacional no cambia la probabilidad matemática de una
# combinación en un sorteo verdaderamente aleatorio. Sirve para investigar
# mejor, detectar señales espurias y evitar depender de un único modelo.
# =====================================================================

# ---- librerías ML opcionales adicionales ----------------------------
MAX_EXTRA_SKLEARN = False
MAX_XGBOOST_AVAILABLE = False

try:
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    MAX_EXTRA_SKLEARN = True
except Exception:
    ExtraTreesClassifier = None
    RandomForestClassifier = None

try:
    from xgboost import XGBClassifier
    MAX_XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None

# CPU: dejamos que NumPy/BLAS y scikit-learn aprovechen los núcleos.
# No se fuerza un número fijo porque el equipo del usuario puede variar.
MAX_CPU_WORKERS = max(1, (os.cpu_count() or 2) - 1)

# Parámetros de capacidad. Se pueden reducir si el equipo tarda demasiado.
MAX_TREE_ESTIMATORS = 500
MAX_RF_ESTIMATORS = 400
MAX_XGB_ESTIMATORS = 350
MAX_NEURAL_ENSEMBLE = 3

def _brier_multilabel(y_true, p):
    y = np.asarray(y_true, dtype=float)
    q = np.asarray(p, dtype=float)
    return float(np.mean((y - q) ** 2))

def _topk_hits(y_true, p, k=12):
    y = np.asarray(y_true)
    p = np.asarray(p)
    total = 0.0
    for i in range(len(y)):
        pred = set(np.argsort(p[i])[-k:])
        actual = set(np.where(y[i] > 0.5)[0])
        total += len(pred & actual)
    return total / max(1, len(y))

class UltimateSklearnEnsemble:
    """
    Entrenamiento real con Scikit-learn.
    Se entrenan dos familias complementarias por número:
      1) LogisticRegression: relación estable y regularizada.
      2) HistGradientBoosting: relaciones no lineales.
    """

    def __init__(self):
        self.scaler = None
        self.logistic = [None] * MAX_NUM
        self.gradient = [None] * MAX_NUM
        self.fitted = False
        self.feature_count = None
        self.training_rows = 0

    def fit(self, X_train, Y_train, X_val=None, Y_val=None):
        if not ULTIMATE_SKLEARN:
            return {"available": False}

        X_train = np.asarray(X_train, dtype=np.float32)
        Y_train = np.asarray(Y_train, dtype=np.float32)
        self.feature_count = X_train.shape[1]
        self.training_rows = len(X_train)

        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X_train)

        for n in range(MAX_NUM):
            y = Y_train[:, n].astype(int)
            if len(np.unique(y)) < 2:
                continue

            # Logistic: robusto, interpretable y muy regularizado.
            lr = LogisticRegression(
                C=0.08,
                max_iter=1800,
                class_weight="balanced",
                solver="lbfgs",
                random_state=ULTIMATE_SEED,
            )
            try:
                lr.fit(Xs, y)
                self.logistic[n] = lr
            except Exception as exc:
                log(f"Logistic n={n+1:02d}: {exc}", "WARNING")

            # Gradient Boosting: no lineal, con regularización y early stopping.
            try:
                gb = HistGradientBoostingClassifier(
                    learning_rate=0.045,
                    max_iter=220,
                    max_leaf_nodes=15,
                    max_depth=5,
                    min_samples_leaf=35,
                    l2_regularization=1.5,
                    early_stopping=True,
                    validation_fraction=0.12,
                    n_iter_no_change=20,
                    random_state=ULTIMATE_SEED + n,
                )
                gb.fit(Xs, y)
                self.gradient[n] = gb
            except Exception as exc:
                log(f"Gradient n={n+1:02d}: {exc}", "WARNING")

        self.fitted = True

        metrics = {"available": True, "rows": len(X_train)}
        if X_val is not None and Y_val is not None and len(X_val):
            pred = self.predict(X_val)
            metrics["val_brier"] = _brier_multilabel(Y_val, pred)
            metrics["val_hits_top12"] = _topk_hits(Y_val, pred, 12)
            metrics["val_hits_top18"] = _topk_hits(Y_val, pred, 18)
        return metrics

    def predict_components(self, X):
        if not self.fitted or self.scaler is None:
            return None, None

        Xs = self.scaler.transform(np.asarray(X, dtype=np.float32))
        log_p = np.full((len(Xs), MAX_NUM), 1.0 / MAX_NUM, dtype=float)
        gb_p = np.full((len(Xs), MAX_NUM), 1.0 / MAX_NUM, dtype=float)

        for n in range(MAX_NUM):
            if self.logistic[n] is not None:
                try:
                    log_p[:, n] = self.logistic[n].predict_proba(Xs)[:, 1]
                except Exception:
                    pass
            if self.gradient[n] is not None:
                try:
                    gb_p[:, n] = self.gradient[n].predict_proba(Xs)[:, 1]
                except Exception:
                    pass

        for arr in (log_p, gb_p):
            arr[:] = np.clip(arr, 1e-9, None)
            arr[:] = arr / (arr.sum(axis=1, keepdims=True) + 1e-12)

        return log_p, gb_p

    def predict(self, X):
        a, b = self.predict_components(X)
        if a is None:
            return None
        return _normalize_rows(0.45 * a + 0.55 * b)

def _normalize_mass(p):
    """Normaliza un vector de masa/probabilidades a suma 1, evitando NaN/inf."""
    p = np.asarray(p, dtype=np.float64)
    p = np.nan_to_num(p, nan=1.0 / MAX_NUM, posinf=1.0, neginf=1.0)
    p = np.clip(p, 1e-12, None)
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(MAX_NUM, 1.0 / MAX_NUM, dtype=np.float64)
    return p / total

def _normalize_rows(a):
    a = np.asarray(a, dtype=float)
    a = np.clip(np.nan_to_num(a, nan=1e-12), 1e-12, None)
    return a / (a.sum(axis=1, keepdims=True) + 1e-12)

# ---------------------------------------------------------------------
# Ensemble Scikit-learn ampliado
# ---------------------------------------------------------------------
class MaximumSklearnEnsemble:
    """
    Extiende el ensemble existente sin quitar Logistic Regression ni
    HistGradientBoosting.

    Componentes internos:
      - Logistic Regression
      - HistGradientBoosting
      - ExtraTrees
      - RandomForest
      - XGBoost, si está instalado

    Para no multiplicar innecesariamente la dimensionalidad del ensemble
    externo, predict_components devuelve:
      [logistic, blend_tree]
    donde blend_tree es una mezcla calibrada de los modelos no lineales.
    """

    def __init__(self):
        super().__init__()
        self.extra = [None] * MAX_NUM
        self.forest = [None] * MAX_NUM
        self.xgb = [None] * MAX_NUM
        self.tree_weights = np.array([0.30, 0.25, 0.25, 0.20], dtype=float)
        self.extended_report = {}
        self.fitted = False
        self.feature_count = None
        self.training_rows = 0

    def fit(self, X_train, Y_train, X_val=None, Y_val=None, progress_callback=None):
        if not ULTIMATE_SKLEARN:
            return {"available": False}

        X_train = np.asarray(X_train, dtype=np.float32)
        Y_train = np.asarray(Y_train, dtype=np.float32)
        self.feature_count = X_train.shape[1]
        self.training_rows = len(X_train)

        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X_train)

        # Reinicializar para permitir reentrenamiento limpio.
        self.logistic = [None] * MAX_NUM
        self.gradient = [None] * MAX_NUM
        self.extra = [None] * MAX_NUM
        self.forest = [None] * MAX_NUM
        self.xgb = [None] * MAX_NUM

        for n in range(MAX_NUM):
            if progress_callback:
                progress_callback(20.0 + (n / MAX_NUM) * 15.0, f"Scikit-learn: entrenando número {n+1}/{MAX_NUM}")
            y = Y_train[:, n].astype(np.int8)
            if len(np.unique(y)) < 2:
                continue

            # 1) Logistic: conserva el modelo original.
            try:
                lr = LogisticRegression(
                    C=0.08,
                    max_iter=2500,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=ULTIMATE_SEED,
                    n_jobs=None,
                )
                lr.fit(Xs, y)
                self.logistic[n] = lr
            except Exception as exc:
                log(f"Maximum Logistic n={n+1:02d}: {exc}", "WARNING")

            # 2) HistGradientBoosting: conserva el modelo original.
            try:
                gb = HistGradientBoostingClassifier(
                    learning_rate=0.035,
                    max_iter=400,
                    max_leaf_nodes=31,
                    max_depth=7,
                    min_samples_leaf=25,
                    l2_regularization=2.0,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=30,
                    random_state=ULTIMATE_SEED + n,
                )
                gb.fit(Xs, y)
                self.gradient[n] = gb
            except Exception as exc:
                log(f"Maximum Gradient n={n+1:02d}: {exc}", "WARNING")

            # 3) ExtraTrees: excelente detector de interacciones no lineales.
            if MAX_EXTRA_SKLEARN:
                try:
                    et = ExtraTreesClassifier(
                        n_estimators=MAX_TREE_ESTIMATORS,
                        max_features="sqrt",
                        min_samples_leaf=4,
                        max_depth=None,
                        class_weight="balanced_subsample",
                        bootstrap=False,
                        n_jobs=MAX_CPU_WORKERS,
                        random_state=ULTIMATE_SEED + 1000 + n,
                    )
                    et.fit(X_train, y)
                    self.extra[n] = et
                except Exception as exc:
                    log(f"ExtraTrees n={n+1:02d}: {exc}", "WARNING")

                # 4) RandomForest: familia distinta para diversificar errores.
                try:
                    rf = RandomForestClassifier(
                        n_estimators=MAX_RF_ESTIMATORS,
                        max_features="sqrt",
                        min_samples_leaf=4,
                        class_weight="balanced_subsample",
                        bootstrap=True,
                        n_jobs=MAX_CPU_WORKERS,
                        random_state=ULTIMATE_SEED + 2000 + n,
                    )
                    rf.fit(X_train, y)
                    self.forest[n] = rf
                except Exception as exc:
                    log(f"RandomForest n={n+1:02d}: {exc}", "WARNING")

            # 5) XGBoost: opcional. Si no está instalado se omite.
            if MAX_XGBOOST_AVAILABLE:
                try:
                    xgb = XGBClassifier(
                        n_estimators=MAX_XGB_ESTIMATORS,
                        max_depth=5,
                        learning_rate=0.035,
                        subsample=0.85,
                        colsample_bytree=0.80,
                        min_child_weight=8,
                        reg_alpha=0.05,
                        reg_lambda=2.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        n_jobs=MAX_CPU_WORKERS,
                        random_state=ULTIMATE_SEED + 3000 + n,
                        verbosity=0,
                    )
                    xgb.fit(X_train, y)
                    self.xgb[n] = xgb
                except Exception as exc:
                    log(f"XGBoost n={n+1:02d}: {exc}", "WARNING")

        self.fitted = True

        metrics = {
            "available": True,
            "rows": len(X_train),
            "extra_trees": any(x is not None for x in self.extra),
            "random_forest": any(x is not None for x in self.forest),
            "xgboost": any(x is not None for x in self.xgb),
        }

        if X_val is not None and Y_val is not None and len(X_val):
            try:
                pred = self.predict(X_val)
                metrics["val_brier"] = _brier_multilabel(Y_val, pred)
                metrics["val_hits_top12"] = _topk_hits(Y_val, pred, 12)
                metrics["val_hits_top18"] = _topk_hits(Y_val, pred, 18)
            except Exception as exc:
                log(f"Métricas validation ampliadas: {exc}", "WARNING")

        self.extended_report = metrics
        return metrics

    @staticmethod
    def _safe_predict(model, X):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            return None

    def predict_components(self, X):
        if not self.fitted or self.scaler is None:
            return None, None

        X = np.asarray(X, dtype=np.float32)
        Xs = self.scaler.transform(X)

        logistic = np.full((len(X), MAX_NUM), 1.0 / MAX_NUM, dtype=float)
        nonlinear_models = [
            np.full((len(X), MAX_NUM), 1.0 / MAX_NUM, dtype=float)
            for _ in range(4)
        ]

        for n in range(MAX_NUM):
            if self.logistic[n] is not None:
                pred = self._safe_predict(self.logistic[n], Xs)
                if pred is not None:
                    logistic[:, n] = pred

            model_sets = (
                (self.gradient[n], 0),
                (self.extra[n], 1),
                (self.forest[n], 2),
                (self.xgb[n], 3),
            )
            for model, idx in model_sets:
                if model is None:
                    continue
                # Gradient usa X escalada; árboles usan X original.
                pred = self._safe_predict(
                    model, Xs if idx == 0 else X
                )
                if pred is not None:
                    nonlinear_models[idx][:, n] = pred

        logistic = _normalize_rows(logistic)

        # Mezcla sólo de modelos realmente presentes.
        available = []
        if any(x is not None for x in self.gradient):
            available.append((self.tree_weights[0], nonlinear_models[0]))
        if any(x is not None for x in self.extra):
            available.append((self.tree_weights[1], nonlinear_models[1]))
        if any(x is not None for x in self.forest):
            available.append((self.tree_weights[2], nonlinear_models[2]))
        if any(x is not None for x in self.xgb):
            available.append((self.tree_weights[3], nonlinear_models[3]))

        if available:
            wsum = sum(w for w, _ in available)
            tree_mix = sum((w / wsum) * p for w, p in available)
            tree_mix = _normalize_rows(tree_mix)
        else:
            tree_mix = np.full_like(logistic, 1.0 / MAX_NUM)

        return logistic, tree_mix


# ---------------------------------------------------------------------
# Red neuronal: ensemble de varias semillas
# ---------------------------------------------------------------------
if TORCH_AVAILABLE:
    class UltimateSequenceNet(nn.Module):
        """
        Red neuronal temporal moderada para el tamaño real del problema.
        Entrada: ventana de sorteos + vector estadístico del instante.
        Salida: 49 probabilidades de pertenencia al siguiente sorteo.
        """
        def __init__(self, feature_size, d_model=128, heads=8, layers=3):
            super().__init__()
            self.proj = nn.Linear(MAX_NUM, d_model)
            self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, d_model))
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=d_model * 3,
                dropout=0.18,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.seq_norm = nn.LayerNorm(d_model)

            self.feat = nn.Sequential(
                nn.Linear(feature_size, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.18),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.18),
            )

            self.fusion = nn.Sequential(
                nn.Linear(d_model + 128, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.20),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, MAX_NUM),
            )

        def forward(self, seq, feat):
            x = self.proj(seq)
            x = x + self.pos[:, :x.shape[1]]
            x = self.encoder(x)
            x = self.seq_norm(x[:, -1])
            f = self.feat(feat)
            return self.fusion(torch.cat([x, f], dim=1))


class UltimateNeuralTrainer:
    def __init__(self):
        self.model = None
        self.feature_scaler = None
        self.fitted = False
        self.best_val_loss = None
        self.epochs_done = 0

    def fit(self, X_seq, X_feat, Y):
        if not TORCH_AVAILABLE:
            return {"available": False}

        X_seq = np.asarray(X_seq, dtype=np.float32)
        X_feat = np.asarray(X_feat, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        if len(X_seq) < 300:
            log("Pocos ejemplos para la red neuronal; se omite entrenamiento.", "WARNING")
            return {"available": True, "trained": False, "reason": "too_few_rows"}

        # División temporal: jamás aleatorizamos train/validación.
        cut = int(len(X_seq) * 0.84)
        val_cut = int(len(X_seq) * 0.92)
        cut = max(200, min(cut, len(X_seq) - 100))
        val_cut = max(cut + 50, min(val_cut, len(X_seq)))

        train_feat = X_feat[:cut]
        val_feat = X_feat[cut:val_cut]

        self.feature_scaler = StandardScaler()
        self.feature_scaler.fit(train_feat)
        X_feat_scaled = self.feature_scaler.transform(X_feat).astype(np.float32)

        device = torch.device(DEVICE)
        self.model = UltimateSequenceNet(X_feat.shape[1]).to(device)

        tx = torch.tensor(X_seq[:cut], dtype=torch.float32)
        tf = torch.tensor(X_feat_scaled[:cut], dtype=torch.float32)
        ty = torch.tensor(Y[:cut], dtype=torch.float32)

        vx = torch.tensor(X_seq[cut:val_cut], dtype=torch.float32, device=device)
        vf = torch.tensor(X_feat_scaled[cut:val_cut], dtype=torch.float32, device=device)
        vy = torch.tensor(Y[cut:val_cut], dtype=torch.float32, device=device)

        # El peso positivo se obtiene del train, evitando imponer una magnitud
        # arbitraria a cada número.
        pos = ty.mean(dim=0)
        neg = 1.0 - pos
        pos_weight = torch.clamp(neg / torch.clamp(pos, min=1e-4), 1.0, 8.0).to(device)

        ds = TensorDataset(tx, tf, ty)
        loader = DataLoader(
            ds,
            batch_size=min(128, max(32, len(ds) // 12)),
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.0007,
            weight_decay=2e-4,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best = float("inf")
        best_state = None
        bad = 0

        max_epochs = 60
        for epoch in range(max_epochs):
            self.model.train()
            for xb, fb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                fb = fb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(xb, fb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                logits = self.model(vx, vf)
                val_loss = criterion(logits, vy).item()
            scheduler.step(val_loss)

            if val_loss + 1e-7 < best:
                best = val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                bad = 0
            else:
                bad += 1

            self.epochs_done = epoch + 1
            if bad >= 9:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        self.best_val_loss = best
        self.fitted = True

        log(
            f"Red neuronal entrenada: épocas={self.epochs_done}, "
            f"val_loss={best:.6f}"
        )
        return {
            "available": True,
            "trained": True,
            "epochs": self.epochs_done,
            "val_loss": best,
        }

    def predict(self, X_seq_last, X_feat_last):
        if not self.fitted or self.model is None:
            return None

        device = torch.device(DEVICE)
        feat = self.feature_scaler.transform(
            np.asarray(X_feat_last, dtype=np.float32)
        ).astype(np.float32)

        with torch.no_grad():
            xs = torch.tensor(X_seq_last, dtype=torch.float32, device=device)
            xf = torch.tensor(feat, dtype=torch.float32, device=device)
            logits = self.model(xs, xf)
            p = torch.sigmoid(logits).cpu().numpy()

        return _normalize_rows(p)


class MaximumNeuralTrainer:
        """
        Ensemble de Transformers independientes.

        No sustituye la red existente conceptualmente: conserva la misma
        entrada/salida, pero entrena varias instancias con semillas distintas
        y promedia sus probabilidades para reducir dependencia de una única
        inicialización.
        """

        def __init__(self, ensemble_size=MAX_NEURAL_ENSEMBLE):
            self.ensemble_size = max(1, int(ensemble_size))
            self.models = []
            self.model = None
            self.feature_scaler = None
            self.fitted = False
            self.epochs_done = 0
            self.best_val_loss = None
            self.reports = []

        def fit(self, X_seq, X_feat, Y, progress_callback=None):
            X_seq = np.asarray(X_seq, dtype=np.float32)
            X_feat = np.asarray(X_feat, dtype=np.float32)
            Y = np.asarray(Y, dtype=np.float32)

            n = len(X_feat)
            if n < 180:
                log("Red máxima: muestras insuficientes; se omite.", "WARNING")
                return {"available": True, "trained": False}

            split = max(120, int(n * 0.82))
            val_cut = max(split + 20, int(n * 0.94))
            val_cut = min(val_cut, n)

            self.feature_scaler = StandardScaler()
            feat_scaled = self.feature_scaler.fit_transform(X_feat).astype(np.float32)

            device = torch.device(DEVICE)
            self.models = []
            self.reports = []

            # Semillas distintas = modelos correlacionados, pero no idénticos.
            seeds = [
                ULTIMATE_SEED + 11 * i
                for i in range(self.ensemble_size)
            ]

            for model_idx, seed in enumerate(seeds):
                log(
                    f"Red neuronal máxima {model_idx + 1}/{self.ensemble_size} "
                    f"(seed={seed})..."
                )
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                model = UltimateSequenceNet(
                    feature_size=X_feat.shape[1],
                    d_model=160,
                    heads=8,
                    layers=4,
                ).to(device)

                tx = torch.tensor(X_seq[:split], dtype=torch.float32, device=device)
                tf = torch.tensor(feat_scaled[:split], dtype=torch.float32, device=device)
                ty = torch.tensor(Y[:split], dtype=torch.float32, device=device)
                vx = torch.tensor(X_seq[split:val_cut], dtype=torch.float32, device=device)
                vf = torch.tensor(feat_scaled[split:val_cut], dtype=torch.float32, device=device)
                vy = torch.tensor(Y[split:val_cut], dtype=torch.float32, device=device)

                pos = ty.mean(dim=0)
                neg = 1.0 - pos
                pos_weight = torch.clamp(
                    neg / torch.clamp(pos, min=1e-4), 1.0, 8.0
                )

                ds = TensorDataset(
                    tx.detach().cpu(),
                    tf.detach().cpu(),
                    ty.detach().cpu(),
                )
                batch_size = min(192, max(32, len(ds) // 10))
                loader = DataLoader(
                    ds,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=(device.type == "cuda"),
                )

                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=5e-4,
                    weight_decay=3e-4,
                )
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=8, T_mult=2, eta_min=5e-6
                )
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

                best = float("inf")
                best_state = None
                bad = 0
                epochs = 90

                for epoch in range(epochs):
                    if progress_callback:
                        base = 35.0 + (model_idx / max(1, self.ensemble_size)) * 35.0
                        span = 35.0 / max(1, self.ensemble_size)
                        progress_callback(base + ((epoch + 1) / epochs) * span, f"Red neuronal {model_idx+1}/{self.ensemble_size}: época {epoch+1}/{epochs}")
                    model.train()
                    for xb, fb, yb in loader:
                        xb = xb.to(device, non_blocking=True)
                        fb = fb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True)

                        optimizer.zero_grad(set_to_none=True)
                        logits = model(xb, fb)
                        loss = criterion(logits, yb)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                    scheduler.step(epoch + 1)

                    model.eval()
                    with torch.no_grad():
                        val_logits = model(vx, vf)
                        val_loss = criterion(val_logits, vy).item()

                    if val_loss + 1e-7 < best:
                        best = val_loss
                        best_state = {
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                        }
                        bad = 0
                    else:
                        bad += 1

                    if bad >= 12:
                        break

                if best_state is not None:
                    model.load_state_dict(best_state)

                model.eval()
                self.models.append(model)
                self.reports.append({
                    "seed": seed,
                    "epochs": epoch + 1,
                    "val_loss": float(best),
                })

                log(
                    f"Red máxima {model_idx + 1}: "
                    f"épocas={epoch + 1}, val_loss={best:.6f}"
                )

            self.model = self.models[0] if self.models else None
            self.epochs_done = max(
                (r["epochs"] for r in self.reports), default=0
            )
            self.best_val_loss = min(
                (r["val_loss"] for r in self.reports), default=None
            )
            self.fitted = bool(self.models)

            return {
                "available": True,
                "trained": self.fitted,
                "ensemble_size": len(self.models),
                "epochs": self.epochs_done,
                "best_val_loss": self.best_val_loss,
                "members": self.reports,
            }

        def predict(self, X_seq_last, X_feat_last):
            if not self.fitted or not self.models or self.feature_scaler is None:
                return None

            device = torch.device(DEVICE)
            feat = self.feature_scaler.transform(
                np.asarray(X_feat_last, dtype=np.float32)
            ).astype(np.float32)

            xs = torch.tensor(
                X_seq_last, dtype=torch.float32, device=device
            )
            xf = torch.tensor(
                feat, dtype=torch.float32, device=device
            )

            preds = []
            with torch.no_grad():
                for model in self.models:
                    model.eval()
                    logits = model(xs, xf)
                    preds.append(torch.sigmoid(logits).cpu().numpy())

            return _normalize_rows(np.mean(preds, axis=0))


# ---------------------------------------------------------------------
# Predictor definitivo ampliado
# ---------------------------------------------------------------------
class UltimateFeatureBuilder:
    """
    Construye explícitamente la muestra que predice el sorteo siguiente.
    Esto corrige un detalle crucial: X[-1] del dataset supervisado predice
    el último sorteo conocido, no el siguiente sorteo futuro.
    """

    @staticmethod
    def future_sample(data, seq_len=SEQ_LEN):
        draws = [tuple(d) for _, d in data]
        if len(draws) < seq_len:
            raise ValueError("No hay suficientes sorteos para la ventana temporal.")

        i = len(draws) - 1
        seq = np.asarray(
            [one_hot_draw(draws[k]) for k in range(i - seq_len + 1, i + 1)],
            dtype=np.float32,
        )[None, :, :]

        feat = FeatureEngineeringExtreme.number_features(data, i)
        return seq, np.asarray(feat, dtype=np.float32)[None, :]


class UltimateWeightOptimizer:
    """
    Selecciona los pesos del ensemble en un conjunto de validación temporal.
    No fija arbitrariamente 75/15/10.
    """

    def __init__(self):
        self.weights = np.array([0.50, 0.20, 0.20, 0.10], dtype=float)
        self.report = {}

    def optimize(self, y, components):
        # components = [stat, logistic, gradient, neural]
        valid = [(i, p) for i, p in enumerate(components) if p is not None]
        if not valid or len(y) == 0:
            return self.weights

        # Una rejilla fina y barata evita introducir un optimizador que pueda
        # sobreajustarse de forma excesiva al pequeño conjunto de validación.
        candidates = []
        grid = np.arange(0.0, 1.01, 0.05)

        # Siempre incluimos combinaciones de 2/3/4 componentes.
        for ws in np.ndindex(*(len(grid),) * len(valid)):
            vals = np.asarray([grid[i] for i in ws], dtype=float)
            if vals.sum() == 0:
                continue
            vals /= vals.sum()

            mix = np.zeros_like(valid[0][1], dtype=float)
            for w, (_, p) in zip(vals, valid):
                mix += w * p

            brier = _brier_multilabel(y, mix)
            hits = _topk_hits(y, mix, 12)
            # Función objetivo: prioriza calibración y, en segundo término,
            # capacidad de concentrar aciertos.
            objective = brier - 0.015 * hits
            candidates.append((objective, vals.copy(), brier, hits))

        candidates.sort(key=lambda x: x[0])
        best = candidates[0]

        out = np.zeros(4, dtype=float)
        for w, (idx, _) in zip(best[1], valid):
            out[idx] = w
        self.weights = out
        self.report = {
            "brier": float(best[2]),
            "hits_top12": float(best[3]),
            "weights": out.tolist(),
        }
        return out


class MaximumUltraPredictor(UltraPredictor):
    """
    Mantiene la API de UltraPredictor y aumenta la capacidad del ensemble.
    """

    def __init__(self):
        super().__init__()
        if ULTIMATE_SKLEARN:
            self.ml_model = MaximumSklearnEnsemble()
        if TORCH_AVAILABLE:
            self.neural = MaximumNeuralTrainer(MAX_NEURAL_ENSEMBLE)
        self.weights = np.array([0.42, 0.13, 0.25, 0.20], dtype=float)


    def optimize_hyperparameters(self, progress_callback=None):
        """
        Optimización temporal de pesos sin la explosión combinatoria de la
        rejilla 21^4. Se evalúan miles de mezclas Dirichlet + puntos extremos
        y una búsqueda local alrededor de la mejor solución.
        """
        if self.data is None or self.X_feat is None:
            raise ValueError("Primero prepara los datos")

        n = len(self.X_feat)
        # La red y Scikit-learn se entrenan hasta el 84 %.
        # La optimización de pesos debe empezar DESPUÉS de ese punto para
        # que la validación sea realmente fuera de muestra y no comparta
        # observaciones con el entrenamiento.
        cut = int(n * 0.84)
        val_end = int(n * 0.95)
        if cut < 250 or val_end <= cut + 10:
            return {"status": "insufficient_validation_data"}

        # Para que la validación sea realmente fuera de muestra, cada fila
        # estadística se calcula usando sólo el histórico disponible entonces.
        stat_rows = []
        y_rows = []
        for j in range(cut, val_end):
            data_index = SEQ_LEN - 1 + j
            if data_index >= len(self.data) - 1:
                break
            p, _ = StatisticalEngine(
                self.data[:data_index + 1]
            ).probability_vector()
            stat_rows.append(_normalize_mass(p))
            y_rows.append(self.Y[j])

        if not stat_rows:
            return {"status": "no_validation_rows"}

        stat_val = np.asarray(stat_rows, dtype=float)
        y_val = np.asarray(y_rows, dtype=float)

        comps = [stat_val, None, None, None]

        if self.ml_model is not None and self.ml_model.fitted:
            try:
                a, b = self.ml_model.predict_components(
                    self.X_feat[cut:cut + len(stat_val)]
                )
                comps[1], comps[2] = a, b
            except Exception as exc:
                log(f"ML no disponible durante optimización: {exc}", "WARNING")

        if getattr(self.neural, "fitted", False):
            try:
                comps[3] = self.neural.predict(
                    self.X_seq[cut:cut + len(stat_val)],
                    self.X_feat[cut:cut + len(stat_val)],
                )
            except Exception as exc:
                log(f"Neural no disponible durante optimización: {exc}", "WARNING")

        valid = [(i, _normalize_rows(p)) for i, p in enumerate(comps) if p is not None]
        if not valid:
            return {"status": "no_components"}

        rng = np.random.default_rng(ULTIMATE_SEED + 909)
        candidates = []

        # Puntos unitarios y mezclas uniformes: sirven de referencias.
        for i in range(len(valid)):
            w = np.zeros(len(valid), dtype=float)
            w[i] = 1.0
            candidates.append(w)
        candidates.append(np.ones(len(valid), dtype=float) / len(valid))

        # 30.000 mezclas aleatorias. Dirichlet favorece tanto soluciones
        # concentradas como distribuidas según alpha.
        total_candidates = 30000
        if progress_callback:
            progress_callback(74.0, "optimizador: generando 30.000 combinaciones de pesos Dirichlet")
        for alpha in (0.25, 0.5, 1.0, 2.0):
            m = total_candidates // 4
            samples = rng.dirichlet(
                np.full(len(valid), alpha, dtype=float),
                size=m,
            )
            candidates.extend(samples)

        # Evaluación vectorizada por lotes: conserva las mismas 30.000 muestras
        # y el mismo objetivo, reduciendo mucho el coste de los bucles Python.
        best = None
        total_eval = len(candidates)
        weights_all = np.asarray(candidates, dtype=np.float64)
        component_stack = np.stack([p for _, p in valid], axis=0)
        batch_size = 256
        for batch_start in range(0, total_eval, batch_size):
            batch_end = min(total_eval, batch_start + batch_size)
            if progress_callback:
                pct = 74.0 + 8.0 * (batch_end / max(1, total_eval))
                progress_callback(pct, f"optimizador: evaluando pesos {batch_end:,}/{total_eval:,}")
            mixes = np.einsum("bc,crn->brn", weights_all[batch_start:batch_end], component_stack, optimize=True)
            mixes = mixes / (mixes.sum(axis=2, keepdims=True) + 1e-12)
            yb = y_val[None, :, :]
            briers = np.mean((yb - mixes) ** 2, axis=(1, 2))
            top12_idx = np.argpartition(mixes, -12, axis=2)[:, :, -12:]
            top18_idx = np.argpartition(mixes, -18, axis=2)[:, :, -18:]
            actual = y_val[None, :, :] > 0.5
            hits12s = np.mean(np.sum(np.take_along_axis(actual, top12_idx, axis=2), axis=2), axis=1)
            hits18s = np.mean(np.sum(np.take_along_axis(actual, top18_idx, axis=2), axis=2), axis=1)
            objectives = briers - 0.010 * hits12s - 0.004 * hits18s
            local_idx = int(np.argmin(objectives))
            item = (float(objectives[local_idx]), weights_all[batch_start + local_idx].copy(), float(briers[local_idx]), float(hits12s[local_idx]), float(hits18s[local_idx]))
            if best is None or item[0] < best[0]:
                best = item

        # Refinamiento local: perturbaciones pequeñas alrededor del mejor.
        if best is not None:
            local = best[1].copy()
            total_local = 4 * 2500
            local_done = 0
            for radius in (0.10, 0.05, 0.02, 0.01):
                for _ in range(2500):
                    local_done += 1
                    if progress_callback and (local_done == 1 or local_done % 500 == 0 or local_done == total_local):
                        pct = 82.0 + 7.0 * (local_done / total_local)
                        progress_callback(pct, f"optimizador: refinamiento local {local_done:,}/{total_local:,}")
                    delta = rng.normal(0.0, radius, size=len(local))
                    trial = np.clip(local + delta, 0.0, None)
                    if trial.sum() <= 0:
                        continue
                    trial /= trial.sum()

                    mix = np.zeros_like(valid[0][1], dtype=float)
                    for weight, (_, p) in zip(trial, valid):
                        mix += weight * p
                    mix = _normalize_rows(mix)

                    brier = _brier_multilabel(y_val, mix)
                    hits12 = _topk_hits(y_val, mix, 12)
                    hits18 = _topk_hits(y_val, mix, 18)
                    objective = brier - 0.010 * hits12 - 0.004 * hits18

                    if objective < best[0]:
                        best = (
                            objective, trial.copy(),
                            brier, hits12, hits18
                        )
                        local = trial.copy()

        weights = np.zeros(4, dtype=float)
        if best is not None:
            for weight, (idx, _) in zip(best[1], valid):
                weights[idx] = weight
            self.weights = weights
            self.training_report["weight_optimization"] = {
                "method": "Dirichlet_30000_plus_local_refinement_vectorized",
                "brier": float(best[2]),
                "hits_top12": float(best[3]),
                "hits_top18": float(best[4]),
                "weights": weights.tolist(),
                "validation_rows": int(len(y_val)),
            }
            log(
                "Pesos optimizados temporalmente: "
                + ", ".join(f"{x:.3f}" for x in weights)
            )
            return self.training_report["weight_optimization"]

        return {"status": "optimization_failed"}

    def train_ensemble(self, n_models=MAX_NEURAL_ENSEMBLE, progress_callback=None):
        if self.data is None:
            raise ValueError("Primero prepara los datos")

        log("=== ENTRENAMIENTO MÁXIMO DEL ENSEMBLE ===")
        log(
            f"Capacidad: CPU workers={MAX_CPU_WORKERS}, "
            f"neural ensemble={MAX_NEURAL_ENSEMBLE}, "
            f"ExtraTrees={MAX_EXTRA_SKLEARN}, "
            f"XGBoost={MAX_XGBOOST_AVAILABLE}"
        )
        started = time.time()

        n = len(self.X_feat)
        split = int(n * 0.84)
        val_end = max(split + 20, int(n * 0.92))
        val_end = min(val_end, n)

        Xtr, Xv = self.X_feat[:split], self.X_feat[split:val_end]
        Ytr, Yv = self.Y[:split], self.Y[split:val_end]

        report = {
            "timestamp": datetime.now().isoformat(),
            "samples": int(n),
            "features": int(self.X_feat.shape[1]),
            "sklearn": bool(ULTIMATE_SKLEARN),
            "pytorch": bool(TORCH_AVAILABLE),
            "extra_trees": bool(MAX_EXTRA_SKLEARN),
            "random_forest": bool(MAX_EXTRA_SKLEARN),
            "xgboost": bool(MAX_XGBOOST_AVAILABLE),
            "cpu_workers": int(MAX_CPU_WORKERS),
        }

        if self.ml_model is not None:
            log(
                "Entrenando Scikit-learn ampliado: "
                "Logistic + HistGradient + ExtraTrees + RandomForest"
                + (" + XGBoost" if MAX_XGBOOST_AVAILABLE else "")
            )
            sk_report = self.ml_model.fit(Xtr, Ytr, Xv, Yv, progress_callback=progress_callback)
            report["sklearn_report"] = sk_report

        if TORCH_AVAILABLE:
            log(
                f"Entrenando ensemble neuronal de {MAX_NEURAL_ENSEMBLE} redes..."
            )
            if progress_callback:
                progress_callback(35.0, "Scikit-learn terminado; iniciando redes neuronales")
            nn_report = self.neural.fit(
                self.X_seq, self.X_feat, self.Y, progress_callback=progress_callback
            )
            report["neural_report"] = nn_report
            self.torch_model = self.neural.model

        # Optimización temporal de pesos conservada.
        if progress_callback:
            progress_callback(72.0, "redes neuronales terminadas; optimizando pesos temporalmente")
        try:
            self.optimize_hyperparameters(progress_callback=progress_callback)
            if progress_callback:
                progress_callback(90.0, "optimización temporal terminada; guardando ensemble")
        except Exception as exc:
            log(f"No se pudieron optimizar pesos: {exc}", "WARNING")

        self._trained_signature = hashlib.sha256(
            f"{self.data[0][0]}|{self.data[-1][0]}|{len(self.data)}|MAXIMUM".encode()
        ).hexdigest()

        if TORCH_AVAILABLE and getattr(self.neural, "fitted", False):
            try:
                torch.save(
                    {
                        "models": [
                            m.state_dict() for m in self.neural.models
                        ],
                        "feature_size": int(self.X_feat.shape[1]),
                        "scaler": self.neural.feature_scaler,
                        "signature": self._trained_signature,
                        "ensemble_size": len(self.neural.models),
                    },
                    ULTIMATE_TORCH_FILE,
                )
            except Exception as exc:
                log(f"No se pudo guardar el ensemble neuronal: {exc}", "WARNING")

        report["weights"] = self.weights.tolist()
        report["elapsed_seconds"] = time.time() - started
        self.training_report = report
        _safe_json_write(ULTIMATE_STATE_FILE, report)

        log(
            f"=== ENTRENAMIENTO MÁXIMO TERMINADO en "
            f"{report['elapsed_seconds']:.1f}s ==="
        )
        return self

    def predict_ultra(self, n_mc_iterations=500000, n_combinations=15):
        # Conservamos toda la lógica anterior y elevamos los límites.
        return super().predict_ultra(
            n_mc_iterations=min(max(50000, int(n_mc_iterations)), 500000),
            n_combinations=min(max(5, int(n_combinations)), 30),
        )


# Sustituimos únicamente las implementaciones definitivas por las ampliadas.
# La UI, scraping, auditorías, estadísticas y flujo de botones permanecen.
UltraPredictor = MaximumUltraPredictor

# Mensaje de capacidad adicional en consola.
log(
    "Extensión MAXIMUM COMPUTE cargada: "
    f"workers={MAX_CPU_WORKERS}, "
    f"ExtraTrees={MAX_EXTRA_SKLEARN}, "
    f"XGBoost={MAX_XGBOOST_AVAILABLE}, "
    f"neural_ensemble={MAX_NEURAL_ENSEMBLE}"
)


# ---------------------------------------------------------------------
# MAIN: usa la interfaz original, pero con el predictor definitivo.
# ---------------------------------------------------------------------


# ================================================================
# FUXING CLI — ORQUESTADOR DIARIO + SEMANAL
# ================================================================
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CLI_RESULT_FILE = Path("fuxing_resultado_telegram.json")
CLI_RUN_REPORT = Path("fuxing_run_report.json")


def cli_print(msg="", level="INFO"):
    """Salida CMD equivalente al monitor de la GUI."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    elapsed = (datetime.now() - logger.start_time).total_seconds()
    print(f"[{timestamp}] [{level}] {msg} (Δ{elapsed:.1f}s)", flush=True)


def cli_progress(pct, msg):
    pct = max(0.0, min(100.0, float(pct)))
    cli_print(f"PROGRESO {pct:6.2f}% — {msg}", "PROGRESS")


def _fmt_combo(combo):
    return " - ".join(f"{int(n):02d}" for n in combo)


def _json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, tuple):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    return obj


def _daily_result_report(result, data, elapsed_seconds):
    combos = result.get("combinations", [])
    top = result.get("top_numbers", [])[:20]
    return {
        "mode": "diaria",
        "target": "próximo sorteo",
        "combination": [int(x) for x in combos[0]] if combos else [],
        "combinations": [[int(x) for x in c] for c in combos],
        "top_numbers": [int(x) for x in top],
        "confidence": float(result.get("confidence", 0.0)),
        "model_agreement": float(result.get("model_agreement", 0.0)),
        "weights": _json_safe(result.get("weights", [])),
        "training_report": _json_safe(result.get("training_report", {})),
        "historical_draws": len(data),
        "first_date": data[0][0].strftime("%Y-%m-%d") if data else None,
        "last_date": data[-1][0].strftime("%Y-%m-%d") if data else None,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": float(elapsed_seconds),
        "warning": result.get("warning", ""),
    }


def _weekly_result_report(result, data, elapsed_seconds):
    bt = result.get("backtest", {})
    details = result.get("details", {})
    return {
        "mode": "semanal",
        "target": "misma combinación durante 7 sorteos",
        "combination": [int(x) for x in result["combination"]],
        "stability_score": float(result.get("stability_score", 0.0)),
        "concentration_score": float(result.get("concentration_score", 0.0)),
        "analog_reliability": float(details.get("analog_reliability", 0.0)),
        "analog_count": int(details.get("analog_count", 0)),
        "pair_count": int(details.get("pair_count", 0)),
        "triple_count": int(details.get("triple_count", 0)),
        "backtest": _json_safe(bt),
        "historical_draws": len(data),
        "first_date": data[0][0].strftime("%Y-%m-%d") if data else None,
        "last_date": data[-1][0].strftime("%Y-%m-%d") if data else None,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": float(elapsed_seconds),
        "warning": "Robustez estadística no equivale a certeza de acierto.",
    }


def run_daily_only(predictor, data, started, final_p, components):
    cli_print("========== PROCESO DIARIO ==========")
    cli_print("5A/5 — Generando combinaciones diarias a partir de la señal común ya calculada", "PREDICTION")

    # Esta parte reproduce la fase de predict_ultra que queda después de obtener
    # final_p. El cálculo costoso de componentes + Monte Carlo se hace una sola vez
    # y lo comparte también la modalidad semanal.
    generator = CombinationEngine(data, final_p)
    combos = generator.generate(
        n=15,
        iterations=400000,
        seed=ULTIMATE_SEED + 7,
    )

    matrix = np.vstack([
        _normalize_rows(p)[0] for p in components if p is not None
    ])
    uncertainty = np.std(matrix, axis=0)
    agreement = 1.0 - float(np.mean(uncertainty))
    agreement = float(np.clip(agreement, 0.0, 1.0))

    result = {
        "combinations": combos,
        "probabilities": final_p,
        "statistical_probability": components[0][0],
        "logistic_probability": components[1][0] if components[1] is not None else None,
        "gradient_probability": components[2][0] if components[2] is not None else None,
        "neural_probability": components[3][0] if components[3] is not None else None,
        "uncertainty": uncertainty,
        "confidence": agreement,
        "weights": predictor.weights.copy(),
        "top_numbers": np.argsort(final_p)[::-1] + 1,
        "training_report": predictor.training_report,
        "model_agreement": agreement,
        "warning": "La consistencia del ensemble no es la probabilidad matemática de acertar 6/6.",
    }
    predictor.last_result = result

    elapsed = time.time() - started
    report = _daily_result_report(result, data, elapsed)
    combo = report["combination"]
    cli_print(f"DIARIA — COMBINACIÓN FINAL: {_fmt_combo(combo)}", "PREDICTION")
    cli_print(f"DIARIA — TOP 20: {', '.join(f'{n:02d}' for n in report['top_numbers'])}", "PREDICTION")
    cli_print(f"DIARIA — ACUERDO ENTRE COMPONENTES: {report['confidence']*100:.2f}%", "PREDICTION")
    cli_print(f"DIARIA — SORTEOS HISTÓRICOS ANALIZADOS: {len(data):,}", "AUDIT")
    cli_print(f"DIARIA — TIEMPO ESPECÍFICO: {elapsed:.1f}s")
    return result, report

def run_weekly_only(data, model_probabilities, started):
    cli_print("========== PROCESO SEMANAL ==========")
    cli_print("5/5 — Calculando estrategia semanal robusta", "PREDICTION")
    engine = WeeklyStrategyEngine(
        data,
        progress_callback=lambda p, m: cli_progress(84.0 + float(p) * 0.16, m),
        model_probabilities=model_probabilities,
    )
    result = engine.predict(
        mc_iterations=180000,
        run_backtest=True,
    )
    elapsed = time.time() - started
    report = _weekly_result_report(result, data, elapsed)
    cli_print(f"SEMANAL — COMBINACIÓN FINAL PARA 7 SORTEOS: {_fmt_combo(report['combination'])}", "PREDICTION")
    cli_print(f"SEMANAL — SORTEOS HISTÓRICOS ANALIZADOS: {len(data):,}", "AUDIT")
    cli_print(f"SEMANAL — ESTABILIDAD: {report['stability_score']*100:.2f}%", "PREDICTION")
    cli_print(f"SEMANAL — ANALOGÍAS HISTÓRICAS: {report['analog_count']}", "PREDICTION")
    bt = report["backtest"]
    if bt.get("tests"):
        cli_print(
            f"SEMANAL — BACKTEST: {bt['tests']} semanas / {bt['draws_evaluated']} sorteos | "
            f"media={bt['mean_hits_per_draw']:.3f} | mejor semana={bt['mean_best_hits_week']:.3f}",
            "AUDIT",
        )
    cli_print(f"SEMANAL — TIEMPO ESPECÍFICO: {elapsed:.1f}s")
    return result, report


def build_model_probability_reference(predictor):
    """Calcula una sola vez la señal final que necesita la fase diaria/semanal.

    Reproduce la parte común de MaximumUltraPredictor.predict_ultra:
    componentes -> pesos activos -> Monte Carlo -> final_p.
    A partir de aquí, la generación de combinaciones diaria y el motor semanal
    se ejecutan en paralelo sobre exactamente el mismo histórico y la misma señal.
    """
    components = predictor._current_components()
    available = np.array([p is not None for p in components], dtype=bool)
    weights = predictor.weights.copy()
    weights[~available] = 0.0
    if weights.sum() <= 0:
        weights[available] = 1.0
    weights /= weights.sum()

    combined = np.zeros(MAX_NUM, dtype=float)
    for w, p in zip(weights, components):
        if p is not None:
            combined += w * _normalize_rows(p)[0]
    combined = _normalize_mass(combined)

    cli_print("Señal común: ejecutando Monte Carlo 500.000 iteraciones una sola vez…", "PREDICTION")
    mc = UltraMonteCarlo(n_iterations=500000, seed=ULTIMATE_SEED)
    mc_p6 = mc.sample_with_constraints(combined)
    mc_p6 = _normalize_mass(mc_p6)
    final_p = _normalize_mass(0.85 * combined + 0.15 * mc_p6)

    cli_print("Señal común diaria/semanal preparada.", "PREDICTION")
    return final_p, components, weights

def run_fuxing_cli(send_telegram=False):
    global logger
    overall_started = time.time()
    logger.start_time = datetime.now()

    cli_print("=" * 78)
    cli_print("FUXING — EDICIÓN CLI / CLOUD", "PREDICTION")
    cli_print(f"Python={__import__('sys').version.split()[0]} | CPU workers={MAX_CPU_WORKERS}")
    cli_print(f"PyTorch disponible={TORCH_AVAILABLE} | Scikit-learn={ULTIMATE_SKLEARN}")
    cli_print(f"ExtraTrees={MAX_EXTRA_SKLEARN} | XGBoost={MAX_XGBOOST_AVAILABLE} | neural_ensemble={MAX_NEURAL_ENSEMBLE}")
    cli_print("OBJETIVO: histórico compartido → preparación compartida → entrenamiento compartido → diaria + semanal EN PARALELO")
    cli_print("=" * 78)

    # ------------------------------------------------------------
    # 1. HISTÓRICO — una sola descarga/actualización
    # ------------------------------------------------------------
    stage_started = time.time()
    cli_print("1/5 — DESCARGANDO / CONTRASTANDO HISTÓRICO COMPLETO 1988 → AÑO ACTUAL", "NETWORK")
    scraper = DataScraper(start_year=START_YEAR, end_year=datetime.now().year)

    def scrape_progress(done, total, year, count, error):
        pct = 2.0 + (done / max(1, total)) * 18.0
        if error:
            cli_print(f"1/5 — AÑO {year}: error/reintento — {done}/{total}", "WARNING")
        else:
            cli_print(f"1/5 — AÑO {year}: {count} sorteos válidos — {done}/{total}", "DATA")
        cli_progress(pct, f"Histórico: año {year}")

    data = scraper.scrape_with_cache(force_refresh=False, progress_callback=scrape_progress)
    if not data:
        raise RuntimeError("El histórico está vacío.")
    cli_print(f"1/5 — HISTÓRICO LISTO: {len(data):,} sorteos | {data[0][0]:%d/%m/%Y} → {data[-1][0]:%d/%m/%Y}", "AUDIT")
    cli_print(f"1/5 — Tiempo: {time.time()-stage_started:.1f}s")

    # ------------------------------------------------------------
    # 2. PREPARACIÓN — una sola vez para ambos modos
    # ------------------------------------------------------------
    stage_started = time.time()
    cli_print("2/5 — PREPARANDO DATOS Y FEATURES UNA SOLA VEZ PARA DIARIA + SEMANAL", "MODEL")
    predictor = MaximumUltraPredictor()
    predictor.prepare_data(data)
    cli_progress(32, f"2/5 — Dataset preparado: {len(predictor.X_seq):,} muestras / {predictor.X_feat.shape[1]} features")
    cli_print(f"2/5 — Tiempo: {time.time()-stage_started:.1f}s")

    # ------------------------------------------------------------
    # 3. OPTIMIZACIÓN — una sola vez y compartida
    # ------------------------------------------------------------
    stage_started = time.time()
    cli_print("3/5 — OPTIMIZACIÓN TEMPORAL / BACKTEST DE PESOS — UNA SOLA VEZ", "OPTIM")
    predictor.optimize_hyperparameters(progress_callback=cli_progress)
    cli_print("3/5 — PESOS OPTIMIZADOS: " + ", ".join(f"{x:.3f}" for x in predictor.weights), "OPTIM")
    cli_print(f"3/5 — Tiempo: {time.time()-stage_started:.1f}s")

    # ------------------------------------------------------------
    # 4. ENTRENAMIENTO — una sola vez y compartido
    # ------------------------------------------------------------
    stage_started = time.time()
    cli_print("4/5 — ENTRENANDO ENSEMBLE UNA SOLA VEZ PARA AMBAS PREDICCIONES", "MODEL")
    predictor.train_ensemble(progress_callback=cli_progress)
    cli_print("4/5 — ENTRENAMIENTO COMPARTIDO TERMINADO", "MODEL")
    cli_print("4/5 — PESOS FINALES: " + ", ".join(f"{x:.3f}" for x in predictor.weights), "OPTIM")
    cli_print(f"4/5 — Tiempo: {time.time()-stage_started:.1f}s")

    # El weekly actual de la GUI recibía la probabilidad final del daily.
    # La calculamos aquí UNA VEZ y, desde ese punto, ambos procesos trabajan en paralelo.
    cli_print("Preparando señal común de probabilidad para el proceso semanal…", "MODEL")
    shared_model_probabilities, shared_components, shared_weights = build_model_probability_reference(predictor)
    cli_print("Señal común preparada. LANZANDO DIARIA + SEMANAL EN PARALELO.", "PREDICTION")

    # ------------------------------------------------------------
    # 5A + 5B — cálculos específicos en paralelo
    # ------------------------------------------------------------
    daily_started = time.time()
    weekly_started = time.time()
    results = {}
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="Fuxing-Pred") as pool:
        futures = {
            pool.submit(run_daily_only, predictor, data, daily_started, shared_model_probabilities, shared_components): "daily",
            pool.submit(run_weekly_only, data, shared_model_probabilities, weekly_started): "weekly",
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                result, report = future.result()
                results[name] = {"raw": result, "report": report}
                cli_print(f"PROCESO {name.upper()} TERMINADO CORRECTAMENTE.", "PREDICTION")
            except Exception as exc:
                cli_print(f"PROCESO {name.upper()} ERROR: {type(exc).__name__}: {exc}", "ERROR")
                raise

    # ------------------------------------------------------------
    # 6. AUDITORÍA FINAL — una sola vez, compartida
    # ------------------------------------------------------------
    cli_print("6/6 — AUDITORÍA FINAL COMPARTIDA DEL DATASET", "AUDIT")
    audit_report = dataset_audit(data)
    randomness_report = randomness_audit(data)

    total_elapsed = time.time() - overall_started
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "execution": {
            "mode": "daily_and_weekly_parallel",
            "historical_start": START_YEAR,
            "historical_end": datetime.now().year,
            "historical_draws": len(data),
            "first_date": data[0][0].strftime("%Y-%m-%d"),
            "last_date": data[-1][0].strftime("%Y-%m-%d"),
            "total_elapsed_seconds": total_elapsed,
            "parallel_specific_stage": True,
        },
        "daily": results["daily"]["report"],
        "weekly": results["weekly"]["report"],
        "audit": _json_safe(audit_report),
        "randomness_audit": _json_safe(randomness_report),
    }
    _safe_json_write(CLI_RESULT_FILE, payload)
    _safe_json_write(CLI_RUN_REPORT, payload)

    cli_print("=" * 78)
    cli_print("RESULTADOS FINALES", "PREDICTION")
    cli_print(f"DIARIA  → {_fmt_combo(results['daily']['report']['combination'])}", "PREDICTION")
    cli_print(f"SEMANAL → {_fmt_combo(results['weekly']['report']['combination'])}", "PREDICTION")
    cli_print(f"SORTEOS ANALIZADOS → {len(data):,}", "AUDIT")
    cli_print(f"HISTÓRICO → {data[0][0]:%d/%m/%Y} → {data[-1][0]:%d/%m/%Y}", "AUDIT")
    cli_print(f"TIEMPO TOTAL → {total_elapsed:.1f}s ({total_elapsed/3600:.2f} h)", "INFO")
    cli_print(f"RESULTADO TELEGRAM → {CLI_RESULT_FILE}", "INFO")
    cli_print("=" * 78)

    if send_telegram:
        launch_telegram_sender()

    return payload


def launch_telegram_sender():
    """Lanza el proceso Telegram separado, sin integrar su lógica en Fuxing."""
    import subprocess
    import sys
    candidates = [
        Path(__file__).with_name("Fuxing_Telegram.py"),
        Path(__file__).with_name("Fuxing_Telegram.exe"),
    ]
    target = next((p for p in candidates if p.exists()), None)
    if target is None:
        cli_print("No se encontró Fuxing_Telegram.py/.exe. Se conserva el JSON para envío externo.", "WARNING")
        return
    cli_print(f"Lanzando proceso Telegram separado: {target.name}", "NETWORK")
    if target.suffix.lower() == ".py":
        cmd = [sys.executable, str(target), "--once"]
    else:
        cmd = [str(target), "--once"]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.stdout:
        for line in completed.stdout.splitlines():
            print(line, flush=True)
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, flush=True)
        raise RuntimeError(f"Fuxing_Telegram terminó con código {completed.returncode}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fuxing CLI — diaria + semanal en paralelo")
    parser.add_argument("--send-telegram", action="store_true", help="envía el resultado mediante Fuxing_Telegram.py/.exe")
    args = parser.parse_args()
    try:
        run_fuxing_cli(send_telegram=args.send_telegram)
    except KeyboardInterrupt:
        cli_print("PROCESO DETENIDO POR EL USUARIO", "INFO")
        raise SystemExit(130)
    except Exception as exc:
        cli_print(f"FUXING FINALIZADO CON ERROR: {type(exc).__name__}: {exc}", "ERROR")
        raise

