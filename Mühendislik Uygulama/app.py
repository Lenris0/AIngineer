# -*- coding: utf-8 -*-
"""
İki aşamalı Mühendislik Analiz Uygulaması
- Giriş ekranı: Proje açıklaması → AI modül seçimi (Asansör, Teleferik, Vinç) → Projeyi Başlat
- Ana ekran: 3 bölme, History (geçmiş), gerçek zamanlı kıyaslama, 4 interaktif grafik, İdeal Durum analizi
"""
from __future__ import annotations

import re
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Sayfa ve modül sabitleri
# ---------------------------------------------------------------------------
PAGE_WELCOME = "welcome"
PAGE_DASHBOARD = "dashboard"

MODULES = {
    "asansör": "Asansör",
    "teleferik": "Teleferik",
    "vinç": "Vinç",
}

IDEAL_SAPMA_MAX_M = 0.02  # İdeal durum için max sapma (m)

# ---------------------------------------------------------------------------
# Sabitler ve halat kapasite tablosu (EN 12385, tipik çelik halat N kopma)
# ---------------------------------------------------------------------------
G = 9.81  # m/s²
MIN_SAFETY_FACTOR = 12
# Halat çapı (mm) -> birim kopma yükü (N/mm²) veya doğrudan kopma (N) tablosu
# Basit model: F_kopma_tek_halat ≈ k * d_mm^2 ; k = 400–500
ROPE_STRENGTH_N_PER_MM2 = 460

# Tipik halat kapasite tablosu (mm -> min kopma yükü N, tek halat)
ROPE_CAPACITY_TABLE = {
    8: 23_000,
    10: 36_000,
    13: 61_000,
    16: 92_000,
    20: 145_000,
    22: 175_000,
    24: 208_000,
}


# ---------------------------------------------------------------------------
# Giriş ekranı: Proje analizi ve modül seçimi
# ---------------------------------------------------------------------------

def analyze_project_description(text: str) -> tuple[str, str]:
    """
    Proje açıklamasından mühendislik modülünü belirler.
    Returns: (module_key, module_name)
    """
    t = text.lower().strip()
    if not t:
        return "asansör", "Asansör"
    scores = {}
    for key, name in MODULES.items():
        score = 0
        if key == "asansör":
            for w in ("asansör", "asansor", "dikey taşıma", "kabin", "binada", "yolcu", "yük taşıma"):
                if w in t:
                    score += 2
        elif key == "teleferik":
            for w in ("teleferik", "teleferik", "havai hat", "kablo", "dağ", "kayak", "taşıma hattı"):
                if w in t:
                    score += 2
        elif key == "vinç":
            for w in ("vinç", "vinc", "kren", "kaldırma", "inşaat", "ağır yük", "tower"):
                if w in t:
                    score += 2
        scores[key] = score
    best_key = max(scores, key=scores.get)
    if scores[best_key] == 0:
        best_key = "asansör"
    return best_key, MODULES[best_key]


def get_ideal_durum_analysis(mevcut_emniyet_kat: float, hata_payi_sapma_m: float) -> str:
    """
    mevcut_emniyet_kat ve hata_payı (sapma) değişkenlerine göre İdeal Durum analizi.
    """
    ideal_sf = MIN_SAFETY_FACTOR
    sf_ok = mevcut_emniyet_kat >= ideal_sf
    sapma_ok = hata_payi_sapma_m <= IDEAL_SAPMA_MAX_M
    lines = [
        "**İdeal Durum analizi:**",
        f"- Emniyet katsayısı: hedef ≥ {ideal_sf}. Şu an **{mevcut_emniyet_kat:.1f}** → " + ("✓ uygun." if sf_ok else f"✗ {ideal_sf - mevcut_emniyet_kat:.1f} birim artırılmalı."),
        f"- Hata payı (sapma Δx): hedef ≤ {IDEAL_SAPMA_MAX_M} m. Şu an **{hata_payi_sapma_m:.4f} m** → " + ("✓ uygun." if sapma_ok else f"✗ %{(hata_payi_sapma_m - IDEAL_SAPMA_MAX_M) / IDEAL_SAPMA_MAX_M * 100:.0f} fazla; düşürülmeli."),
    ]
    if not sf_ok or not sapma_ok:
        lines.append("Öneri: Halat sayısı/çapı artırın veya hız/ivme/yükü düşürerek hem SF hem sapmayı ideal aralığa getirin.")
    return "\n".join(lines)


def build_realtime_comparison(
    prev: dict,
    cur_m_kg: float,
    cur_v_ms: float,
    cur_a_nominal: float,
    cur_n_ropes: int,
    cur_d_mm: float,
    cur_F_dyn: float,
    cur_sf: float,
    cur_sapma: float,
) -> str | None:
    """
    Önceki ayar ile şimdiki ayarı kıyasla; detaylı cümle döndür.
    Örnek: 'Önceki tasarımda hızın 2 m/s idi, şimdi 3 m/s yaptın. Halat gerilmesi %15 arttı, güvenlik katsayısı 12'den 10.2'ye düştü.'
    """
    if not prev or prev.get("F_dyn") is None:
        return None
    p_m, p_v, p_a = prev.get("m_kg"), prev.get("v_ms"), prev.get("a_nominal")
    p_F, p_sf, p_sapma = prev.get("F_dyn"), prev.get("safety_factor"), prev.get("sapma")
    parts = []
    if p_v is not None and abs(cur_v_ms - p_v) > 0.01:
        parts.append(f"Önceki tasarımda hızın **{p_v:.2f} m/s** idi, şimdi **{cur_v_ms:.2f} m/s** yaptın.")
    if p_m is not None and abs(cur_m_kg - p_m) > 1:
        parts.append(f"Yükü **{p_m:.0f} kg**'dan **{cur_m_kg:.0f} kg**'a çıkardın/indirdin.")
    if p_a is not None and abs(cur_a_nominal - p_a) > 0.01:
        parts.append(f"İvme **{p_a:.2f} m/s²**'den **{cur_a_nominal:.2f} m/s²**'ye değişti.")
    if not parts:
        return None
    tension_pct = (cur_F_dyn - p_F) / p_F * 100 if p_F and p_F > 0 else 0
    parts.append(f"Bu durum halat gerilmesini **%{tension_pct:+.1f}** değiştirdi")
    if p_sf is not None:
        parts.append(f", güvenlik katsayısını **{p_sf:.1f}**'den **{cur_sf:.1f}**'ye " + ("düşürdü." if cur_sf < p_sf else "çıkardı."))
    else:
        parts.append(".")
    if p_sapma is not None and p_sapma > 0:
        sapma_pct = (cur_sapma - p_sapma) / p_sapma * 100
        parts.append(f" Durma sapması **%{sapma_pct:+.1f}** değişti.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Action-Oriented AI: Komut ayrıştırma, doğrulama, hafıza, hata raporu
# ---------------------------------------------------------------------------

PARAM_LIMITS = {"m_kg": (200, 2000), "v_ms": (0.25, 4.0), "a_nominal": (0.2, 2.0), "n_ropes": (2, 12), "d_mm": (8, 24)}
STEP_INCREASE = {"m_kg": 100, "v_ms": 0.2, "a_nominal": 0.1, "n_ropes": 1, "d_mm": 1}


def _clamp(param: str, value: float) -> float:
    lo, hi = PARAM_LIMITS[param]
    if param == "n_ropes":
        return int(np.clip(round(value), lo, hi))
    return float(np.clip(value, lo, hi))


def parse_action_command(
    msg: str,
    m_kg: float,
    v_ms: float,
    a_nominal: float,
    n_ropes: int,
    d_mm: float,
    last_action: dict | None,
) -> tuple[dict | None, str, bool, str]:
    """
    Kullanıcı metnindeki komutları çözümle; önerilen güncellemeyi döndür.
    Returns: (updates_dict | None, reply_part, needs_confirm, warning_message)
    """
    t = msg.lower().strip()
    updates = {}
    reply_part = ""
    needs_confirm = False
    warning = ""

    # Sayı çıkarma (örn: 1000, 2.5, 16)
    numbers = re.findall(r"\d+[.,]?\d*", t.replace(",", "."))
    nums = [float(n) for n in numbers]

    # "biraz daha artır" / "biraz daha düşür" -> önceki aksiyona göre
    if "biraz daha" in t or "biraz daha artır" in t or "daha artır" in t:
        if last_action and last_action.get("param"):
            p = last_action["param"]
            step = STEP_INCREASE.get(p, 1)
            if last_action.get("action") == "increase":
                cur = {"m_kg": m_kg, "v_ms": v_ms, "a_nominal": a_nominal, "n_ropes": n_ropes, "d_mm": d_mm}[p]
                updates[p] = _clamp(p, cur + step)
                reply_part = f"Önceki isteğe göre **{p}** tekrar artırılıyor."
            elif last_action.get("action") == "decrease":
                cur = {"m_kg": m_kg, "v_ms": v_ms, "a_nominal": a_nominal, "n_ropes": n_ropes, "d_mm": d_mm}[p]
                updates[p] = _clamp(p, cur - step)
                reply_part = f"Önceki isteğe göre **{p}** tekrar düşürülüyor."
        else:
            reply_part = "Hangi parametreyi artırayım? (Örn: 'hızı artır', 'yükü 1000 yap')"
        return (updates if updates else None, reply_part, needs_confirm, warning)

    # Yük / kütle
    if "yük" in t or "kütle" in t or "agirlik" in t or "m=" in t or (t.startswith("1000") and "kg" in t):
        for n in nums:
            if 100 <= n <= 3000:
                updates["m_kg"] = _clamp("m_kg", n)
                reply_part = f"Yük **{updates['m_kg']:.0f} kg** olarak ayarlanacak."
                break
        if "artır" in t and not updates:
            updates["m_kg"] = _clamp("m_kg", m_kg + STEP_INCREASE["m_kg"])
            reply_part = f"Yük **{updates['m_kg']:.0f} kg** yapılıyor (artırıldı)."
        elif "düşür" in t and not updates:
            updates["m_kg"] = _clamp("m_kg", m_kg - STEP_INCREASE["m_kg"])
            reply_part = f"Yük **{updates['m_kg']:.0f} kg** yapılıyor (düşürüldü)."
        if not updates and nums:
            updates["m_kg"] = _clamp("m_kg", nums[0])
            reply_part = f"Yük **{updates['m_kg']:.0f} kg** olarak ayarlanacak."
        return (updates if updates else None, reply_part, needs_confirm, warning)

    # Hız
    if "hız" in t or "v=" in t:
        for n in nums:
            if 0.2 <= n <= 5:
                updates["v_ms"] = _clamp("v_ms", n)
                reply_part = f"Hız **{updates['v_ms']:.2f} m/s** olarak ayarlanacak."
                break
        if "artır" in t and not updates:
            updates["v_ms"] = _clamp("v_ms", v_ms + STEP_INCREASE["v_ms"])
            reply_part = f"Hız **{updates['v_ms']:.2f} m/s** yapılıyor (artırıldı)."
        elif "düşür" in t and not updates:
            updates["v_ms"] = _clamp("v_ms", v_ms - STEP_INCREASE["v_ms"])
            reply_part = f"Hız **{updates['v_ms']:.2f} m/s** yapılıyor (düşürüldü)."
        if not updates and nums:
            updates["v_ms"] = _clamp("v_ms", nums[0])
            reply_part = f"Hız **{updates['v_ms']:.2f} m/s** olarak ayarlanacak."
        return (updates if updates else None, reply_part, needs_confirm, warning)

    # İvme
    if "ivme" in t or "a=" in t:
        for n in nums:
            if 0.1 <= n <= 2.5:
                updates["a_nominal"] = _clamp("a_nominal", n)
                reply_part = f"İvme **{updates['a_nominal']:.2f} m/s²** olarak ayarlanacak."
                break
        if "artır" in t and not updates:
            updates["a_nominal"] = _clamp("a_nominal", a_nominal + STEP_INCREASE["a_nominal"])
            reply_part = f"İvme **{updates['a_nominal']:.2f} m/s²** yapılıyor (artırıldı)."
        elif "düşür" in t and not updates:
            updates["a_nominal"] = _clamp("a_nominal", a_nominal - STEP_INCREASE["a_nominal"])
            reply_part = f"İvme **{updates['a_nominal']:.2f} m/s²** yapılıyor (düşürüldü)."
        if not updates and nums:
            updates["a_nominal"] = _clamp("a_nominal", nums[0])
            reply_part = f"İvme **{updates['a_nominal']:.2f} m/s²** olarak ayarlanacak."
        return (updates if updates else None, reply_part, needs_confirm, warning)

    # Halat sayısı
    if "halat sayı" in t or "halat sayisi" in t or "ropes" in t:
        for n in nums:
            if 2 <= n <= 12:
                updates["n_ropes"] = _clamp("n_ropes", n)
                reply_part = f"Halat sayısı **{updates['n_ropes']}** olarak ayarlanacak."
                break
        if not updates and nums:
            updates["n_ropes"] = _clamp("n_ropes", int(nums[0]))
            reply_part = f"Halat sayısı **{updates['n_ropes']}** olarak ayarlanacak."
        return (updates if updates else None, reply_part, needs_confirm, warning)

    # Halat çapı
    if "halat çap" in t or "çap" in t and "halat" in t or "d_mm" in t or "cap" in t:
        for n in nums:
            if 8 <= n <= 24:
                updates["d_mm"] = _clamp("d_mm", n)
                reply_part = f"Halat çapı **{updates['d_mm']:.0f} mm** olarak ayarlanacak."
                break
        if not updates and nums:
            updates["d_mm"] = _clamp("d_mm", nums[0])
            reply_part = f"Halat çapı **{updates['d_mm']:.0f} mm** olarak ayarlanacak."
        return (updates if updates else None, reply_part, needs_confirm, warning)

    # Sadece sayı (bağlam: yük gibi kabul et)
    if len(nums) == 1 and 200 <= nums[0] <= 2000 and ("yap" in t or "ol" in t or "et" in t):
        updates["m_kg"] = _clamp("m_kg", nums[0])
        reply_part = f"Yük **{updates['m_kg']:.0f} kg** olarak ayarlanacak."
        return (updates, reply_part, needs_confirm, warning)

    return (None, "", False, "")


def validate_proposed_params(
    current: dict,
    proposed: dict,
) -> tuple[bool, str]:
    """
    Önerilen parametreleri kontrol et. Riskliyse (SF düşer, sapma belirgin artar) True ve uyarı metni döndür.
    """
    m = proposed.get("m_kg", current["m_kg"])
    v = proposed.get("v_ms", current["v_ms"])
    a = proposed.get("a_nominal", current["a_nominal"])
    n = proposed.get("n_ropes", current["n_ropes"])
    d = proposed.get("d_mm", current["d_mm"])
    new_sf = compute_safety_factor(m, a, n, d)
    new_sapma, _, _ = compute_stopping_deviation(v, a)
    cur_sf = compute_safety_factor(current["m_kg"], current["a_nominal"], current["n_ropes"], current["d_mm"])
    cur_sapma, _, _ = compute_stopping_deviation(current["v_ms"], current["a_nominal"])
    warnings = []
    if new_sf < MIN_SAFETY_FACTOR:
        warnings.append(f"emniyet katsayısı {new_sf:.1f}'e düşecek (min 12).")
    elif new_sf < cur_sf and cur_sf >= MIN_SAFETY_FACTOR:
        warnings.append(f"emniyet katsayısı {cur_sf:.1f}'den {new_sf:.1f}'e düşecek.")
    if cur_sapma > 0 and new_sapma > cur_sapma * 1.05:
        pct = (new_sapma - cur_sapma) / cur_sapma * 100
        warnings.append(f"hata payı (sapma) yaklaşık %{pct:.0f} artacak ({cur_sapma:.4f} m → {new_sapma:.4f} m).")
    if not warnings:
        return False, ""
    return True, " " + " ".join(warnings)


def format_error_report(prev_snapshot: dict | None, new_m: float, new_v: float, new_a: float, new_n: int, new_d: float) -> str:
    """Önceki duruma göre hata payı ve SF değişimini sayısal rapor olarak ver."""
    if not prev_snapshot:
        return ""
    new_sf = compute_safety_factor(new_m, new_a, new_n, new_d)
    new_sapma, _, _ = compute_stopping_deviation(new_v, new_a)
    p_sf = prev_snapshot.get("safety_factor")
    p_sapma = prev_snapshot.get("sapma")
    lines = ["**Hata hesaplama raporu (önceki duruma göre):**"]
    if p_sapma is not None and p_sapma > 0:
        delta_pct = (new_sapma - p_sapma) / p_sapma * 100
        lines.append(f"- Hata payı (Δx): **{p_sapma:.4f} m** → **{new_sapma:.4f} m** (%{delta_pct:+.1f}).")
    if p_sf is not None:
        lines.append(f"- Emniyet katsayısı: **{p_sf:.1f}** → **{new_sf:.1f}**.")
    return "\n".join(lines)


def get_rope_breaking_force_from_table(d_mm: float) -> float:
    """Tablo değeri veya interpolasyon; tabloda yoksa formül."""
    d_int = int(round(d_mm))
    if d_int in ROPE_CAPACITY_TABLE:
        return float(ROPE_CAPACITY_TABLE[d_int])
    keys = sorted(ROPE_CAPACITY_TABLE.keys())
    if d_mm <= keys[0]:
        return ROPE_CAPACITY_TABLE[keys[0]] * (d_mm / keys[0]) ** 2
    if d_mm >= keys[-1]:
        return ROPE_CAPACITY_TABLE[keys[-1]] * (d_mm / keys[-1]) ** 2
    for i in range(len(keys) - 1):
        if keys[i] <= d_mm <= keys[i + 1]:
            t = (d_mm - keys[i]) / (keys[i + 1] - keys[i])
            return ROPE_CAPACITY_TABLE[keys[i]] * (1 - t) + ROPE_CAPACITY_TABLE[keys[i + 1]] * t
    return (np.pi / 4) * (d_mm ** 2) * ROPE_STRENGTH_N_PER_MM2


def compute_dynamic_force(m_kg: float, a_ms2: float) -> float:
    """F = m · (g + a) [N]"""
    return m_kg * (G + a_ms2)


def compute_rope_breaking_total(n_ropes: int, d_mm: float) -> float:
    """Toplam halat kopma yükü [N] - tablo + formül."""
    single = get_rope_breaking_force_from_table(d_mm)
    return n_ropes * single


def compute_safety_factor(m_kg: float, a_ms2: float, n_ropes: int, d_mm: float) -> float:
    """Emniyet katsayısı = Toplam kopma / F_dinamik"""
    F_dyn = compute_dynamic_force(m_kg, a_ms2)
    F_break = compute_rope_breaking_total(n_ropes, d_mm)
    if F_dyn <= 0:
        return float("inf")
    return F_break / F_dyn


# ---------------------------------------------------------------------------
# Gerçekçi hata formülü: Δx = (1/2)·a·t_delay² + v·t_sensor
# Sensör gecikmesi ve sistem toleransı (ivmeye bağlı) hesaba katılır.
# ---------------------------------------------------------------------------

def compute_stopping_deviation(v_ms: float, a_ms2: float) -> tuple[float, float, float]:
    """
    Durma mesafesindeki sapma: Δx = 0.5·a·t_delay² + v·t_sensor [m]
    t_delay: kontrol/sistem gecikmesi (s), ivme arttıkça tolerans artar
    t_sensor: sensör gecikmesi (s)
    """
    t_delay = 0.06 + 0.015 * a_ms2  # sistem gecikmesi, ivmeye bağlı
    t_sensor = 0.02 + 0.008 * min(a_ms2, 1.5)  # sensör gecikmesi
    delta_x = 0.5 * a_ms2 * (t_delay ** 2) + v_ms * t_sensor
    return delta_x, t_delay, t_sensor


# ---------------------------------------------------------------------------
# Simülasyon verileri (sayısal analiz)
# ---------------------------------------------------------------------------

def run_motion_simulation(
    v_nominal: float,
    a_nominal: float,
    t_end: float = 12.0,
    n_points: int = 300,
    brake_decel: float | None = None,
):
    """
    Hareket simülasyonu: ivme → hız → konum.
    Hedef durma noktası ve gerçekleşen durma noktası, gerçekçi formül Δx ile hesaplanır.
    """
    if brake_decel is None:
        brake_decel = -min(a_nominal * 1.2, 2.5)
    t = np.linspace(0, t_end, n_points)
    dt = t[1] - t[0]

    # Faz 1: ivmelenme, faz 2: sabit hız, faz 3: fren
    t_ramp = 1.5
    t_brake_start = t_end - 2.0
    accel = np.zeros_like(t)
    accel[t < t_ramp] = a_nominal * t[t < t_ramp] / t_ramp
    accel[(t >= t_ramp) & (t < t_brake_start)] = a_nominal
    accel[t >= t_brake_start] = brake_decel

    velocity = np.zeros_like(t)
    position = np.zeros_like(t)
    for i in range(1, len(t)):
        velocity[i] = velocity[i - 1] + accel[i - 1] * dt
        velocity[i] = max(0, min(v_nominal, velocity[i]))
        position[i] = position[i - 1] + velocity[i - 1] * dt

    target_stop_position = position[-1]
    sapma_miktarı, t_delay, t_sensor = compute_stopping_deviation(v_nominal, a_nominal)
    actual_stop_position = target_stop_position + sapma_miktarı
    stop_error = sapma_miktarı

    # Konum sapması zaman serisi: formüle dayalı birikimli etki (t_delay/t_sensor ile orantılı)
    position_deviation = np.zeros_like(t)
    for i in range(1, len(t)):
        if t[i] <= t_ramp:
            a_eff = a_nominal * t[i] / t_ramp
        elif t[i] <= t_brake_start:
            a_eff = a_nominal
        else:
            a_eff = brake_decel
        v_eff = velocity[i]
        position_deviation[i] = 0.5 * max(0, a_eff) * (t_delay ** 2) + v_eff * t_sensor
    velocity_deviation = np.gradient(position_deviation, dt)

    return {
        "t": t,
        "accel": accel,
        "velocity": velocity,
        "position": position,
        "velocity_deviation": velocity_deviation,
        "position_deviation": position_deviation,
        "target_stop": target_stop_position,
        "actual_stop": actual_stop_position,
        "stop_error": stop_error,
        "sapma_miktarı": sapma_miktarı,
        "t_delay": t_delay,
        "t_sensor": t_sensor,
    }


def get_load_sweep_data(n_ropes: int, d_mm: float, a_nominal: float, m_min: float = 200, m_max: float = 2000):
    """Değişen yük m ile halat gerilmesi ve emniyet katsayısı."""
    m_range = np.linspace(m_min, m_max, 80)
    tension = np.array([compute_dynamic_force(m, a_nominal) for m in m_range])
    F_break = compute_rope_breaking_total(n_ropes, d_mm)
    safety = F_break / tension
    return m_range, tension, safety


# ---------------------------------------------------------------------------
# Plotly interaktif grafikler
# ---------------------------------------------------------------------------

def plot_elevator_schematic(m_kg: float, a_ms2: float, n_ropes: int, d_mm: float, v_ms: float):
    """2D asansör şeması - Plotly shapes + hover."""
    F_dyn = compute_dynamic_force(m_kg, a_ms2)
    sf = compute_safety_factor(m_kg, a_ms2, n_ropes, d_mm)
    cabin_y = 2.0 + min(v_ms * 2.2, 7.0)

    fig = go.Figure()
    # Şaft
    fig.add_shape(type="rect", x0=2.5, y0=0.5, x1=7.5, y1=11.5, line=dict(color="gray", width=2), fillcolor="rgba(240,240,240,0.5)")
    # Kabin
    fig.add_shape(type="rect", x0=3, y0=cabin_y, x1=7, y1=cabin_y + 1.8, line=dict(color="black", width=1.5), fillcolor="steelblue")
    # Halat çizgileri
    for i in range(min(n_ropes, 6)):
        xi = 3.2 + (i % 3) * 1.2
        fig.add_trace(go.Scatter(x=[xi, 3.5 + (i % 3) * 1.0], y=[11.2, cabin_y + 1.8], mode="lines", line=dict(color="black", width=max(1, d_mm / 6)), hoverinfo="skip"))

    fig.add_annotation(x=5, y=cabin_y + 0.9, text="Kabin", showarrow=False, font=dict(size=12, color="white"))
    fig.add_annotation(x=5, y=11.7, text="Şaft", showarrow=False)
    fig.add_annotation(x=0.8, y=6, text=f"m = {m_kg:.0f} kg<br>a = {a_ms2:.2f} m/s²<br>F = m(g+a) = {F_dyn:.0f} N<br>Halat: {n_ropes}×Ø{d_mm:.0f} mm<br>Emniyet = {sf:.1f}", showarrow=False, font=dict(size=10), bgcolor="wheat", bordercolor="gray")

    fig.update_layout(
        title="Asansör 2D Şematik (interaktif)",
        xaxis=dict(range=[0, 10], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, 12], scaleanchor="x", showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        height=380,
        margin=dict(l=60, r=40, t=50, b=40),
        hovermode="closest",
    )
    return fig


def plot_acceleration_chart(sim: dict):
    """İvme grafiği: zamana bağlı hız ve konum sapması; hover ile gerçek sayısal değerler."""
    t, accel, vel, pos_dev = sim["t"], sim["accel"], sim["velocity"], sim["position_deviation"]
    t_delay, t_sensor = sim["t_delay"], sim["t_sensor"]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("İvme ve Hız (m/s, m/s²)", "Konum Sapması (m)"))

    fig.add_trace(
        go.Scatter(
            x=t, y=accel, name="İvme (m/s²)", line=dict(color="royalblue", width=2),
            hovertemplate="Zaman: %{x:.3f} s<br>İvme: %{y:.4f} m/s²<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t, y=vel, name="Hız (m/s)", line=dict(color="darkgreen", width=2),
            hovertemplate="Zaman: %{x:.3f} s<br>Hız: %{y:.4f} m/s<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t, y=pos_dev, name="Konum sapması", line=dict(color="coral", width=2),
            hovertemplate="Zaman: %{x:.3f} s<br>Sapma: %{y:.5f} m<br>t_delay: " + f"{t_delay:.3f} s<br>t_sensor: {t_sensor:.3f} s<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_layout(height=400, title="İvme Grafiği – Hız ve Konum Sapması", hovermode="x unified", margin=dict(t=60))
    fig.update_yaxes(title_text="İvme / Hız", row=1, col=1)
    fig.update_yaxes(title_text="Konum sapması (m)", row=2, col=1)
    fig.update_xaxes(title_text="Zaman (s)", row=2, col=1)
    return fig


def plot_load_chart(m_range: np.ndarray, tension: np.ndarray, safety: np.ndarray, n_ropes: int, d_mm: float):
    """Yük grafiği: yüke göre halat gerilmesi ve emniyet katsayısı; hover ile gerçek değerler."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=m_range, y=tension / 1000, name="Halat gerilmesi (kN)", line=dict(color="blue", width=2),
            hovertemplate="Yük: %{x:.2f} kg<br>Gerilme: %{y:.3f} kN<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=m_range, y=safety, name="Emniyet katsayısı", line=dict(color="green", width=2),
            hovertemplate="Yük: %{x:.2f} kg<br>Emniyet katsayısı: %{y:.3f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_hline(y=MIN_SAFETY_FACTOR, line_dash="dash", line_color="red", annotation_text="Min SF=12", secondary_y=True)
    fig.update_layout(
        title=f"Yük Grafiği – Halat Gerilmesi ve Emniyet (Halat: {n_ropes}×Ø{d_mm:.0f} mm)",
        height=360,
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Yük (kg)")
    fig.update_yaxes(title_text="Gerilme (kN)", secondary_y=False)
    fig.update_yaxes(title_text="Emniyet katsayısı", secondary_y=True)
    return fig


def plot_tension_only_chart(m_range: np.ndarray, tension: np.ndarray, n_ropes: int, d_mm: float):
    """Gerilme grafiği: Yük (kg) vs Halat gerilmesi (kN); hover ile sayısal değerler."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=m_range, y=tension / 1000, name="Halat gerilmesi (kN)", line=dict(color="darkblue", width=2),
            hovertemplate="Yük: %{x:.2f} kg<br>Gerilme: %{y:.3f} kN<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Gerilme Grafiği – Yük vs Halat Gerilmesi (Halat: {n_ropes}×Ø{d_mm:.0f} mm)",
        height=320,
        hovermode="x unified",
        xaxis_title="Yük (kg)",
        yaxis_title="Gerilme (kN)",
    )
    return fig


def plot_error_analysis_chart(sim: dict):
    """Hata analizi: Hedef vs Gerçekleşen durma; hover ile gerçek sayısal değerler."""
    t = sim["t"]
    target_stop = sim["target_stop"]
    actual_stop = sim["actual_stop"]
    stop_error = sim["stop_error"]
    position = sim["position"]
    pos_with_error = position + sim["position_deviation"]
    running_error = pos_with_error - position
    sapma = sim["sapma_miktarı"]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=("Hedef vs Gerçekleşen Konum (son 50 nokta)", "Durma Noktası Sapması Δx (m)"))

    n_show = 50
    t_tail = t[-n_show:]
    pos_tail = position[-n_show:]
    pos_err_tail = pos_with_error[-n_show:]

    fig.add_trace(
        go.Scatter(
            x=t_tail, y=pos_tail, name="Hedef konum", line=dict(color="blue", width=2),
            hovertemplate="Zaman: %{x:.4f} s<br>Hedef konum: %{y:.5f} m<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_tail, y=pos_err_tail, name="Gerçekleşen konum", line=dict(color="red", width=2),
            hovertemplate="Zaman: %{x:.4f} s<br>Gerçekleşen: %{y:.5f} m<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[t[-1]], y=[actual_stop], name="Gerçek durma", mode="markers",
            marker=dict(size=12, color="red", symbol="x"),
            hovertemplate="Gerçek durma: %{y:.5f} m<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[t[-1]], y=[target_stop], name="Hedef durma", mode="markers",
            marker=dict(size=12, color="blue", symbol="circle"),
            hovertemplate="Hedef durma: %{y:.5f} m<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t, y=running_error, name="Sapma Δx", line=dict(color="purple", width=1.5),
            hovertemplate="Zaman: %{x:.3f} s<br>Sapma: %{y:.5f} m<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_hline(
        y=stop_error, line_dash="dash", line_color="orange", row=2, col=1,
        annotation_text=f"Δx = ½·a·t_delay² + v·t_sensor = {sapma:.4f} m",
    )

    fig.update_layout(height=420, title="Hata Analizi – Hedef vs Gerçekleşen Durma Noktası", hovermode="x unified")
    fig.update_yaxes(title_text="Konum (m)", row=1, col=1)
    fig.update_yaxes(title_text="Sapma (m)", row=2, col=1)
    fig.update_xaxes(title_text="Zaman (s)", row=2, col=1)
    return fig


# ---------------------------------------------------------------------------
# Mühendislik motoru: optimal tasarım (en düşük hata, en yüksek güvenlik)
# ---------------------------------------------------------------------------

def compute_design_score(m_kg: float, a_ms2: float, v_ms: float, n_ropes: int, d_mm: float) -> tuple[float, float, float]:
    """
    Tasarım puanı: yüksek emniyet + düşük sapma tercih edilir.
    Returns: (safety_score 0-1, error_penalty 0-1, combined_score 0-1 yüksek iyi)
    """
    sf = compute_safety_factor(m_kg, a_ms2, n_ropes, d_mm)
    safety_score = min(1.0, sf / 20.0)
    if sf < MIN_SAFETY_FACTOR:
        safety_score = 0.0
    sapma_miktarı, _, _ = compute_stopping_deviation(v_ms, a_ms2)
    error_penalty = min(1.0, sapma_miktarı / 0.05)
    combined = 0.7 * safety_score + 0.3 * (1.0 - error_penalty)
    return safety_score, error_penalty, combined


def get_optimal_design(
    m_min: float = 300,
    m_max: float = 1200,
    v_range: tuple[float, float] = (0.8, 2.0),
    a_range: tuple[float, float] = (0.4, 1.0),
    n_ropes_options: list[int] = (4, 6, 8),
    d_mm_options: list[float] = (12, 13, 16, 20),
) -> dict:
    """Tüm parametreleri tarayıp en iyi tasarımı döndür (Tavsiye Edilen Tasarım)."""
    best = None
    best_score = -1.0
    m_vals = np.linspace(m_min, m_max, 6)
    v_vals = np.linspace(v_range[0], v_range[1], 4)
    a_vals = np.linspace(a_range[0], a_range[1], 4)
    for m in m_vals:
        for v in v_vals:
            for a in a_vals:
                for n in n_ropes_options:
                    for d in d_mm_options:
                        sf = compute_safety_factor(m, a, n, d)
                        if sf < MIN_SAFETY_FACTOR:
                            continue
                        safety_s, error_p, combined = compute_design_score(m, a, v, n, d)
                        if combined > best_score:
                            best_score = combined
                            best = {"m_kg": m, "v_ms": v, "a_ms2": a, "n_ropes": n, "d_mm": d, "safety_factor": sf, "score": combined}
    if best is None:
        best = {"m_kg": 800, "v_ms": 1.2, "a_ms2": 0.7, "n_ropes": 6, "d_mm": 16, "safety_factor": 14.0, "score": 0.5}
    return best


# ---------------------------------------------------------------------------
# AI sohbet ve cevap üretimi (veri odaklı: mevcut_emniyet_kat, sapma_miktarı, risk %)
# ---------------------------------------------------------------------------

def _risk_yuzde(mevcut_emniyet_kat: float) -> float:
    """Emniyet katsayısı 12'nin altındaysa risk yüzdesi (0–100)."""
    if mevcut_emniyet_kat >= MIN_SAFETY_FACTOR:
        return 0.0
    return max(0, (MIN_SAFETY_FACTOR - mevcut_emniyet_kat) / MIN_SAFETY_FACTOR * 100)


def _build_context_block(m_kg: float, v_ms: float, mevcut_emniyet_kat: float, sapma_m: float) -> str:
    """Her cevapta okunacak mevcut simülasyon verileri (Context)."""
    return (
        f"**Context (mevcut simülasyon):** "
        f"Yük = **{m_kg:.0f} kg**, Hız = **{v_ms:.2f} m/s**, "
        f"Emniyet Katsayısı = **{mevcut_emniyet_kat:.1f}**, Sapma Δx = **{sapma_m:.4f} m**."
    )


def _build_previous_vs_current_report(prev: dict, cur_m: float, cur_v: float, cur_sf: float, cur_sapma: float) -> str:
    """Önceki durumla şimdikini matematiksel olarak karşılaştır; % hata farkı."""
    if not prev or prev.get("safety_factor") is None:
        return "Önceki durum verisi yok; en az bir parametre değişikliği yaptıktan sonra kıyaslama yapılabilir."
    pm, pv = prev.get("m_kg"), prev.get("v_ms")
    psf, psa = prev.get("safety_factor"), prev.get("sapma")
    lines = ["**Önceki durumla kıyaslama (matematiksel):**"]
    if pm is not None and pm > 0:
        lines.append(f"- Yük: {pm:.0f} kg → {cur_m:.0f} kg (**%{(cur_m - pm) / pm * 100:+.1f}**).")
    if pv is not None and pv > 0:
        lines.append(f"- Hız: {pv:.2f} m/s → {cur_v:.2f} m/s (**%{(cur_v - pv) / pv * 100:+.1f}**).")
    if psf is not None and psf > 0:
        lines.append(f"- Emniyet katsayısı: {psf:.1f} → {cur_sf:.1f} (**%{(cur_sf - psf) / psf * 100:+.1f}**).")
    if psa is not None and psa > 0:
        lines.append(f"- Hata payı (Δx): {psa:.4f} m → {cur_sapma:.4f} m (**%{(cur_sapma - psa) / psa * 100:+.1f}**).")
    return "\n".join(lines)


def _build_domino_effect_warning() -> str:
    """Parametreler arası bağıntı: domino etkisi uyarısı."""
    return (
        "**⚠️ Domino etkisi uyarısı:** "
        "İvme değerini artırmak konforu (sapma Δx) doğrudan bozar; formülde Δx = ½·a·t_delay² + v·t_sensor. "
        "İvmeyi artırırsanız motor gücü ihtiyacı da F = m(g+a) ile artar. "
        "Hızı artırmak sapmayı ve frenleme mesafesini yükseltir. "
        "Yükü artırmak halat gerilimini ve emniyet katsayısı üzerindeki baskıyı artırır."
    )


def _build_scenarios_and_risks(
    m_kg: float,
    v_ms: float,
    a_nominal: float,
    n_ropes: int,
    d_mm: float,
    mevcut_sf: float,
    mevcut_sapma: float,
) -> str:
    """
    Tek 'İdeal Tasarım' yerine 'Seçenekler ve Riskler' tablosu.
    Senaryo A: Hız odaklı (sapmayı düşürmek için hızı düşür).
    Senaryo B: Donanım odaklı (hızı koru, halat sayısını artır).
    Her senaryo için AVANTAJ/DEZAVANTAJ ve mevcut değerlerle % fark (hata kıyaslaması).
    """
    # Senaryo A: hızı 1.0 m/s'ye çek (sapma düşsün)
    v_A = 1.0
    sapma_A, _, _ = compute_stopping_deviation(v_A, a_nominal)
    sf_A = mevcut_sf  # yük/ivme/halat aynı
    if mevcut_sapma > 1e-6:
        sapma_degisim_A = (sapma_A - mevcut_sapma) / mevcut_sapma * 100
    else:
        sapma_degisim_A = 0.0
    verimlilik_degisim_A = (v_A - v_ms) / v_ms * 100 if v_ms > 0 else 0  # hız düşünce verimlilik düşer

    # Senaryo B: halat sayısını 6'ya çıkar (hız aynı)
    n_B = 6
    sapma_B = mevcut_sapma  # hız/ivme aynı
    sf_B = compute_safety_factor(m_kg, a_nominal, n_B, d_mm)
    sf_degisim_B = (sf_B - mevcut_sf) / mevcut_sf * 100 if mevcut_sf > 0 else 0

    # Hata kıyaslaması tablosu: Mevcut | Sen A | Sen B | % Fark A | % Fark B
    def _pct(cur: float, new: float) -> str:
        if cur == 0:
            return "—"
        return f"%{(new - cur) / cur * 100:+.1f}"

    lines = [
        "## Seçenekler ve Riskler",
        "",
        "| Parametre | Mevcut | Senaryo A | Senaryo B | % Fark (A) | % Fark (B) |",
        "|------------|--------|-----------|-----------|------------|------------|",
        f"| Yük (kg) | {m_kg:.0f} | {m_kg:.0f} | {m_kg:.0f} | — | — |",
        f"| Hız (m/s) | {v_ms:.2f} | {v_A:.2f} | {v_ms:.2f} | {_pct(v_ms, v_A)} | — |",
        f"| Halat sayısı | {n_ropes} | {n_ropes} | {n_B} | — | {_pct(n_ropes, n_B)} |",
        f"| Emniyet katsayısı | {mevcut_sf:.1f} | {sf_A:.1f} | {sf_B:.1f} | — | {_pct(mevcut_sf, sf_B)} |",
        f"| Sapma Δx (m) | {mevcut_sapma:.4f} | {sapma_A:.4f} | {sapma_B:.4f} | {_pct(mevcut_sapma, sapma_A)} | — |",
        "",
        "---",
        "",
        "**Senaryo A (Hız odaklı):** Sapmayı düşürmek için hızı **1.0 m/s**'ye çekebilirsiniz. "
        f"**AVANTAJ:** Sapma yaklaşık **%{abs(sapma_degisim_A):.0f}** {'azalır' if sapma_degisim_A < 0 else 'artar'}. "
        f"**DEZAVANTAJ:** Taşıma kapasitesi (verimlilik) hızla orantılı düştüğü için yaklaşık **%{abs(verimlilik_degisim_A):.0f}** düşer.",
        "",
        "**Senaryo B (Maliyet/Donanım odaklı):** Hızı korumak istiyorsanız halat sayısını **6**'ya çıkarabilirsiniz. "
        f"**AVANTAJ:** Emniyet katsayısı yaklaşık **%{sf_degisim_B:.0f}** artar. "
        "**DEZAVANTAJ:** Kurulum ve bakım maliyeti (halat sayısı) artar.",
    ]
    return "\n".join(lines)


def get_ai_reply(
    user_message: str,
    m_kg: float,
    a_ms2: float,
    v_ms: float,
    n_ropes: int,
    d_mm: float,
    mevcut_emniyet_kat: float,
    sim: dict,
    comparison_text: str | None = None,
    prev_snapshot: dict | None = None,
) -> str:
    """Kullanıcı mesajına göre AI yanıtı; Context + veri odaklı; önceki durumla kıyaslama destekli."""
    msg = user_message.strip().lower()
    F_dyn = compute_dynamic_force(m_kg, a_ms2)
    F_break = compute_rope_breaking_total(n_ropes, d_mm)
    sapma_miktarı = sim["sapma_miktarı"]
    risk = _risk_yuzde(mevcut_emniyet_kat)

    # Veri bağlantısı: her cevapta mevcut simülasyon verilerini Context olarak oku
    context_block = _build_context_block(m_kg, v_ms, mevcut_emniyet_kat, sapma_miktarı)

    ideal_analysis = get_ideal_durum_analysis(mevcut_emniyet_kat, sapma_miktarı)
    data_intro = (
        f"Şu anki **{n_ropes}** adet halat ve **{d_mm:.0f}** mm çap ile emniyet katsayınız **{mevcut_emniyet_kat:.1f}**, "
        f"hata payı (sapma) **{sapma_miktarı:.4f} m**. "
    )
    if risk > 0:
        data_intro += f"Bu ivme ve yük için **%{risk:.1f}** oranında risk taşıyorsunuz (SF < 12). "
    else:
        data_intro += "Emniyet katsayısı yeterli aralıkta. "
    data_intro += "\n\n" + ideal_analysis
    if comparison_text:
        data_intro = comparison_text + "\n\n" + data_intro

    # Önceki durumla kıyasla: hafızadaki eski verilerle şimdikini matematiksel karşılaştır
    if "kıyasla" in msg or "önceki durum" in msg or "karşılaştır" in msg:
        report = _build_previous_vs_current_report(
            prev_snapshot or {}, m_kg, v_ms, mevcut_emniyet_kat, sapma_miktarı
        )
        return context_block + "\n\n" + report

    if "ideal" in msg or "en iyi" in msg or "optimum" in msg or "tavsiye" in msg or "öner" in msg or "seçenek" in msg or "senaryo" in msg:
        scenarios_block = _build_scenarios_and_risks(
            m_kg, v_ms, a_nominal, n_ropes, d_mm, mevcut_emniyet_kat, sapma_miktarı
        )
        domino_block = _build_domino_effect_warning()
        return (
            context_block + "\n\n" + data_intro
            + "\n\n" + domino_block
            + "\n\n" + scenarios_block
        )

    if "emniyet" in msg or "güvenlik" in msg or "safety" in msg or "katsayı" in msg:
        if mevcut_emniyet_kat < MIN_SAFETY_FACTOR:
            d_req = np.sqrt((MIN_SAFETY_FACTOR * F_dyn) / (n_ropes * (np.pi / 4) * ROPE_STRENGTH_N_PER_MM2))
            n_req = int(np.ceil((MIN_SAFETY_FACTOR * F_dyn) / get_rope_breaking_force_from_table(d_mm)))
            return (
                context_block + "\n\n" + data_intro
                + f"\n\nEmniyet katsayınız **{mevcut_emniyet_kat:.1f}** (minimum 12 olmalı). "
                f"Halat sayın olan **{n_ropes}** adet halat, bu ivme için **%{risk:.1f}** oranında risk taşıyor. "
                f"**Öneriler:** Halat çapını en az **{d_req:.0f} mm** yapın veya halat sayısını en az **{n_req}** yapın; "
                "veya yük/ivmeyi düşürün."
            )
        return context_block + "\n\n" + data_intro + "\n\nPeriyodik halat muayenesini sürdürün."

    if "hata" in msg or "sapma" in msg or "durma" in msg:
        err = sim["stop_error"]
        return (
            context_block + "\n\n" + data_intro
            + f"\n\nHedef durma: **{sim['target_stop']:.4f} m**, gerçekleşen: **{sim['actual_stop']:.4f} m**, "
            f"fark: **{err:.4f} m** (Δx = ½·a·t_delay² + v·t_sensor; t_delay={sim['t_delay']:.3f} s, t_sensor={sim['t_sensor']:.3f} s). "
            "Hassas frenleme ve sensör kalibrasyonu ile sapma azaltılabilir. "
            "\n\n" + _build_domino_effect_warning()
        )

    if "kuvvet" in msg or "gerilme" in msg or "f=" in msg:
        return (
            context_block + "\n\n" + data_intro
            + f"\n\n**F = m(g+a)** = {m_kg}×(9.81+{a_ms2}) = **{F_dyn:.0f} N**. "
            f"Toplam halat kopma: **{F_break:.0f} N**, emniyet katsayısı = **{mevcut_emniyet_kat:.1f}**."
        )

    return (
        context_block + "\n\n" + data_intro
        + "\n\n'Seçenekler ve riskler' (senaryo analizi), 'önceki durumla kıyasla' veya 'hata analizi' yazarak detay alabilirsiniz."
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def inject_scroll_css():
    """Orta ve sağ panelin bağımsız scroll; sağ panel özel chat odası (500px, kaydırılabilir)."""
    st.markdown("""
        <style>
        div[data-testid="column"] { box-sizing: border-box; }
        /* Orta sütun: mevcut kaydırma */
        div[data-testid="column"]:nth-of-type(2) {
            height: 82vh; min-height: 400px; max-height: 82vh;
            overflow-y: auto; overflow-x: hidden;
            display: flex; flex-direction: column;
        }
        div[data-testid="column"]:nth-of-type(2) > div { min-height: 0; }
        /* Sağ sütun: sabit yükseklikte chat paneli, kendi içinde kaydırılabilir */
        div[data-testid="column"]:nth-of-type(3) {
            height: 500px; min-height: 400px; max-height: 500px;
            overflow-y: auto; overflow-x: hidden;
            display: flex; flex-direction: column;
            border: 1px solid #e0e0e0; border-radius: 8px;
            padding: 8px; background: #fafafa;
        }
        div[data-testid="column"]:nth-of-type(3) > div { min-height: 0; }
        /* Chat mesaj alanı en alta kaydırma için (yeni mesajda aşağı) */
        .stChatMessage { margin-bottom: 0.5rem; }
        div[data-testid="column"]:nth-of-type(2)::-webkit-scrollbar,
        div[data-testid="column"]:nth-of-type(3)::-webkit-scrollbar { width: 8px; }
        div[data-testid="column"]:nth-of-type(2)::-webkit-scrollbar-track,
        div[data-testid="column"]:nth-of-type(3)::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 4px; }
        div[data-testid="column"]:nth-of-type(2)::-webkit-scrollbar-thumb,
        div[data-testid="column"]:nth-of-type(3)::-webkit-scrollbar-thumb { background: #888; border-radius: 4px; }
        </style>
    """, unsafe_allow_html=True)


def render_welcome_screen():
    """Giriş ekranı: proje açıklaması, AI modül seçimi, Projeyi Başlat."""
    st.title("Hoş geldiniz")
    st.markdown("Yapmak istediğiniz projenin **ne hakkında** olması gerektiğini yazar mısınız?")
    project = st.text_area(
        "Proje açıklaması",
        value=st.session_state.get("project_description", ""),
        placeholder="Örn: 8 katlı bir binada yolcu taşıyacak dikey asansör tasarımı...",
        height=120,
        key="welcome_project",
    )
    st.session_state.project_description = project

    if st.button("Analiz Et", key="analyze_btn"):
        module_key, module_name = analyze_project_description(project)
        st.session_state.detected_module_key = module_key
        st.session_state.detected_module_name = module_name
        st.session_state.analysis_done = True
        st.rerun()

    if st.session_state.get("analysis_done") and st.session_state.get("project_description"):
        module_name = st.session_state.get("detected_module_name", "Asansör")
        st.success(f"Projeniz **{module_name}** mühendislik modülüne uygun görülüyor.")
        if st.button("Projeyi Başlat", type="primary", key="start_project_btn"):
            st.session_state.page = PAGE_DASHBOARD
            st.session_state.analysis_done = False
            st.rerun()


def render_dashboard():
    """Ana çalışma ekranı: 3 bölme, History, gerçek zamanlı kıyaslama, 4 grafik, İdeal Durum."""
    # Akıllı navigasyon: üstte her zaman Giriş Ekranına Dön
    if st.button("← Giriş Ekranına Dön", key="back_to_welcome"):
        st.session_state.page = PAGE_WELCOME
        st.rerun()

    inject_scroll_css()

    if "history" not in st.session_state:
        st.session_state.history = []
    if "previous_sapma" not in st.session_state:
        st.session_state.previous_m_kg = None
        st.session_state.previous_v_ms = None
        st.session_state.previous_a_nominal = None
        st.session_state.previous_n_ropes = None
        st.session_state.previous_d_mm = None
        st.session_state.previous_sapma = None
        st.session_state.previous_F_dyn = None
        st.session_state.previous_safety_factor = None
    # Action-Oriented AI: parametreler session_state'te; AI bunları güncelleyebilir
    if "params_m_kg" not in st.session_state:
        st.session_state.params_m_kg = 800
        st.session_state.params_v_ms = 1.6
        st.session_state.params_a_nominal = 0.8
        st.session_state.params_n_ropes = 4
        st.session_state.params_d_mm = 13
    if "last_action" not in st.session_state:
        st.session_state.last_action = None
    if "pending_params" not in st.session_state:
        st.session_state.pending_params = None
        st.session_state.pending_warning = None

    st.title("Mühendislik Analiz Dashboard")
    st.caption("Sol: Parametreler ve Geçmiş | Orta: Şema ve 4 interaktif grafik | Sağ: AI (komutlarla değişken kontrolü)")

    col_left, col_center, col_right = st.columns([1.0, 1.9, 1.2])

    with col_left:
        st.subheader("Parametre Girişleri")
        m_kg = st.slider(
            "Kabin + Yük Kütlesi (kg)", 200, 2000,
            value=int(st.session_state.params_m_kg), step=50,
        )
        v_ms = st.slider(
            "Nominal Hız (m/s)", 0.25, 4.0,
            value=float(st.session_state.params_v_ms), step=0.05,
        )
        a_nominal = st.slider(
            "İvme (m/s²)", 0.2, 2.0,
            value=float(st.session_state.params_a_nominal), step=0.05,
        )
        n_ropes = st.slider(
            "Halat Sayısı", 2, 12,
            value=int(st.session_state.params_n_ropes), step=1,
        )
        d_mm = st.slider(
            "Halat Çapı (mm)", 8, 24,
            value=int(st.session_state.params_d_mm), step=1,
        )
        # Kullanıcı slider ile değiştirdiyse session_state ile senkronize et
        st.session_state.params_m_kg = m_kg
        st.session_state.params_v_ms = v_ms
        st.session_state.params_a_nominal = a_nominal
        st.session_state.params_n_ropes = n_ropes
        st.session_state.params_d_mm = d_mm

        F_dyn = compute_dynamic_force(m_kg, a_nominal)
        F_break = compute_rope_breaking_total(n_ropes, d_mm)
        mevcut_emniyet_kat = compute_safety_factor(m_kg, a_nominal, n_ropes, d_mm)
        current_sapma, _, _ = compute_stopping_deviation(v_ms, a_nominal)

        st.markdown("---")
        st.markdown("**Hesaplanan değerler**")
        st.metric("F = m(g+a)", f"{F_dyn:.0f} N")
        st.metric("Halat kopma toplam", f"{F_break:.0f} N")
        st.metric("Emniyet katsayısı", f"{mevcut_emniyet_kat:.1f}")
        st.metric("Hata payı Δx (m)", f"{current_sapma:.4f}")
        if mevcut_emniyet_kat < MIN_SAFETY_FACTOR:
            st.error("Emniyet katsayısı 12'nin altında!")

        # Hafıza: History listesi – her parametre değişimini kaydet
        current_snapshot = {
            "m_kg": m_kg, "v_ms": v_ms, "a_nominal": a_nominal,
            "n_ropes": n_ropes, "d_mm": d_mm,
            "F_dyn": F_dyn, "safety_factor": mevcut_emniyet_kat, "sapma": current_sapma,
        }
        last = st.session_state.history[-1] if st.session_state.history else None
        if last is None or (
            last["m_kg"] != m_kg or last["v_ms"] != v_ms or last["a_nominal"] != a_nominal
            or last["n_ropes"] != n_ropes or last["d_mm"] != d_mm
        ):
            st.session_state.history.append(current_snapshot)

        with st.expander("📜 Geçmiş (History)", expanded=False):
            for i, h in enumerate(reversed(st.session_state.history[-20:]), 1):
                st.markdown(
                    f"**{len(st.session_state.history) - i + 1}.** "
                    f"m={h['m_kg']:.0f} kg, v={h['v_ms']:.2f} m/s, a={h['a_nominal']:.2f}, "
                    f"SF={h['safety_factor']:.1f}, Δx={h['sapma']:.4f} m"
                )

    # Gerçek zamanlı kıyaslama: önceki ayar vs şimdiki
    prev_for_compare = st.session_state.history[-2] if len(st.session_state.history) >= 2 else None
    comparison_text = build_realtime_comparison(
        prev_for_compare, m_kg, v_ms, a_nominal, n_ropes, d_mm, F_dyn, mevcut_emniyet_kat, current_sapma
    )

    with col_center:
        st.subheader("2D Şematik")
        fig_sch = plot_elevator_schematic(m_kg, a_nominal, n_ropes, d_mm, v_ms)
        st.plotly_chart(fig_sch, use_container_width=True)

        sim = run_motion_simulation(v_ms, a_nominal)
        m_range, tension, safety = get_load_sweep_data(n_ropes, d_mm, a_nominal)

        st.subheader("1) İvme Grafiği")
        st.plotly_chart(plot_acceleration_chart(sim), use_container_width=True)

        st.subheader("2) Yük Grafiği – Emniyet Katsayısı")
        st.plotly_chart(plot_load_chart(m_range, tension, safety, n_ropes, d_mm), use_container_width=True)

        st.subheader("3) Gerilme Grafiği")
        st.plotly_chart(plot_tension_only_chart(m_range, tension, n_ropes, d_mm), use_container_width=True)

        st.subheader("4) Hata Payı Grafiği – Hedef vs Gerçek Durma")
        st.plotly_chart(plot_error_analysis_chart(sim), use_container_width=True)

    with col_right:
        st.subheader("AI Mühendislik Danışmanı (Action-Oriented)")
        st.caption("Kalıcı hafıza: Konuşma geçmişi sayfa yenilense de korunur (st.session_state).")
        if comparison_text:
            st.info(comparison_text)

        # Bekleyen onay varsa göster (Numeric Validation)
        if st.session_state.pending_params:
            st.warning(
                st.session_state.pending_warning or "Bu değişiklik risk içeriyor. Onaylıyor musunuz?"
            )
            col_ok, col_no = st.columns(2)
            with col_ok:
                if st.button("Onayla", key="confirm_pending"):
                    for k, v in st.session_state.pending_params.items():
                        setattr(st.session_state, f"params_{k}", v)
                    st.session_state.pending_params = None
                    st.session_state.pending_warning = None
                    st.rerun()
            with col_no:
                if st.button("İptal", key="cancel_pending"):
                    st.session_state.pending_params = None
                    st.session_state.pending_warning = None
                    st.rerun()

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Özel kaydırma alanı: mesajlar st.chat_message ile; kimin ne dediği net
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input alanı panelin en altında sabit
        if prompt := st.chat_input("Komut veya soru (örn: yükü 1000 kg yap, önceki durumla kıyasla)"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            prompt_lower = prompt.strip().lower()
            reply = None
            handled = False

            # Onayla / İptal yanıtı (Numeric Validation sonrası)
            if st.session_state.pending_params:
                if any(x in prompt_lower for x in ("onayla", "evet", "tamam", "kabul")):
                    for k, v in st.session_state.pending_params.items():
                        setattr(st.session_state, f"params_{k}", v)
                    prev_snap = st.session_state.history[-1] if st.session_state.history else None
                    report = format_error_report(
                        prev_snap,
                        st.session_state.params_m_kg,
                        st.session_state.params_v_ms,
                        st.session_state.params_a_nominal,
                        st.session_state.params_n_ropes,
                        st.session_state.params_d_mm,
                    )
                    reply = "Değişiklik uygulandı.\n\n" + report
                    st.session_state.pending_params = None
                    st.session_state.pending_warning = None
                elif any(x in prompt_lower for x in ("iptal", "hayır", "vazgeç")):
                    reply = "İptal edildi. Parametreler değiştirilmedi."
                    st.session_state.pending_params = None
                    st.session_state.pending_warning = None
                else:
                    reply = get_ai_reply(prompt, m_kg, a_nominal, v_ms, n_ropes, d_mm, mevcut_emniyet_kat, sim, comparison_text=comparison_text, prev_snapshot=prev_for_compare)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                handled = True

            if not handled:
                # Komut ayrıştır (Değişken Kontrolü + Hafıza)
                current_params = {"m_kg": m_kg, "v_ms": v_ms, "a_nominal": a_nominal, "n_ropes": n_ropes, "d_mm": d_mm}
                updates, action_reply, needs_confirm, warning = parse_action_command(
                    prompt, m_kg, v_ms, a_nominal, n_ropes, d_mm, st.session_state.last_action
                )

                if updates:
                    proposed = {**current_params, **updates}
                    is_risky, warn_msg = validate_proposed_params(current_params, proposed)
                    if is_risky:
                        st.session_state.pending_params = proposed
                        st.session_state.pending_warning = (
                            action_reply + " Ancak bu değişiklik şunları yapacak:" + warn_msg + " **Onaylıyor musun?**"
                        )
                        reply = st.session_state.pending_warning
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                    else:
                        for k, v in updates.items():
                            setattr(st.session_state, f"params_{k}", v)
                        action_type = "increase" if any(x in prompt_lower for x in ("artır", "yükselt")) else "decrease" if any(x in prompt_lower for x in ("düşür", "azalt")) else "set"
                        st.session_state.last_action = {"param": list(updates.keys())[0], "action": action_type, "value": list(updates.values())[0]}
                        prev_snap = st.session_state.history[-1] if st.session_state.history else None
                        report = format_error_report(
                            prev_snap,
                            st.session_state.params_m_kg,
                            st.session_state.params_v_ms,
                            st.session_state.params_a_nominal,
                            st.session_state.params_n_ropes,
                            st.session_state.params_d_mm,
                        )
                        reply = action_reply + "\n\n" + report
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                else:
                    reply = get_ai_reply(
                        prompt, m_kg, a_nominal, v_ms, n_ropes, d_mm, mevcut_emniyet_kat, sim,
                        comparison_text=comparison_text, prev_snapshot=prev_for_compare,
                    )
                    if action_reply:
                        reply = action_reply + "\n\n" + reply
                    st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()

        if st.session_state.messages and st.button("Sohbeti temizle", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

    # Bir sonraki kıyaslama için önceki değerleri güncelle
    st.session_state.previous_m_kg = m_kg
    st.session_state.previous_v_ms = v_ms
    st.session_state.previous_a_nominal = a_nominal
    st.session_state.previous_n_ropes = n_ropes
    st.session_state.previous_d_mm = d_mm
    st.session_state.previous_sapma = current_sapma
    st.session_state.previous_F_dyn = F_dyn
    st.session_state.previous_safety_factor = mevcut_emniyet_kat


def main():
    st.set_page_config(page_title="Mühendislik Analiz", layout="wide", initial_sidebar_state="collapsed")
    if "page" not in st.session_state:
        st.session_state.page = PAGE_WELCOME
    if st.session_state.page == PAGE_WELCOME:
        render_welcome_screen()
    else:
        render_dashboard()


if __name__ == "__main__":
    main()
