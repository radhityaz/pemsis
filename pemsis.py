#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blood Bank — Minimax Loss (CN statis) | OR-Tools CP-SAT
Interactive CLI + tqdm + full visualization (CSV & PNG)

Install:
  pip install ortools numpy pandas matplotlib tqdm

Run:
  python pemsis.py
"""

import os, sys, time, math, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from ortools.sat.python import cp_model

# ============== INPUT HELPERS ==============

def ask_int(msg, default=None, minv=None, maxv=None, hint=None):
    if hint: print(hint)
    while True:
        s = input(f"{msg}" + (f" [default {default}]: " if default is not None else ": "))
        if s.strip()=="" and default is not None:
            val = default
        else:
            try: val = int(s)
            except: print("Masukkan bilangan bulat."); continue
        if minv is not None and val < minv: print(f"Minimal {minv}."); continue
        if maxv is not None and val > maxv: print(f"Maksimal {maxv}."); continue
        return val

def ask_float(msg, default=None, minv=None, maxv=None, hint=None):
    if hint: print(hint)
    while True:
        s = input(f"{msg}" + (f" [default {default}]: " if default is not None else ": "))
        if s.strip()=="" and default is not None:
            val = default
        else:
            try: val = float(s)
            except: print("Masukkan angka desimal."); continue
        if minv is not None and val < minv: print(f"Minimal {minv}."); continue
        if maxv is not None and val > maxv: print(f"Maksimal {maxv}."); continue
        return val

def ask_choice(msg, choices, default=None, hint=None):
    if hint: print(hint)
    idxs = ", ".join([f"{i}:{c}" for i,c in enumerate(choices)])
    while True:
        s = input(f"{msg} ({idxs})" + (f" [default {default}]: " if default is not None else ": "))
        if s.strip()=="" and default is not None:
            i = default
        else:
            try: i = int(s)
            except: print("Pilih indeks."); continue
        if 0 <= i < len(choices): return choices[i]
        print("Indeks tidak valid.")

def ensure_dir(p): os.makedirs(p, exist_ok=True)

# ============== SUGGESTION LABELS & CHEATS ==============

def sigma_label(s):
    # std noise multiplikatif (≈ tingkat dispersi)
    if s < 0.08:    return "calm (rendah)"
    if s < 0.16:    return "normal (sedang)"
    if s <= 0.30:   return "wild (tinggi)"
    return "extreme (sangat tinggi)"

def rate_label(pct):
    # mean demand % kapasitas/hari
    if pct < 8:     return "rendah (hati-hati SL)"
    if pct <= 12:   return "basis (umum)"
    if pct <= 15:   return "tinggi (agresif)"
    return "surge (ekstrem)"

def amp_label(mult):
    if mult <= 1.15: return "mild"
    if mult <= 1.35: return "normal"
    if mult <= 1.60: return "strong"
    return "extreme"

def osc_amp_label(a):
    if a <= 0.10: return "mild (±10%)"
    if a <= 0.20: return "normal (±20%)"
    if a <= 0.35: return "wild (±35%)"
    return "extreme"

def dip_label(mult):
    if mult >= 0.85: return "mild"
    if mult >= 0.70: return "normal"
    if mult >= 0.50: return "strong"
    return "extreme"

def print_cheats():
    print("""
=== Cheat-sheet singkat (bantu isi input) ===
• Mean demand rate (% kapasitas/hari):
  <8%  : rendah (hati-hati SL) | 8–12%: basis (umum) | 12–15%: tinggi | >15%: surge
• Sigma dispersi (std noise multiplikatif):
  0.00–0.07: calm | 0.08–0.15: normal | 0.16–0.30: wild | >0.30: extreme
• Motif multiplier (>1):
  1.10–1.20: mild | 1.25–1.35: normal | 1.40–1.60: strong | >1.60: extreme
• Oscillation amplitude:
  0.10: ±10% (mild) | 0.20: ±20% (normal) | 0.35: ±35% (wild)
• Supply dip multiplier (<1):
  0.90: mild | 0.70: normal | 0.50: strong | <0.50: extreme
""")

# ============== SCENARIO GENERATION ==============

def gen_series(mu, T, sigma):
    noise = np.random.normal(1.0, sigma, T)
    noise = np.clip(noise, 0.1, 3.0)
    lam = np.maximum(np.round(mu * noise), 0).astype(int)
    return np.random.poisson(lam).astype(int)

def motif_apply(arr, motif, params):
    x = arr.copy()
    T = len(x)

    def clamp(i): return max(0, min(T, int(i)))

    if motif == "none": return x

    if motif in ("pulse","level_shift"):
        t0 = clamp(params.get("t0", T//3))
        span = clamp(params.get("span", 3))
        amp  = params.get("amp", 1.3)
        x[t0:t0+span] = (x[t0:t0+span]*amp).astype(int)

    elif motif == "ramp":
        t0 = clamp(params.get("t0", T//3))
        span = clamp(params.get("span", 7))
        amp_end = params.get("amp_end", 1.3)
        for k in range(span):
            if t0+k < T:
                w = 1.0 + (amp_end-1.0)*(k/max(1,span-1))
                x[t0+k] = int(round(x[t0+k]*w))

    elif motif == "oscillation":
        period = max(2, params.get("period", 7))
        amp = params.get("amp", 0.2)
        phase = params.get("phase", 0.0)
        t = np.arange(T)
        osc = 1.0 + amp*np.sin(2*np.pi*(t/period + phase))
        x = (x*osc).astype(int)

    elif motif == "supply_dip":
        t0 = clamp(params.get("t0", T//2))
        span = clamp(params.get("span", 5))
        dip = params.get("dip", 0.7)
        x[t0:t0+span] = (x[t0:t0+span]*dip).astype(int)

    return x

# ============== CP-SAT MODEL (MINIMAX LOSS) ==============

def build_and_solve(CAPACITY_MAX, T, AGE_EXPIRY, AGE_CN, B0_TOTAL,
                    GRACE_DAYS, scenarios, time_limit_s=120, workers=8):
    model = cp_model.CpModel()

    CN = model.NewIntVar(0, CAPACITY_MAX, "CN")  # keputusan (labu)
    Ages = list(range(0, AGE_EXPIRY))  # 0..(AGE_EXPIRY-1)
    last_age = AGE_EXPIRY - 1

    B, Use, Short, OutAge, InCN, ExcessCN = {}, {}, {}, {}, {}, {}

    # Stok awal: semua fresh (umur 0)
    B0_age = np.zeros(AGE_EXPIRY, dtype=int); B0_age[0] = B0_TOTAL
    counted = lambda t: t >= GRACE_DAYS

    for s_idx, sc in enumerate(scenarios):
        D, R = sc["D"], sc["R"]
        sumD = int(D.sum())
        max_daily = int(max(D.max(), R.max(), CAPACITY_MAX))

        for t in range(T):
            Short[(s_idx,t)]   = model.NewIntVar(0, sumD, f"Short_s{s_idx}_t{t}")
            OutAge[(s_idx,t)]  = model.NewIntVar(0, CAPACITY_MAX, f"OutAge_s{s_idx}_t{t}")
            InCN[(s_idx,t)]    = model.NewIntVar(0, CAPACITY_MAX, f"InCN_s{s_idx}_t{t}")
            ExcessCN[(s_idx,t)]= model.NewIntVar(0, CAPACITY_MAX, f"ExcessCN_s{s_idx}_t{t}")
            for a in Ages:
                B[(s_idx,a,t)]   = model.NewIntVar(0, CAPACITY_MAX, f"B_s{s_idx}_a{a}_t{t}")
                Use[(s_idx,a,t)] = model.NewIntVar(0, max_daily, f"Use_s{s_idx}_a{a}_t{t}")

        # Aging + Demand balance + Expiry + CN + Capacity
        for t in range(T):
            # age 0
            model.Add(B[(s_idx,0,t)] == R[t] - Use[(s_idx,0,t)])
            # age >=1
            if t == 0:
                for a in range(1, AGE_EXPIRY):
                    model.Add(B[(s_idx,a,0)] == B0_age[a-1] - Use[(s_idx,a,0)])
            else:
                for a in range(1, AGE_EXPIRY):
                    model.Add(B[(s_idx,a,t)] == B[(s_idx,a-1,t-1)] - Use[(s_idx,a,t)])

            # demand balance
            model.Add(sum(Use[(s_idx,a,t)] for a in Ages) + Short[(s_idx,t)] == D[t])

            # expiry (last age -> expiry)
            if t == 0:
                model.Add(OutAge[(s_idx,t)] >= 0 - Use[(s_idx,last_age,0)])
            else:
                model.Add(OutAge[(s_idx,t)] >= B[(s_idx,last_age,t-1)] - Use[(s_idx,last_age,t)])
            model.Add(OutAge[(s_idx,t)] >= 0)

            # pool CN & pembersihan
            model.Add(InCN[(s_idx,t)] == sum(B[(s_idx,a,t)] for a in range(AGE_CN, AGE_EXPIRY)))
            model.Add(ExcessCN[(s_idx,t)] >= InCN[(s_idx,t)] - CN)
            model.Add(ExcessCN[(s_idx,t)] >= 0)

            # kapasitas
            model.Add(sum(B[(s_idx,a,t)] for a in Ages) <= CAPACITY_MAX)

        # SL ≥ 97% (di hari yang dihitung)
        total_D_cnt = int(sum(D[t] for t in range(T) if counted(t)))
        if total_D_cnt > 0:
            model.Add(sum(Short[(s_idx,t)] for t in range(T) if counted(t)) <= int(round(0.03*total_D_cnt)))

    # Minimax loss: minimize T_epig dengan Z_s ≤ T_epig
    T_epig = model.NewIntVar(0, 10**9, "T_epig")
    for s_idx, sc in enumerate(scenarios):
        Z_terms = []
        for t in range(T):
            if counted(t): Z_terms += [Short[(s_idx,t)], OutAge[(s_idx,t)], ExcessCN[(s_idx,t)]]
        if Z_terms: model.Add(sum(Z_terms) <= T_epig)
    model.Minimize(T_epig)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = workers

    print("\n[Solving] tergantung jumlah skenario & horizon...")
    status = solver.Solve(model)
    ok = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    results = {"status": solver.StatusName(status), "CN": solver.Value(CN) if ok else None,
               "T_epig": solver.Value(T_epig) if ok else None, "per_scenario": []}
    if not ok: return results, None

    # Ekstraksi hasil (termasuk matriks stok-umur untuk heatmap)
    Ages = list(range(0, AGE_EXPIRY))
    all_Bcube = []  # list per-skenario: ndarray [age, t]
    for s_idx, sc in enumerate(tqdm(scenarios, desc="Extracting")):
        D, R = sc["D"], sc["R"]
        Short_list, Out_list, Exc_list, InCN_list, Stock_list, Use_tot = [], [], [], [], [], []
        Bcube = np.zeros((AGE_EXPIRY, T), dtype=int)

        for t in range(T):
            short_t = solver.Value(Short[(s_idx,t)])
            out_t   = solver.Value(OutAge[(s_idx,t)])
            exc_t   = solver.Value(ExcessCN[(s_idx,t)])
            incn_t  = solver.Value(InCN[(s_idx,t)])
            stock_t = 0
            use_t   = 0
            for a in Ages:
                bv = solver.Value(B[(s_idx,a,t)])
                uv = solver.Value(Use[(s_idx,a,t)])
                Bcube[a,t] = bv
                stock_t += bv
                use_t   += uv

            Short_list.append(short_t); Out_list.append(out_t); Exc_list.append(exc_t)
            InCN_list.append(incn_t);   Stock_list.append(stock_t); Use_tot.append(use_t)

        counted_idx = [t for t in range(T) if t >= GRACE_DAYS]
        D_cnt  = int(sum(D[t] for t in counted_idx)) if counted_idx else 0
        S_cnt  = int(sum(Short_list[t] for t in counted_idx)) if counted_idx else 0
        Out_cnt= int(sum(Out_list[t]   for t in counted_idx)) if counted_idx else 0
        Exc_cnt= int(sum(Exc_list[t]   for t in counted_idx)) if counted_idx else 0
        Waste_cnt = Out_cnt + Exc_cnt
        Zs = S_cnt + Waste_cnt
        SL = 1.0 if D_cnt==0 else 1 - (S_cnt / D_cnt)

        df = pd.DataFrame({
            "t": np.arange(T),
            "Demand": D, "Supply": R, "Use_total": Use_tot,
            "Short": Short_list, "OutAge": Out_list, "ExcessCN": Exc_list,
            "Waste": (np.array(Out_list)+np.array(Exc_list)),
            "InCN": InCN_list, "TotalStock": Stock_list
        })

        results["per_scenario"].append({
            "name": sc["name"], "df": df,
            "SL": SL, "Z": Zs,
            "Short_sum": S_cnt, "Out_sum": Out_cnt,
            "Excess_sum": Exc_cnt, "Waste_sum": Waste_cnt
        })
        all_Bcube.append(Bcube)

    return results, all_Bcube

# ============== PLOTTING ==============

def plot_series(df, title, ycols, outpath):
    plt.figure()
    for c in ycols: plt.plot(df["t"], df[c], label=c)
    plt.xlabel("Day"); plt.ylabel("Units"); plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(outpath); plt.close()

def plot_heatmap_B(Bcube, outpath):
    plt.figure()
    plt.imshow(Bcube, aspect="auto", origin="lower")
    plt.colorbar(label="Units")
    plt.yticks(np.arange(Bcube.shape[0]), [f"a{a}" for a in range(Bcube.shape[0])])
    plt.xlabel("Day"); plt.ylabel("Age"); plt.title("Stock by Age (Heatmap)")
    plt.tight_layout(); plt.savefig(outpath); plt.close()

# ============== MAIN (INTERACTIVE) ==============

def main():
    print("=== Blood Bank — Minimax Loss (CN statis) | OR-Tools CP-SAT ===")
    np.random.seed(int(time.time()) % (2**32-1))
    print_cheats()

    # Global params
    CAPACITY_MAX = ask_int("Capacity_max (labu)?",
                           default=5000, minv=1,
                           hint="Hint: sesuaikan dengan kapasitas riil gudang kamu.")
    T = ask_int("Horizon T (hari)?",
                default=30, minv=7,
                hint="Hint: >30 hari memberi gambaran stabil; 30–45 umum dipakai.")
    AGE_EXPIRY = ask_int("Expired di hari ke (default 27)?",
                         default=27, minv=5,
                         hint="Hint: RBC sering ~42 hari; di studi ini pakai 27 sesuai model.")
    AGE_CN = ask_int("Checkpoint CN (umur hari, default 24)?",
                     default=24, minv=1, maxv=AGE_EXPIRY-1,
                     hint="Hint: pool CN = umur ≥ checkpoint; kebijakan pembersihan berlaku di sini.")
    B0_TOTAL = ask_int("Stok awal total (semua fresh, labu)?",
                       default=2500, minv=0, maxv=CAPACITY_MAX,
                       hint="Hint: 50% kapasitas (2.500 dari 5.000) sering jadi titik tengah aman start-up.")
    GRACE_DAYS = ask_int("Grace days untuk SL & objektif (0=none)?",
                         default=0, minv=0, maxv=T-1,
                         hint="Hint: kalau start benar-benar kosong, kasih 5–7 hari. Di sini boleh 0.")

    # Build scenarios interactively
    print("\n=== Bangun skenario (interaktif) ===")
    S = ask_int("Berapa jumlah skenario?",
                default=6, minv=1,
                hint="Hint: 6–12 skenario memberi keseimbangan coverage & waktu solve.")
    scenarios = []
    for i in tqdm(range(S), desc="Create scenarios"):
        print(f"\n-- Skenario #{i+1} --")
        r_pct = ask_float("Mean demand (% dari kapasitas per hari)?",
                          default=10.0, minv=0.1, maxv=100.0,
                          hint="Hint: 8–12% basis; 12–15% tinggi; >15% surge.")
        print(f"  → interpretasi rate: {rate_label(r_pct)}")
        muD = int(round((r_pct/100.0) * CAPACITY_MAX))

        sigmaD = ask_float("Sigma dispersi DEMAND (ex 0.10)?",
                           default=0.10, minv=0.0, maxv=1.0,
                           hint="Hint: 0.00–0.07 calm | 0.08–0.15 normal | 0.16–0.30 wild | >0.30 extreme")
        print(f"  → DEMAND sigma {sigmaD:.2f} = {sigma_label(sigmaD)}")

        sigmaR = ask_float("Sigma dispersi SUPPLY (ex 0.10)?",
                           default=0.10, minv=0.0, maxv=1.0,
                           hint="Hint: supply cenderung lebih tenang; 0.05–0.15 umum.")
        print(f"  → SUPPLY sigma {sigmaR:.2f} = {sigma_label(sigmaR)}")

        supply_bias_pct = ask_float("Supply bias vs demand (%), ex 0?",
                                    default=0.0, minv=-50.0, maxv=50.0,
                                    hint="Hint: + berarti supply > demand sepanjang rata-rata.")
        muR = int(round(muD * (1.0 + supply_bias_pct/100.0)))
        print(f"  → μR ≈ {muR} (bias {supply_bias_pct:+.1f}%)")

        motif_choices = ["none","pulse","level_shift","ramp","oscillation","supply_dip"]
        motifD = ask_choice("Motif untuk DEMAND", motif_choices, default=0,
                            hint="Hint: pulse/level_shift menaikkan lokal; ramp bertahap; oscillation musiman.")
        motifR = ask_choice("Motif untuk SUPPLY", motif_choices, default=0,
                            hint="Hint: supply_dip cocok untuk drop donor.")

        def motif_params(m, T):
            if m == "none": return {}
            if m in ("pulse","level_shift","ramp","supply_dip"):
                t0 = ask_int(f"  {m}: t0 (start day 0..{T-1})?",
                             default=T//3, minv=0, maxv=T-1,
                             hint="Hint: titik mulai motif.")
                span = ask_int(f"  {m}: span hari?",
                               default=5, minv=1, maxv=T,
                               hint="Hint: durasi dampak motif.")
                if m in ("pulse","level_shift"):
                    amp = ask_float(f"  {m}: multiplier (>1, ex 1.30)?",
                                    default=1.30, minv=1.0, maxv=5.0,
                                    hint="Hint: 1.15 mild | 1.30 normal | 1.50 strong | >1.60 extreme")
                    print(f"    → {amp_label(amp)}")
                    return {"t0":t0,"span":span,"amp":amp}
                elif m == "ramp":
                    amp_end = ask_float("  ramp: multiplier akhir (>1, ex 1.30)?",
                                        default=1.30, minv=1.0, maxv=5.0,
                                        hint="Hint: besaran kenaikan di akhir ramp.")
                    print(f"    → {amp_label(amp_end)} (akhir)")
                    return {"t0":t0,"span":span,"amp_end":amp_end}
                else: # supply_dip
                    dip = ask_float("  supply_dip: multiplier (<1, ex 0.70)?",
                                    default=0.70, minv=0.05, maxv=1.0,
                                    hint="Hint: 0.90 mild | 0.70 normal | 0.50 strong | <0.50 extreme")
                    print(f"    → {dip_label(dip)}")
                    return {"t0":t0,"span":span,"dip":dip}
            if m == "oscillation":
                period = ask_int("  oscillation: period (hari, ex 7)?",
                                 default=7, minv=2, maxv=T,
                                 hint="Hint: pola mingguan sering 7.")
                amp = ask_float("  oscillation: amplitude (0..1, ex 0.20)?",
                                default=0.20, minv=0.0, maxv=1.0,
                                hint="Hint: 0.10 mild | 0.20 normal | 0.35 wild")
                phase = ask_float("  oscillation: phase 0..1?",
                                  default=0.0, minv=0.0, maxv=1.0,
                                  hint="Hint: geser fase (awal osilasi).")
                print(f"    → {osc_amp_label(amp)}")
                return {"period":period,"amp":amp,"phase":phase}
            return {}

        paramsD = motif_params(motifD, T)
        paramsR = motif_params(motifR, T)

        D = gen_series(muD, T, sigmaD); D = motif_apply(D, motifD, paramsD)
        R = gen_series(muR, T, sigmaR); R = motif_apply(R, motifR, paramsR)

        scenarios.append({"name": f"S{i+1}_D{muD}_R{muR}_{motifD}_{motifR}", "D": D, "R": R})

    # Solve
    print("\n=== Menyusun & menyelesaikan model (minimax loss, CN statis) ===")
    time_limit = ask_int("Time limit solve (detik)?",
                         default=120, minv=5,
                         hint="Hint: tambah kalau skenario banyak/horizon panjang.")
    workers = ask_int("Jumlah worker (CPU threads)?",
                      default=8, minv=1,
                      hint="Hint: set sesuai core CPU kamu.")

    results, all_Bcube = build_and_solve(
        CAPACITY_MAX=CAPACITY_MAX, T=T, AGE_EXPIRY=AGE_EXPIRY, AGE_CN=AGE_CN,
        B0_TOTAL=B0_TOTAL, GRACE_DAYS=GRACE_DAYS, scenarios=scenarios,
        time_limit_s=time_limit, workers=workers
    )

    print("\n=== Hasil ===")
    print(f"Status: {results['status']}")
    if results["CN"] is None:
        print("Model infeasible atau time limit terlalu ketat. Longgarkan parameter atau tambah waktu solve.")
        sys.exit(0)
    print(f"CN optimal (labu): {results['CN']}")
    print(f"Worst-case Loss (Short+Waste): {results['T_epig']}")

    # Export & visualize
    out_dir_csv, out_dir_fig = "exports", "plots"
    ensure_dir(out_dir_csv); ensure_dir(out_dir_fig)

    # Summary CSV
    rows = []
    for it in results["per_scenario"]:
        rows.append({
            "scenario": it["name"], "SL": it["SL"], "Z": it["Z"],
            "Short_sum": it["Short_sum"], "Out_sum": it["Out_sum"],
            "Excess_sum": it["Excess_sum"], "Waste_sum": it["Waste_sum"]
        })
    df_sum = pd.DataFrame(rows).sort_values("Z", ascending=False)
    df_sum.to_csv(os.path.join(out_dir_csv, "summary.csv"), index=False)
    print("Ringkasan per skenario → exports/summary.csv")

    # Per-skenario CSV & plots (termasuk heatmap stok-umur)
    for idx, it in enumerate(tqdm(results["per_scenario"], desc="Saving CSV")):
        it["df"].to_csv(os.path.join(out_dir_csv, f"{it['name']}.csv"), index=False)

    for idx, it in enumerate(tqdm(results["per_scenario"], desc="Plotting")):
        name, df = it["name"], it["df"]
        plot_series(df, f"{name} — Demand vs Supply",
                    ["Demand","Supply"], os.path.join(out_dir_fig, f"{name}_D_vs_R.png"))
        plot_series(df, f"{name} — Short / Waste / OutAge / ExcessCN",
                    ["Short","Waste","OutAge","ExcessCN"], os.path.join(out_dir_fig, f"{name}_Short_Waste.png"))
        plot_series(df, f"{name} — InCN (Pool 24+)",
                    ["InCN"], os.path.join(out_dir_fig, f"{name}_InCN.png"))
        plot_series(df, f"{name} — Total Stock",
                    ["TotalStock"], os.path.join(out_dir_fig, f"{name}_TotalStock.png"))
        # Heatmap stok-umur
        plot_heatmap_B(all_Bcube[idx], os.path.join(out_dir_fig, f"{name}_Heatmap_StockByAge.png"))

    print(f"\nSelesai.\n- CSV: {out_dir_csv}\n- Plots: {out_dir_fig}\nTip: cek exports/summary.csv untuk lihat skenario terburuk (Z terbesar).")

if __name__ == "__main__":
    main()
