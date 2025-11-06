"""Streamlit interface for the PEMSIS blood bank minimax optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from pemsis import (
    amp_label,
    build_and_solve,
    dip_label,
    generate_scenario,
    osc_amp_label,
    rate_label,
    sigma_label,
)

st.set_page_config(page_title="PEMSIS Blood Bank Optimizer", layout="wide")

st.title("PEMSIS – Blood Bank Minimax Optimizer")
st.markdown(
    """
    Gunakan antarmuka ini untuk mengonfigurasi skenario demand/supply darah,
    menjalankan optimasi minimax dengan OR-Tools, dan memvisualisasikan hasilnya.
    Semua perhitungan dijalankan sepenuhnya di browser Streamlit Cloud.
    """
)


@dataclass
class ScenarioConfig:
    name: str
    demand_pct: float
    sigma_demand: float
    sigma_supply: float
    supply_bias_pct: float
    motif_demand: str
    motif_supply: str
    params_demand: Dict[str, float]
    params_supply: Dict[str, float]
    seed: Optional[int]


def parse_optional_int(label: str, value: str, errors: List[str]) -> Optional[int]:
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        errors.append(f"{label} harus berupa bilangan bulat.")
        return None


def render_motif_inputs(prefix: str, motif: str, horizon: int) -> Dict[str, float]:
    params: Dict[str, float] = {}
    span_default = min(5, horizon)
    if motif in ("pulse", "level_shift", "ramp", "supply_dip"):
        t0 = st.number_input(
            "t0 (hari mulai)",
            min_value=0,
            max_value=max(0, horizon - 1),
            value=min(horizon // 3, max(0, horizon - 1)),
            key=f"{prefix}_t0",
        )
        span = st.number_input(
            "Durasi (hari)",
            min_value=1,
            max_value=max(1, horizon),
            value=max(1, span_default),
            key=f"{prefix}_span",
        )
        params.update({"t0": int(t0), "span": int(span)})

        if motif in ("pulse", "level_shift"):
            amp = st.number_input(
                "Multiplier (>1)",
                min_value=1.0,
                max_value=5.0,
                value=1.3,
                step=0.05,
                key=f"{prefix}_amp",
            )
            st.caption(f"Intensitas: {amp_label(float(amp))}")
            params["amp"] = float(amp)
        elif motif == "ramp":
            amp_end = st.number_input(
                "Multiplier akhir (>1)",
                min_value=1.0,
                max_value=5.0,
                value=1.3,
                step=0.05,
                key=f"{prefix}_amp_end",
            )
            st.caption(f"Intensitas akhir: {amp_label(float(amp_end))}")
            params["amp_end"] = float(amp_end)
        else:  # supply_dip
            dip = st.number_input(
                "Multiplier (<1)",
                min_value=0.05,
                max_value=1.0,
                value=0.7,
                step=0.05,
                key=f"{prefix}_dip",
            )
            st.caption(f"Penurunan: {dip_label(float(dip))}")
            params["dip"] = float(dip)

    elif motif == "oscillation":
        period = st.number_input(
            "Periode (hari)",
            min_value=2,
            max_value=max(2, horizon),
            value=min(7, max(2, horizon)),
            key=f"{prefix}_period",
        )
        amp = st.number_input(
            "Amplitudo (0-1)",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            key=f"{prefix}_osc_amp",
        )
        st.caption(f"Variasi: {osc_amp_label(float(amp))}")
        phase = st.number_input(
            "Fase (0-1)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key=f"{prefix}_phase",
        )
        params.update({"period": int(period), "amp": float(amp), "phase": float(phase)})

    return params


with st.sidebar:
    st.header("Parameter Global")
    capacity = st.number_input("Capacity max (labu)", min_value=1, value=5000, step=100)
    horizon = st.number_input("Horizon T (hari)", min_value=7, max_value=120, value=30, step=1)
    expiry = st.number_input("Expired di hari ke", min_value=5, max_value=120, value=27, step=1)
    checkpoint = st.number_input(
        "Checkpoint CN (umur hari)",
        min_value=1,
        max_value=max(1, int(expiry) - 1),
        value=min(24, max(1, int(expiry) - 1)),
        step=1,
    )
    initial_stock = st.number_input(
        "Stok awal total (fresh)",
        min_value=0,
        max_value=int(capacity),
        value=min(2500, int(capacity)),
        step=50,
    )
    grace_days = st.number_input(
        "Grace days untuk SL",
        min_value=0,
        max_value=max(0, int(horizon) - 1),
        value=0,
        step=1,
    )
    time_limit = st.slider("Batas waktu solver (detik)", min_value=10, max_value=600, value=120, step=10)
    workers = st.slider("Jumlah worker parallel", min_value=1, max_value=32, value=8, step=1)
    scenario_count = st.number_input("Jumlah skenario", min_value=1, max_value=12, value=3, step=1)
    seed_input = st.text_input("Global random seed", value="")

    st.markdown("---")
    st.caption(
        "Cheat-sheet: rate <8% rendah, 8–12% basis, 12–15% tinggi, >15% surge.\n"
        "Sigma 0.00–0.07 calm, 0.08–0.15 normal, 0.16–0.30 wild, >0.30 extreme."
    )

errors: List[str] = []
global_seed = parse_optional_int("Global random seed", seed_input, errors)

scenario_configs: List[ScenarioConfig] = []

motif_options = {
    "none": "None",
    "pulse": "Pulse",
    "level_shift": "Level shift",
    "ramp": "Ramp",
    "oscillation": "Oscillation",
    "supply_dip": "Supply dip",
}

for idx in range(int(scenario_count)):
    with st.expander(f"Skenario {idx + 1}", expanded=idx == 0):
        name = st.text_input("Nama skenario", value=f"Scenario {idx + 1}", key=f"name_{idx}")
        demand_pct = st.number_input(
            "% Demand vs kapasitas",
            min_value=0.1,
            max_value=100.0,
            value=10.0,
            step=0.5,
            key=f"demand_pct_{idx}",
        )
        st.caption(f"Interpretasi rate: {rate_label(float(demand_pct))}")

        sigma_demand = st.number_input(
            "Sigma demand",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.01,
            key=f"sigma_d_{idx}",
        )
        st.caption(f"Dispersi demand: {sigma_label(float(sigma_demand))}")

        sigma_supply = st.number_input(
            "Sigma supply",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.01,
            key=f"sigma_r_{idx}",
        )
        st.caption(f"Dispersi supply: {sigma_label(float(sigma_supply))}")

        supply_bias = st.number_input(
            "Bias supply vs demand (%)",
            min_value=-50.0,
            max_value=50.0,
            value=0.0,
            step=1.0,
            key=f"bias_{idx}",
        )

        motif_demand = st.selectbox(
            "Motif demand",
            options=list(motif_options.keys()),
            format_func=lambda k: motif_options[k],
            key=f"motif_d_{idx}",
        )
        params_demand = render_motif_inputs(f"motif_d_{idx}", motif_demand, int(horizon))

        motif_supply = st.selectbox(
            "Motif supply",
            options=list(motif_options.keys()),
            format_func=lambda k: motif_options[k],
            key=f"motif_s_{idx}",
        )
        params_supply = render_motif_inputs(f"motif_s_{idx}", motif_supply, int(horizon))

        seed_text = st.text_input("Seed skenario (opsional)", value="", key=f"seed_{idx}")
        scenario_seed = parse_optional_int(f"Seed skenario #{idx + 1}", seed_text, errors)

        scenario_configs.append(
            ScenarioConfig(
                name=name,
                demand_pct=float(demand_pct),
                sigma_demand=float(sigma_demand),
                sigma_supply=float(sigma_supply),
                supply_bias_pct=float(supply_bias),
                motif_demand=motif_demand,
                motif_supply=motif_supply,
                params_demand=params_demand,
                params_supply=params_supply,
                seed=scenario_seed,
            )
        )

if errors:
    st.error("\n".join(errors))

if "results" not in st.session_state:
    st.session_state.results = None

if st.button("Jalankan optimasi", type="primary") and not errors:
    master_rng = np.random.default_rng(global_seed) if global_seed is not None else np.random.default_rng()
    generated_scenarios = []
    for cfg in scenario_configs:
        mu_demand = int(round(cfg.demand_pct / 100.0 * capacity))
        mu_supply = int(round(mu_demand * (1.0 + cfg.supply_bias_pct / 100.0)))
        rng_seed = cfg.seed if cfg.seed is not None else master_rng.integers(0, 2**32 - 1)
        scenario_rng = np.random.default_rng(int(rng_seed))
        generated_scenarios.append(
            generate_scenario(
                cfg.name,
                int(horizon),
                mu_demand,
                cfg.sigma_demand,
                cfg.motif_demand,
                cfg.params_demand,
                mu_supply,
                cfg.sigma_supply,
                cfg.motif_supply,
                cfg.params_supply,
                rng=scenario_rng,
            )
        )

    with st.spinner("Menjalankan solver OR-Tools..."):
        results, bcubes = build_and_solve(
            int(capacity),
            int(horizon),
            int(expiry),
            int(checkpoint),
            int(initial_stock),
            int(grace_days),
            generated_scenarios,
            time_limit_s=int(time_limit),
            workers=int(workers),
            progress=False,
        )

    st.session_state.results = {
        "results": results,
        "bcubes": bcubes,
        "scenarios": generated_scenarios,
    }

output_state = st.session_state.results

if output_state:
    results = output_state["results"]
    bcubes = output_state["bcubes"]

    st.subheader("Ringkasan Hasil")
    status = results["status"]
    cols = st.columns(3)
    cols[0].metric("Status", status)
    cols[1].metric("CN optimum", f"{results.get('CN', 0)} labu")
    cols[2].metric("T_epig", f"{results.get('T_epig', 0)}")

    if results.get("per_scenario"):
        for idx, scenario_result in enumerate(results["per_scenario"]):
            st.markdown("---")
            st.subheader(f"{scenario_result['name']}")
            df = scenario_result["df"]
            metrics_cols = st.columns(4)
            metrics_cols[0].metric("Service level", f"{scenario_result['SL']*100:.2f}%")
            metrics_cols[1].metric("Shortage", int(scenario_result["Short_sum"]))
            metrics_cols[2].metric("Waste", int(scenario_result["Waste_sum"]))
            metrics_cols[3].metric("Objective Z", int(scenario_result["Z"]))

            st.dataframe(df, use_container_width=True)

            has_heatmap = bcubes is not None and idx < len(bcubes)
            if has_heatmap:
                tab_series, tab_heatmap = st.tabs(["Demand & Supply", "Heatmap Stok"])
            else:
                [tab_series] = st.tabs(["Demand & Supply"])
                tab_heatmap = None

            with tab_series:
                series_df = df.set_index("t")[
                    ["Demand", "Supply", "Use_total", "TotalStock"]
                ]
                st.line_chart(series_df)

            if has_heatmap and tab_heatmap is not None:
                with tab_heatmap:
                    heatmap_df = pd.DataFrame(bcubes[idx])
                    heatmap_df.index.name = "Age"
                    heatmap_df.columns = [f"Day {c}" for c in heatmap_df.columns]
                    melted = heatmap_df.reset_index().melt(
                        id_vars="Age", var_name="Day", value_name="Units"
                    )
                    # Strip the "Day " prefix to keep the axis labels numeric.
                    melted["Day"] = (
                        melted["Day"].str.replace("Day ", "", regex=False).astype(int)
                    )
                    heatmap_chart = (
                        alt.Chart(melted)
                        .mark_rect()
                        .encode(
                            x=alt.X("Day:O", title="Day"),
                            y=alt.Y("Age:O", title="Age"),
                            color=alt.Color(
                                "Units:Q",
                                title="Units",
                                scale=alt.Scale(scheme="reds"),
                            ),
                            tooltip=["Age:O", "Day:O", "Units:Q"],
                        )
                    )
                    st.altair_chart(heatmap_chart, use_container_width=True)

else:
    st.info("Konfigurasi parameter kemudian klik *Jalankan optimasi* untuk melihat hasil.")
