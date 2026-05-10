from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


def _fmt(v: Any, digits: int = 4) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _dominant_action(freq0: float, freq1: float, freq2: float) -> str:
    pairs = [("0", freq0), ("1", freq1), ("2", freq2)]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[0][0]


def build_window_explain_input(
    *,
    policy: str,
    compare_to: Optional[str],
    filters: Dict[str, Any],
    window: Dict[str, Any],
    neighbors: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "context": {
            "kind": "window",
            "policy": policy,
            "compare_to": compare_to,
            "filters": filters,
            "window_range": {
                "k": int(window["k"]),
                "t_start": int(window["t_start"]),
                "t_end": int(window["t_end"]),
            },
        },
        "bk": {
            "state_avg": {
                "q_bar": float(window["q_bar"]),
                "mu_bar": float(window["mu_bar"]),
                "sigma_bar": float(window["sigma_bar"]),
                "rho_bar": float(window["rho_bar"]),
            },
            "action_freq": {
                "freq0": float(window["freq0"]),
                "freq1": float(window["freq1"]),
                "freq2": float(window["freq2"]),
            },
            "r_bar": float(window["r_bar"]),
            "H": float(window["H_bar"]),
        },
        "neighbors": neighbors,
    }


def build_cluster_explain_input(
    *,
    policy_1: str,
    policy_2: str,
    filters: Dict[str, Any],
    cluster_row: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "context": {
            "kind": "cluster_compare",
            "policy": policy_1,
            "compare_to": policy_2,
            "filters": filters,
            "window_range": {"cluster": int(cluster_row["cluster"])},
        },
        "bk": {
            "policy_1": {
                "state_avg": {
                    "q_bar": float(cluster_row["q1"]),
                    "mu_bar": float(cluster_row["mu1"]),
                    "sigma_bar": float(cluster_row["sigma1"]),
                    "rho_bar": float(cluster_row["rho1"]),
                },
                "action_freq": {
                    "freq0": float(cluster_row["p0_1"]),
                    "freq1": float(cluster_row["p1_1"]),
                    "freq2": float(cluster_row["p2_1"]),
                },
                "r_bar": float(cluster_row["r1"]),
                "H": float(cluster_row["h1"]),
            },
            "policy_2": {
                "state_avg": {
                    "q_bar": float(cluster_row["q2"]),
                    "mu_bar": float(cluster_row["mu2"]),
                    "sigma_bar": float(cluster_row["sigma2"]),
                    "rho_bar": float(cluster_row["rho2"]),
                },
                "action_freq": {
                    "freq0": float(cluster_row["p0_2"]),
                    "freq1": float(cluster_row["p1_2"]),
                    "freq2": float(cluster_row["p2_2"]),
                },
                "r_bar": float(cluster_row["r2"]),
                "H": float(cluster_row["h2"]),
            },
            "distance_D": float(cluster_row.get("js_div", cluster_row.get("D", 0.0))),
        },
        "neighbors": None,
    }


def explain_stub(explain_input: Dict[str, Any]) -> str:
    context = explain_input.get("context", {})
    bk = explain_input.get("bk", {})
    kind = context.get("kind", "window")

    if kind == "cluster_compare":
        p1 = context.get("policy", "policy_1")
        p2 = context.get("compare_to", "policy_2")
        b1 = bk.get("policy_1", {})
        b2 = bk.get("policy_2", {})

        f1 = b1.get("action_freq", {})
        f2 = b2.get("action_freq", {})
        d1 = _dominant_action(float(f1.get("freq0", 0.0)), float(f1.get("freq1", 0.0)), float(f1.get("freq2", 0.0)))
        d2 = _dominant_action(float(f2.get("freq0", 0.0)), float(f2.get("freq1", 0.0)), float(f2.get("freq2", 0.0)))

        s1 = b1.get("state_avg", {})
        s2 = b2.get("state_avg", {})
        rho1 = float(s1.get("rho_bar", 0.0))
        rho2 = float(s2.get("rho_bar", 0.0))

        lines = [
            (
                f"In cluster {context.get('window_range', {}).get('cluster')}, the action-distribution gap "
                f"between {p1} and {p2} is D={_fmt(bk.get('distance_D', 0.0))}. "
                f"The dominant action is {d1} for {p1} and {d2} for {p2}."
            ),
            (
                f"For state averages, rho_bar is {rho1:.4f} vs {rho2:.4f}, "
                f"and r_bar is {_fmt(b1.get('r_bar', 0.0))} vs {_fmt(b2.get('r_bar', 0.0))}."
            ),
        ]

        if d1 != d2:
            lines.append("The two policies prefer different dominant actions in this state cluster, which drives a larger D.")
        else:
            lines.append("The dominant action is the same, but the secondary action shares differ, which still yields a noticeable D.")

        return "\n".join(lines)

    state = bk.get("state_avg", {})
    af = bk.get("action_freq", {})
    dom = _dominant_action(float(af.get("freq0", 0.0)), float(af.get("freq1", 0.0)), float(af.get("freq2", 0.0)))

    text = [
        (
            f"In window k={context.get('window_range', {}).get('k')}, policy {context.get('policy')} has "
            f"average reward r_bar={_fmt(bk.get('r_bar', 0.0))} and policy entropy H={_fmt(bk.get('H', 0.0))}."
        ),
        (
            f"State averages in this window are q={_fmt(state.get('q_bar'))}, mu={_fmt(state.get('mu_bar'))}, "
            f"sigma={_fmt(state.get('sigma_bar'))}, rho={_fmt(state.get('rho_bar'))}; "
            f"action frequencies are ({_fmt(af.get('freq0'),3)}, {_fmt(af.get('freq1'),3)}, {_fmt(af.get('freq2'),3)}), "
            f"with dominant action {dom}."
        ),
    ]

    neighbors = explain_input.get("neighbors")
    if neighbors:
        prev_bk = neighbors.get("prev")
        next_bk = neighbors.get("next")
        if prev_bk or next_bk:
            text.append(
                "Neighboring windows indicate local policy drift; compare adjacent action frequencies and r_bar "
                "to judge whether a policy shift is occurring."
            )

    return "\n".join(text)


def explain_with_openai(explain_input: Dict[str, Any], model: str = "gpt-4.1-mini") -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is not installed") from exc

    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are a grounded RL behavior analyst. "
        "Use only numeric evidence in the provided JSON. "
        "Do not mention any field that is not in the input. "
        "If uncertain, say uncertain. Keep it concise (4-6 sentences)."
    )
    user_prompt = json.dumps(explain_input, ensure_ascii=False, indent=2)

    response = client.responses.create(
        model=model,
        temperature=0.2,
        max_output_tokens=260,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    text = getattr(response, "output_text", "").strip()
    if text:
        return text

    return explain_stub(explain_input)
