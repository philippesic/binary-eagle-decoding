"""Strict session adapter for actual native w1ax_eagle_round_v1 events.

All costs are instrumented CPU wall intervals. No GPU cost is inferred.
Token-prefix identity is an output-history join, never a private-state identity.
"""

from __future__ import annotations

import hashlib
import json

STAGES = (
    "draft",
    "checkpoint",
    "target_decode_sync",
    "process",
    "check",
    "kv_repair",
    "accept_hook",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count(value):
    return type(value) is int and value >= 0


def identity(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def union_us(spans):
    merged = []
    for left, right in sorted(spans):
        if not merged or left > merged[-1][1]:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    return sum(right - left for left, right in merged)


def adapt_session(rows, metadata, aggregates=None):
    """Join one serial task to response IDs; reconcile supplied native deltas exactly.

    Metadata can omit provenance/cap: unresolved domains then stay explicit.
    Replay quality rows have an offset in native accepted counts; emission roles
    are unresolved rather than guessed. Checkpoint attempts never emit tokens.
    """
    require(bool(rows), "empty session")
    require(
        metadata.get("sampling_mode") in ("greedy", "sample_and_match"), "unsupported verifier mode"
    )
    task = rows[0]["task_id"]
    require(count(task), "invalid task identity")
    clock_domain = metadata.get("clock_domain", "unbound_native_process_clock")
    output = metadata.get("generated_token_ids")
    all_emitted = [
        t for r in rows if r["status"] != "checkpoint_replay" for t in r["emitted_token_ids"]
    ]
    leading = []
    unresolved = {
        "gpu_time",
        "private_cache_feature_processor_state",
        "prefill_setup_inter_round_and_trace_io",
        "legacy_stage_envelopes_not_callback_intervals",
    }
    if output is None:
        unresolved.add("response_emission_join")
    else:
        require(isinstance(output, list) and all(count(t) for t in output), "invalid response IDs")
        if all_emitted == output:
            pass
        elif output and all_emitted == output[1:]:
            leading = output[:1]
            unresolved.add("leading_seed_cost")
        else:
            raise ValueError("trace/response output mismatch")
    prompt = metadata.get("prompt_token_ids")
    if prompt is None:
        unresolved.add("committed_root_identity")
    else:
        require(isinstance(prompt, list) and all(count(t) for t in prompt), "invalid prompt IDs")
    cap = metadata.get("generation_cap")
    require(cap is None or count(cap), "invalid cap")
    if cap is None:
        unresolved.add("remaining_cap")
    if not metadata.get("ancestry"):
        unresolved.add("source_model_device_precision_ancestry")
    eos_ids = metadata.get("eos_token_ids", [])
    require(isinstance(eos_ids, list) and all(count(t) for t in eos_ids), "invalid EOS IDs")
    eos = set(eos_ids)
    history = list(leading)
    totals = dict(
        proposed_all=0,
        proposed_new=0,
        proposed_quality=0,
        accepted_decisions=0,
        accepted_emitted=0,
        accepted_not_emitted=0,
        correction=0,
        bonus=0,
        target_no_proposal=0,
        unresolved_role_emitted=0,
        emitted=0,
        verified_rounds=0,
        quality_rounds=0,
        checkpoint_attempts=0,
        eos_emitted=0,
        cap_terminated=0,
    )
    per_round, round_spans, begin_spans, stage_spans = [], [], [], {s: [] for s in STAGES}
    for i, row in enumerate(rows):
        require(row.get("schema") == "w1ax_eagle_round_v1", "wrong schema")
        require(row.get("clock") == "ggml_time_us_cpu_wall", "wrong clock")
        require(
            row["task_id"] == task
            and count(row.get("round_index"))
            and row.get("round_index") == i,
            "task or round order mismatch",
        )
        require(type(row.get("replay")) is bool, "missing replay flag")
        status = row["status"]
        require(status in ("complete", "no_proposal", "checkpoint_replay"), "wrong status")
        p, e, a = row["proposed_token_ids"], row["emitted_token_ids"], row["n_accepted"]
        require(
            isinstance(p, list) and isinstance(e, list) and all(count(t) for t in p + e),
            "invalid token IDs",
        )
        require(count(a) and a <= len(p), "invalid acceptance")
        require(
            count(row["n_proposed"])
            and count(row["n_emitted"])
            and row["n_proposed"] == len(p)
            and row["n_emitted"] == len(e),
            "count/ID mismatch",
        )
        start, end = row["round_start_us"], row["round_end_us"]
        require(
            count(start)
            and count(end)
            and end > start
            and count(row["round_us"])
            and row["round_us"] == end - start,
            "invalid round bounds",
        )
        if round_spans:
            require(start >= round_spans[-1][1], "overlapping attempts within serial task")
        round_spans.append((start, end))
        named = []
        intervals = []
        for stage in (*STAGES, "begin"):
            span = row["spans_us"].get(stage)
            require(
                isinstance(span, list) and len(span) == 2 and all(count(v) for v in span),
                "invalid span",
            )
            left, right = span
            require(
                right >= left and count(row[f"{stage}_us"]) and row[f"{stage}_us"] == right - left,
                "duration/span mismatch",
            )
            if span == [0, 0]:
                continue
            if stage == "begin":
                require(right <= start, "begin inside round")
                begin_spans.append((left, right))
            else:
                require(start <= left <= right <= end, "stage outside round")
                stage_spans[stage].append((left, right))
                named.append((left, right))
            intervals.append(
                dict(
                    stage=stage,
                    start_us=left,
                    end_us=right,
                    clock_domain=clock_domain,
                    timing_kind="host_wall_existing_sync"
                    if stage == "target_decode_sync"
                    else "host_call_wall_completion_unproven",
                    lifetime="outside_round_begin" if stage == "begin" else "verification_attempt",
                    attribution="shared_batch_possible"
                    if stage in ("draft", "target_decode_sync", "process")
                    else "slot",
                )
            )
        named_sum = sum(right - left for left, right in named)
        require(
            count(row["residual_us"]) and row["residual_us"] == max(0, end - start - named_sum),
            "residual mismatch",
        )
        union = union_us(named)
        accepted_emitted = correction = bonus = target_only = role_unknown = 0
        before = len(history)
        remaining = None if cap is None else cap - before
        require(remaining is None or remaining >= len(e), "emission exceeds cap")
        root = None if prompt is None or output is None else identity(prompt + history)
        native = row.get("session_context_v1")
        if native:
            require(count(native["generated_before"]), "invalid native generation position")
            require(
                type(native["remaining_before"]) is int and native["remaining_before"] >= -1,
                "invalid native remaining cap",
            )
            if "generated_after" in native:
                require(
                    native["generated_after"] == native["generated_before"] + len(e),
                    "native generated count mismatch",
                )
            if output is not None:
                require(
                    native["generated_before"] == before, "native root generation position mismatch"
                )
            if cap is not None and output is not None:
                require(native["remaining_before"] == remaining, "native remaining cap mismatch")
            require(
                native["native_stop_type"] in ("none", "eos", "limit", "word"), "invalid stop type"
            )
            # Slot prompt+seed describes model input after context shift. It is
            # preserved separately; never assert equality to full output prefix.
        if status == "checkpoint_replay":
            require(not e, "checkpoint attempt emitted")
            totals["checkpoint_attempts"] += 1
        else:
            totals["quality_rounds"] += 1
            totals["verified_rounds"] += status == "complete"
            require(0 < len(e) <= a + 1 + int(row["replay"]), "invalid emission length")
            if status == "no_proposal":
                require(not p and a == 0 and len(e) == 1, "invalid no-proposal")
                target_only = len(e)
            elif row["replay"]:
                role_unknown = len(e)
                unresolved.add("replay_emission_roles")
            else:
                accepted_emitted = min(a, len(e))
                require(e[:accepted_emitted] == p[:accepted_emitted], "accepted prefix mismatch")
                if len(e) > a:
                    if a == len(p):
                        bonus = 1
                    else:
                        correction = 1
                        require(e[a] != p[a], "correction matches rejected draft")
            if len(e) <= a and not row["replay"]:
                require(i == len(rows) - 1, "truncation before terminal round")
            totals["proposed_quality"] += len(p)
            totals["accepted_decisions"] += a
            totals["accepted_emitted"] += accepted_emitted
            totals["accepted_not_emitted"] += (
                max(0, a - accepted_emitted) if not row["replay"] else 0
            )
            totals["correction"] += correction
            totals["bonus"] += bonus
            totals["target_no_proposal"] += target_only
            totals["unresolved_role_emitted"] += role_unknown
            totals["emitted"] += len(e)
        totals["proposed_all"] += len(p)
        # Legacy derivation is bound to source9e2 iterate(drafting)/replay flow.
        # The additive producer field is authoritative when present.
        newly_proposed = len(p) if not row["replay"] else 0
        if native and "new_proposal_tokens" in native:
            newly_proposed = native["new_proposal_tokens"]
            require(
                count(newly_proposed) and newly_proposed <= len(p),
                "invalid newly generated proposal count",
            )
            require(not row["replay"] or newly_proposed == 0, "replay generated fresh proposals")
        totals["proposed_new"] += newly_proposed
        terminal = []
        if e and e[-1] in eos:
            terminal.append("eos")
            totals["eos_emitted"] += 1
        if cap is not None and before + len(e) == cap and e:
            terminal.append("cap_reached")
            totals["cap_terminated"] += 1
        if native and native["native_stop_type"] != "none":
            terminal.append("native_" + native["native_stop_type"])
        if terminal:
            require(i == len(rows) - 1, "terminal before last attempt")
        require(not any(t in eos for t in e[:-1]), "EOS before emission end")
        callback_intervals = []
        if native:
            for field, stage in (
                ("target_batches", "target_decode_sync"),
                ("process_batches", "process"),
            ):
                for callback in native.get(field, []):
                    left, right = callback["start_us"], callback["end_us"]
                    require(
                        count(left) and count(right) and start <= left <= right <= end,
                        "invalid callback bounds",
                    )
                    envelope = row["spans_us"][stage]
                    require(
                        envelope[0] <= left <= right <= envelope[1],
                        "callback outside stage envelope",
                    )
                    require(count(callback["batch_tokens"]), "invalid callback batch shape")
                    for key in (
                        "slot_tokens",
                        "slot_output_rows",
                        "feature_tokens",
                        "draft_decode_tokens",
                    ):
                        if key in callback:
                            require(
                                count(callback[key])
                                and (
                                    key == "draft_decode_tokens"
                                    or callback[key] <= callback["batch_tokens"]
                                ),
                                "invalid callback token count",
                            )
                    callback_intervals.append(
                        dict(
                            stage=stage,
                            clock_domain=clock_domain,
                            lifetime="native_batch_callback_within_attempt",
                            **callback,
                        )
                    )
        history.extend(e)
        per_round.append(
            dict(
                round_index=i,
                status=status,
                round_start_us=start,
                round_end_us=end,
                root_generated_position=before,
                root_prefix_sha256=root,
                remaining_cap=remaining,
                proposed=len(p),
                accepted_decisions=a,
                accepted_emitted=accepted_emitted,
                correction=correction,
                bonus=bonus,
                target_no_proposal=target_only,
                unresolved_role_emitted=role_unknown,
                emitted=len(e),
                terminal_reasons=terminal,
                native_context=native,
                intervals=intervals,
                callback_intervals=callback_intervals,
                stage_union_us=union,
                stage_overlap_us=named_sum - union,
                other_inside_round_us=end - start - union,
                attribution_note=(
                    "other is outside named envelopes; "
                    "within-envelope scheduler/other work is unresolved"
                ),
            )
        )
    require(
        totals["emitted"]
        == totals["accepted_emitted"]
        + totals["correction"]
        + totals["bonus"]
        + totals["target_no_proposal"]
        + totals["unresolved_role_emitted"],
        "emission conservation",
    )
    expected = dict(
        proposed=totals["proposed_new"],
        accepted=totals["accepted_decisions"],
        rounds=totals["verified_rounds"],
    )
    reconciliation = {}
    for field, actual in (aggregates or {}).items():
        require(field in expected and count(actual), "unsupported aggregate")
        reconciliation[field] = dict(
            native=actual, trace=expected[field], difference=expected[field] - actual
        )
        require(actual == expected[field], f"native aggregate mismatch: {field}")
    if aggregates is None:
        unresolved.add("native_aggregate_join")
    round_union = union_us(round_spans)
    return dict(
        schema="native_session_accounting_v1",
        session_id=metadata.get("session_id"),
        ancestry=metadata.get("ancestry"),
        verifier_mode=metadata["sampling_mode"],
        counts=totals,
        leading_untraced_emitted=len(leading),
        response_emitted=len(output) if output is not None else None,
        rounds=per_round,
        aggregate_reconciliation=reconciliation,
        costs=dict(
            clock_domain=clock_domain,
            timing_kind="instrumented_host_wall",
            round_sum_us=sum(right - left for left, right in round_spans),
            round_union_us=round_union,
            begin_union_us=union_us(begin_spans),
            stage_inclusive_union_us={s: union_us(v) for s, v in stage_spans.items()},
            other_inside_round_us=sum(r["other_inside_round_us"] for r in per_round),
            callback_union_us_by_stage={
                stage: union_us(
                    [
                        (c["start_us"], c["end_us"])
                        for r in per_round
                        for c in r["callback_intervals"]
                        if c["stage"] == stage
                    ]
                )
                for stage in ("target_decode_sync", "process")
            },
            callback_coverage="available_extension_callbacks_only",
            gpu_us=None,
            full_session_us=None,
        ),
        unresolved_domains=sorted(unresolved),
        session_ranking_identified=False,
    )


def pool_sessions(sessions):
    """Pool emission totals and unique CPU intervals within bound process clocks.

    Per-stage costs are inclusive; summing them double counts overlaps. Independent
    process clocks cannot be combined into one global critical-path duration.
    """
    require(bool(sessions), "empty sessions")
    require(
        all(s["costs"]["clock_domain"] != "unbound_native_process_clock" for s in sessions),
        "pool requires bound process clock domains",
    )
    session_ids = [s["session_id"] for s in sessions]
    require(
        all(isinstance(v, str) and v for v in session_ids)
        and len(set(session_ids)) == len(session_ids),
        "duplicate or missing session IDs",
    )
    counts = {key: sum(s["counts"][key] for s in sessions) for key in sessions[0]["counts"]}
    clocks, stages = {}, {}
    for session in sessions:
        clock = session["costs"]["clock_domain"]
        for row in session["rounds"]:
            # Recover complete round bounds from the adapter's preserved fields.
            clocks.setdefault(clock, []).append((row["round_start_us"], row["round_end_us"]))
            for interval in row["intervals"]:
                key = (clock, interval["stage"])
                stages.setdefault(key, []).append((interval["start_us"], interval["end_us"]))
    return dict(
        schema="native_session_pool_v1",
        response_emitted=(
            sum(s["response_emitted"] for s in sessions)
            if all(s["response_emitted"] is not None for s in sessions)
            else None
        ),
        leading_untraced_emitted=sum(s["leading_untraced_emitted"] for s in sessions),
        counts=counts,
        round_union_us_by_clock={k: union_us(v) for k, v in clocks.items()},
        stage_inclusive_union_us_by_clock={
            c: {s: union_us(stages.get((c, s), [])) for s in (*STAGES, "begin")} for c in clocks
        },
        full_session_us=None,
        session_ranking_identified=False,
        unresolved_domains=sorted(set().union(*(s["unresolved_domains"] for s in sessions))),
    )


def main():
    """One mapped native task + external response/provenance metadata -> JSON."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="one task's native round JSONL")
    parser.add_argument("metadata", type=Path, help="session response/provenance JSON")
    parser.add_argument("--aggregates", type=Path, help="request native counter deltas JSON")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
    metadata = json.loads(args.metadata.read_text())
    aggregate = json.loads(args.aggregates.read_text()) if args.aggregates else None
    print(json.dumps(adapt_session(rows, metadata, aggregate), indent=2))


if __name__ == "__main__":
    main()
