"""True target/Q4-EAGLE readiness probe; no candidate admission is fabricated."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_dspark_screen as screen
import benchmark_native_eagle as common
from probe import EOS_FIXTURE, one_word_eos_satisfied

ROOT = Path(__file__).resolve().parents[2]


def run(config_path: Path, output: Path):
    cfg = json.loads(config_path.read_text())
    arms = cfg.get("diagnostic_arms", ["target_only", "eagle_q4_0"])
    allowed = {"target_only", "eagle_q4_0", "dspark_3", "dspark_7", "dflash_3", "dflash_7"}
    if not isinstance(arms, list) or len(arms) != 2 or len(set(arms)) != 2 or not set(arms) <= allowed:
        raise ValueError("bounded diagnostic requires two distinct supported arms")
    assets = {"binary", "target"} | {a.rsplit("_", 1)[0] if a.startswith(("dspark_", "dflash_")) else a for a in arms if a != "target_only"}
    for name in assets:
        if common.sha256(Path(cfg[name]["path"])) != cfg[name]["sha256"]:
            raise ValueError(f"changed baseline artifact: {name}")
    protocol_path = ROOT / "configs/dspark-screen/protocol.json"
    if common.sha256(protocol_path) != cfg["protocol_sha256"]:
        raise ValueError("baseline protocol changed")
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("batch_tokens") != 32 or protocol.get("microbatch_tokens") != 32:
        raise ValueError("requires common batch/microbatch32")
    prompts_path = ROOT / protocol["prompt_file"]
    if common.sha256(prompts_path) != protocol["prompt_sha256"]:
        raise ValueError("baseline prompts changed")
    prompts = common.load_prompts(prompts_path)
    output.mkdir(parents=True, exist_ok=False)
    common.json_write(output / "config.json", cfg)
    environment = common.environment_manifest(Path(cfg["binary"]["path"]))
    common.json_write(output / "environment.json", environment)
    gpu = environment["gpu_capability_query"]
    if gpu.get("exit_code") != 0 or "RTX 2080 Ti" not in gpu.get("stdout", "") or "7.5" not in gpu.get("stdout", ""):
        raise ValueError("requires actual RTX2080Ti/SM75")
    common.json_write(output / "gpu-before.json", common.gpu_snapshot())
    request_config = {"evaluation": {"max_output_tokens": 128, "temperature": 0.0,
                                     "seed": 42, "enable_thinking": False}}
    cases = [("warmup-00", prompts[0]), ("warmup-01", prompts[1]),
             ("prompt-00", prompts[0]), ("prompt-01", prompts[1]), ("eos", EOS_FIXTURE)]
    results = {}
    for arm in arms:
        cell = output / arm
        cell.mkdir()
        port = cfg.get("port", 18290)
        if not common.available_port("127.0.0.1", port):
            raise RuntimeError("baseline port occupied")
        env = {k: v for k, v in os.environ.items() if not k.startswith(("GGML_", "W1AX_", "DSPARK_", "EAGLE_"))}
        env.update(CUDA_VISIBLE_DEVICES="0", W1AX_ROUND_TRACE_JSONL=str(cell / "rounds.jsonl"))
        if arm.startswith(("dspark_", "dflash_")):
            env["DSPARK_REQUIRE_AUTHOR_LAYOUT"] = "1"
        if cfg.get("verify_positions"):
            positions = cfg["verify_positions"]
            if not isinstance(positions, list) or any(not isinstance(p, int) or p < 0 for p in positions):
                raise ValueError("invalid raw-logit diagnostic positions")
            env["W1AX_VERIFY_TRACE_JSONL"] = str(cell / "verify.jsonl")
            env["W1AX_VERIFY_TRACE_POSITIONS"] = ",".join(map(str, positions))
        command = screen.command(cfg, protocol, arm, port)
        common.json_write(cell / "launch.json", {"command": command,
                          "trace_env": {k: v for k, v in env.items() if k.startswith(("W1AX_", "GGML_", "CUDA_"))}})
        proc = None
        with (cell / "server.log").open("wb") as log:
            try:
                screen._launching = True
                try:
                    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                    common.json_write(cell / "owned_process.json", {"pid": proc.pid, "pgid": proc.pid})
                finally:
                    screen._launching = False
                if screen._pending_signal is not None:
                    signum, screen._pending_signal = screen._pending_signal, None
                    screen.interrupted(signum, None)
                common.wait_ready(proc, f"http://127.0.0.1:{port}", 300)
                common.json_write(cell / "gpu-loaded.json", common.gpu_snapshot())
                results[arm] = {}
                for name, prompt in cases:
                    dest = cell / name
                    result = common.execute_request(f"http://127.0.0.1:{port}",
                                                    common.request_body(request_config, prompt), 180, dest)
                    if result["generated_token_ids"] is None:
                        raise ValueError("baseline omitted raw output IDs")
                    if name == "eos" and not one_word_eos_satisfied(json.loads((dest / "response.json").read_text()), result):
                        raise ValueError("baseline raw EOS boundary failed")
                    results[arm][name] = result
            finally:
                if proc is not None:
                    screen.stop_owned_server(proc)
                    common.json_write(cell / "stopped.json", {"pid": proc.pid, "pgid": proc.pid, "returncode": proc.returncode})
        if arm != "target_only" and not any(r.get("n_proposed", 0) > 0 for r in screen.rows(cell / "rounds.jsonl")):
            raise ValueError("diagnostic drafter made no actual proposals")
    paired = {name: results[arms[0]][name]["generated_token_ids"] == results[arms[1]][name]["generated_token_ids"] for name, _ in cases}
    common.json_write(output / "readiness.json", {"scope": "two-arm native diagnostic; not five-repeat throughput", "arms": arms,
                       "candidate_admission": None, "results": results, "paired_output_ids": paired})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, screen.interrupted)
    signal.signal(signal.SIGINT, screen.interrupted)
    run(args.config.resolve(), args.output.resolve())
