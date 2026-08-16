"""
NL-to-Simulation Compiler — Backend
Computational Research Hackathon Project

Pipeline:
  1. User describes a 1D heat conduction problem in plain English.
  2. LLM call #1 parses that into a strict, validated SimulationSpec (JSON).
  3. A numerical stability check (CFL condition) guards the solver.
  4. A finite-difference solver actually computes the physics.
  5. LLM call #2 explains the results, grounded ONLY in the computed numbers.
"""

import os
import json
import logging
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sim-compiler")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Add it to a .env file (see .env.example).")

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

app = FastAPI(title="NL-to-Simulation Compiler")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=2000)


class SimulationSpec(BaseModel):
    """Structured spec the LLM must produce from a natural-language description
    of a 1D heat conduction problem."""

    rod_length_m: float = Field(..., gt=0, le=10, description="Length of the rod in meters")
    thermal_diffusivity: float = Field(
        ..., gt=0, le=1.0, description="Thermal diffusivity alpha (m^2/s)"
    )
    initial_temp_c: float = Field(..., ge=-273, le=2000)
    left_boundary_temp_c: float = Field(..., ge=-273, le=2000)
    right_boundary_temp_c: float = Field(..., ge=-273, le=2000)
    boundary_type: str = Field(
        "dirichlet", description="Currently only 'dirichlet' (fixed-temperature ends) supported"
    )
    simulation_time_s: float = Field(..., gt=0, le=10000)

    @field_validator("boundary_type")
    @classmethod
    def check_boundary_type(cls, v):
        if v.lower() != "dirichlet":
            raise ValueError("Only 'dirichlet' boundary conditions are currently supported")
        return v.lower()


class SimulationResult(BaseModel):
    spec: SimulationSpec
    times: List[float]
    x_grid: List[float]
    temperature_grid: List[List[float]]  # [time_step][x_index]
    explanation: str


# ---------------------------------------------------------------------------
# LLM call #1 — parse natural language into a structured SimulationSpec
# ---------------------------------------------------------------------------

PARSE_SYSTEM_PROMPT = """You are a physics-modeling assistant that converts natural-language
descriptions of 1D heat conduction problems into a strict JSON spec.

Return ONLY a JSON object with these exact fields:
- rod_length_m (float, 0 < x <= 10)
- thermal_diffusivity (float, 0 < x <= 1.0) — infer a reasonable value for the described
  material if not stated (e.g. metals ~1e-4 to 1e-5, insulators/wood/plastic ~1e-7)
- initial_temp_c (float)
- left_boundary_temp_c (float)
- right_boundary_temp_c (float)
- boundary_type (string, always "dirichlet")
- simulation_time_s (float, 0 < x <= 10000)

If the user doesn't specify a value, choose a sensible physically realistic default and
proceed — never ask a clarifying question, always return a complete spec.
Respond with JSON only, no prose, no markdown fences."""


def parse_prompt_to_spec(prompt: str) -> SimulationSpec:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            response_format={"type": "json_object"},
            temperature=0.2,
            messages=[
                {"role": "system", "content": PARSE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as e:
        logger.error(f"LLM parse call failed: {e}")
        raise HTTPException(status_code=502, detail=f"LLM parsing service unavailable: {e}")

    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.error(f"LLM returned invalid JSON: {raw!r}")
        raise HTTPException(status_code=502, detail="LLM returned malformed JSON spec")

    try:
        spec = SimulationSpec(**data)
    except Exception as e:
        logger.error(f"Spec validation failed: {e} | raw data: {data}")
        raise HTTPException(status_code=422, detail=f"Parsed spec failed validation: {e}")

    return spec


# ---------------------------------------------------------------------------
# Numerical solver — explicit finite-difference 1D heat equation
# ---------------------------------------------------------------------------

N_POINTS = 50   # spatial grid resolution
N_FRAMES = 40   # number of time snapshots returned to frontend


def solve_heat_equation(spec: SimulationSpec):
    dx = spec.rod_length_m / (N_POINTS - 1)

    # CFL stability condition for explicit finite differences: alpha*dt/dx^2 <= 0.5
    max_stable_dt = 0.5 * dx**2 / spec.thermal_diffusivity
    dt = min(max_stable_dt * 0.9, spec.simulation_time_s / N_FRAMES)

    n_steps = max(int(spec.simulation_time_s / dt), N_FRAMES)
    dt = spec.simulation_time_s / n_steps

    r = spec.thermal_diffusivity * dt / dx**2
    if r > 0.5:
        # Safety net — should not trigger given the dt choice above.
        logger.warning(f"Unstable r={r:.3f} detected, clamping dt for stability")
        dt = 0.5 * dx**2 / spec.thermal_diffusivity * 0.9
        n_steps = max(int(spec.simulation_time_s / dt), N_FRAMES)
        r = spec.thermal_diffusivity * dt / dx**2

    u = np.full(N_POINTS, spec.initial_temp_c, dtype=float)
    u[0] = spec.left_boundary_temp_c
    u[-1] = spec.right_boundary_temp_c

    x_grid = np.linspace(0, spec.rod_length_m, N_POINTS)
    save_every = max(n_steps // N_FRAMES, 1)

    times, frames = [], []

    for step in range(n_steps + 1):
        if step % save_every == 0 or step == n_steps:
            times.append(round(step * dt, 4))
            frames.append(u.copy().tolist())

        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
        u_new[0] = spec.left_boundary_temp_c
        u_new[-1] = spec.right_boundary_temp_c
        u = u_new

    return times, x_grid.tolist(), frames


# ---------------------------------------------------------------------------
# LLM call #2 — explain results, grounded in the actual computed numbers
# ---------------------------------------------------------------------------

EXPLAIN_SYSTEM_PROMPT = """You are a physics teaching assistant. You will be given the
setup and computed results of a 1D heat conduction simulation as JSON. Write a 3-4
sentence explanation of what happened, referencing ONLY the numeric values provided.
Do not invent numbers. Be concise and precise, suitable for someone reviewing a live demo."""


def explain_results(spec: SimulationSpec, frames) -> str:
    initial_profile = frames[0]
    final_profile = frames[-1]

    summary = {
        "rod_length_m": spec.rod_length_m,
        "thermal_diffusivity": spec.thermal_diffusivity,
        "simulation_time_s": spec.simulation_time_s,
        "left_boundary_c": spec.left_boundary_temp_c,
        "right_boundary_c": spec.right_boundary_temp_c,
        "initial_mean_c": round(float(np.mean(initial_profile)), 2),
        "final_min_c": round(min(final_profile), 2),
        "final_max_c": round(max(final_profile), 2),
        "final_mean_c": round(float(np.mean(final_profile)), 2),
    }

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.3,
            messages=[
                {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(summary)},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM explain call failed: {e}")
        # Non-fatal — fall back to a templated explanation so the demo never breaks.
        return (
            f"The rod's average temperature moved from {summary['initial_mean_c']}\u00b0C "
            f"to {summary['final_mean_c']}\u00b0C over {spec.simulation_time_s}s, "
            f"settling between the boundary values of {spec.left_boundary_temp_c}\u00b0C "
            f"and {spec.right_boundary_temp_c}\u00b0C. (LLM explanation unavailable: {e})"
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/api/simulate", response_model=SimulationResult)
def simulate(req: PromptRequest):
    logger.info(f"Prompt received: {req.prompt!r}")

    spec = parse_prompt_to_spec(req.prompt)
    logger.info(f"Parsed spec: {spec.model_dump()}")

    times, x_grid, frames = solve_heat_equation(spec)
    logger.info(f"Solved {len(times)} frames over {spec.simulation_time_s}s")

    explanation = explain_results(spec, frames)

    return SimulationResult(
        spec=spec,
        times=times,
        x_grid=x_grid,
        temperature_grid=frames,
        explanation=explanation,
    )


@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
