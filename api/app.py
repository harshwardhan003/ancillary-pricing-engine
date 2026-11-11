"""Real-time ranking service.

POST /rank   take a partner session plus candidate car offers, return prices and order,
             respecting that partner's markup limits and booking-value setting
GET  /health liveness and model info
GET  /metrics Prometheus text format (request count, latency histogram, mean p_book)

Environment:
  MODEL_PATH   path to model.json            (default artifacts/model.json)
  PARTNER_CONFIG  path to partner_config.json (default artifacts/partner_config.json)
  LAMBDA          fallback booking value if a partner has no calibrated setting (default 10)
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ancillary import config as C
from ancillary.model import ConversionModel
from ancillary.optimiser import optimise
from ancillary.simulator import Sessions, rank_from_key

BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)


class SessionIn(BaseModel):
    partner: Literal["Ryanair", "easyJet", "Southwest Airlines", "Qantas", "Airbnb", "eDreams ODIGEO"]
    market: Literal["IE", "GB", "DE", "FR", "ES", "IT", "US", "AU"]
    device: Literal["mobile", "desktop"]
    returning_customer: bool = False
    lead_days: int = Field(ge=0, le=330, description="days between booking and trip")
    trip_days: int = Field(ge=1, le=60)
    party_size: int = Field(ge=1, le=9)


class OfferIn(BaseModel):
    offer_id: str
    supplier: str
    car_class: Literal["mini", "economy", "compact", "suv", "van"]
    supplier_rating: float = Field(ge=1, le=5)
    cost_per_day: float = Field(gt=0)
    ref_price_per_day: float | None = Field(default=None, gt=0,
                                            description="market reference price; defaults to cost x 1.20")


class RankRequest(BaseModel):
    session: SessionIn
    offers: list[OfferIn] = Field(min_length=2, max_length=12)
    policy: Literal["ml", "control"] = "ml"


class OfferOut(BaseModel):
    offer_id: str
    position: int
    price_per_day: float
    markup: float
    p_book: float
    expected_margin: float


class Guardrails(BaseModel):
    partner: str
    markup_min: float
    markup_max: float
    booking_value_lam: float
    revenue_share: float


class RankResponse(BaseModel):
    policy: str
    guardrails: Guardrails
    model_version: str
    latency_ms: float
    offers: list[OfferOut]


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests = {"ml": 0, "control": 0}
        self.errors = 0
        self.hist = [0] * (len(BUCKETS) + 1)
        self.lat_sum = 0.0
        self.p_sum = 0.0
        self.p_n = 0

    def observe(self, policy: str, seconds: float, p_mean: float):
        with self.lock:
            self.requests[policy] += 1
            self.lat_sum += seconds
            self.hist[next((i for i, b in enumerate(BUCKETS) if seconds <= b), len(BUCKETS))] += 1
            self.p_sum += p_mean
            self.p_n += 1

    def render(self) -> str:
        with self.lock:
            lines = ["# TYPE rank_requests_total counter"]
            lines += [f'rank_requests_total{{policy="{k}"}} {v}' for k, v in self.requests.items()]
            lines += ["# TYPE rank_errors_total counter", f"rank_errors_total {self.errors}",
                      "# TYPE rank_latency_seconds histogram"]
            cum = 0
            for b, c in zip(BUCKETS, self.hist):
                cum += c
                lines.append(f'rank_latency_seconds_bucket{{le="{b}"}} {cum}')
            cum += self.hist[-1]
            lines += [f'rank_latency_seconds_bucket{{le="+Inf"}} {cum}',
                      f"rank_latency_seconds_sum {self.lat_sum:.6f}",
                      f"rank_latency_seconds_count {cum}",
                      "# TYPE rank_mean_p_book gauge",
                      f"rank_mean_p_book {(self.p_sum / self.p_n) if self.p_n else 0:.6f}"]
            return "\n".join(lines) + "\n"


def _to_sessions(req: RankRequest) -> Sessions:
    se, offers = req.session, req.offers
    k = len(offers)
    sup_names = {n: i for i, n in enumerate(C.SUPPLIERS)}
    cost = np.array([[o.cost_per_day for o in offers]])
    ref = np.array([[o.ref_price_per_day or o.cost_per_day * (1 + C.BASELINE_MARKUP)
                     for o in offers]])
    one = lambda v: np.array([v])
    return Sessions(
        lead=one(float(se.lead_days)), trip=one(se.trip_days), party=one(se.party_size),
        market=one(C.MARKETS.index(se.market)), mobile=one(int(se.device == "mobile")),
        returning=one(int(se.returning_customer)), partner=one(C.PARTNERS.index(se.partner)), month=one(6),
        cls=np.array([[C.CAR_CLASSES.index(o.car_class) for o in offers]]),
        sup=np.array([[sup_names.get(o.supplier, 0) for o in offers]]),
        rating=np.array([[o.supplier_rating for o in offers]]), cost=cost, ref=ref,
    ) if k else None


def create_app(model_path: str | None = None, lam: float | None = None,
               partner_config: str | None = None) -> FastAPI:
    state: dict = {}
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        path = model_path or os.environ.get("MODEL_PATH", "artifacts/model.json")
        state["model"] = ConversionModel.load(path)
        fallback = lam if lam is not None else float(os.environ.get("LAMBDA", "10"))
        cfg_path = Path(partner_config or os.environ.get("PARTNER_CONFIG", "artifacts/partner_config.json"))
        cfg = json.loads(cfg_path.read_text())["partners"] if cfg_path.exists() else {}
        state["lam"] = np.array([cfg[p]["lam"] if p in cfg else fallback for p in C.PARTNERS])
        state["calibrated"] = bool(cfg)
        yield

    app = FastAPI(title="Ancillary pricing and sort engine", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        m = state.get("model")
        return {"status": "ok" if m else "loading", "model_version": m.version if m else None,
                "calibrated_partner_config": state.get("calibrated"),
                "lambda_by_partner": dict(zip(C.PARTNERS, state["lam"].tolist())) if "lam" in state else None}

    @app.get("/metrics")
    def prometheus():
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.post("/rank", response_model=RankResponse)
    def rank(req: RankRequest):
        t0 = time.perf_counter()
        model: ConversionModel = state["model"]
        try:
            s = _to_sessions(req)
            if req.policy == "control":
                price = s.cost * (1 + C.BASELINE_MARKUP)
                pos = rank_from_key(price)
                from ancillary.features import features_from_sessions
                p = model.predict(features_from_sessions(s, price, pos))
                markup = np.full_like(price, C.BASELINE_MARKUP)
            else:
                plan = optimise(model, s, lam=state["lam"])
                price, pos, p, markup = plan.price, plan.rank, plan.p_book, plan.markup
        except Exception as exc:  # pragma: no cover - defensive
            metrics.errors += 1
            raise HTTPException(status_code=500, detail=f"ranking failed: {exc}") from exc

        pi = int(s.partner[0])
        net = 1.0 - float(C.PARTNER_SHARE[pi])
        margin = p * (price - s.cost) * s.trip[:, None] * net  # CarTrawler net of partner revenue share
        order = np.argsort(pos[0])
        out = [OfferOut(offer_id=req.offers[i].offer_id, position=int(pos[0, i]) + 1,
                        price_per_day=round(float(price[0, i]), 2), markup=round(float(markup[0, i]), 2),
                        p_book=round(float(p[0, i]), 5), expected_margin=round(float(margin[0, i]), 3))
               for i in order]
        dt = time.perf_counter() - t0
        metrics.observe(req.policy, dt, float(p.mean()))
        guard = Guardrails(partner=req.session.partner, markup_min=float(C.PARTNER_MARKUP_MIN[pi]),
                           markup_max=float(C.PARTNER_MARKUP_MAX[pi]),
                           booking_value_lam=float(state["lam"][pi]), revenue_share=float(C.PARTNER_SHARE[pi]))
        return RankResponse(policy=req.policy, guardrails=guard, model_version=model.version,
                            latency_ms=round(dt * 1000, 2), offers=out)

    return app


app = create_app()
