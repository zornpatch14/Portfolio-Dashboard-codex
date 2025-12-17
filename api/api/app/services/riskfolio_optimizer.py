from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator, Dict, Iterable, List
from uuid import uuid4

import numpy as np
import pandas as pd
import riskfolio as rp
import riskfolio.src.RiskFunctions as rk
from fastapi import HTTPException, status

from api.app.constants import DEFAULT_ACCOUNT_EQUITY
from api.app.schemas import (
    AllocationRow,
    BacktestSeries,
    ContractRow,
    FrontierPoint,
    JobEvent,
    JobResult,
    JobStatusResponse,
    MeanRiskSettings,
    OptimizerCorrelation,
    OptimizerJobRequest,
    OptimizerJobResponse,
    OptimizerSummary,
    SeriesPoint,
)

from .data_store import DataStore, store as data_store

logger = logging.getLogger(__name__)


class MeanRiskOptimizer:
    """Concrete Riskfolio mean-risk optimizer backed by portfolio cache data."""

    def __init__(self, store: DataStore) -> None:
        self.store = store

    def optimize(self, request: OptimizerJobRequest) -> JobResult:
        settings = request.mean_risk or MeanRiskSettings()
        returns, metas = self.store.returns_frame(request.selection)
        if returns.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No return history available for the current selection.",
            )

        rm_code = self._risk_measure(settings.risk_measure)
        obj_code = self._objective(settings.objective)
        kelly_mode = self._kelly_mode(settings.return_model)
        risk_free = self._annual_to_daily_rate(settings.risk_free_rate)

        portfolio = rp.Portfolio(returns=returns)
        portfolio.alpha = settings.alpha
        portfolio.lowerlng = settings.bounds.default_min
        portfolio.upperlng = settings.bounds.default_max
        portfolio.budget = settings.budget
        if settings.min_return is not None:
            portfolio.lowerret = settings.min_return
        if settings.turnover_limit is not None:
            portfolio.allowTO = True
            portfolio.turnover = settings.turnover_limit
        if settings.max_risk is not None:
            limit_attr = self._risk_limit_attribute(rm_code)
            if limit_attr:
                setattr(portfolio, limit_attr, settings.max_risk)

        A, b = self._linear_constraints(metas, returns.columns.tolist(), settings)
        if A is not None and b is not None:
            portfolio.ainequality = A
            portfolio.binequality = b

        portfolio.assets_stats(method_mu=settings.method_mu, method_cov=settings.method_cov)

        try:
            weights_df = portfolio.optimization(
                model="Classic",
                rm=rm_code,
                obj=obj_code,
                kelly=kelly_mode,
                rf=risk_free,
                l=settings.risk_aversion,
                hist=True,
            )
        except Exception as exc:  # pragma: no cover - CVXPY runtime
            logger.exception("Riskfolio optimization failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Riskfolio optimization failed: {exc}",
            ) from exc

        if weights_df is None or weights_df.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Riskfolio returned no solution."
            )

        capital = request.selection.account_equity or DEFAULT_ACCOUNT_EQUITY
        weights = weights_df.iloc[:, 0].astype(float)

        summary = self._build_summary(weights, returns, portfolio, settings, risk_free, capital, rm_code)
        allocation_rows = self._allocation_rows(weights, metas, request.selection, capital)
        contract_rows = self._contract_rows(allocation_rows, request.selection, capital)
        frontier = self._efficient_frontier(
            settings, portfolio, rm_code, kelly_mode, risk_free, returns
        )
        backtest = self._backtested_equity_series(returns, weights_df.iloc[:, 0], capital)
        correlation = self._correlation_payload(returns)

        return JobResult(
            summary=summary,
            weights=allocation_rows,
            contracts=contract_rows,
            frontier=frontier,
            backtest=backtest,
            correlation=correlation,
        )

    def _risk_measure(self, label: str) -> str:
        mapping = {
            "variance": "MV",
            "std": "MV",
            "mv": "MV",
            "msv": "MSV",
            "semi_variance": "MSV",
            "cvar": "CVaR",
            "cdar": "CDaR",
            "evar": "EVaR",
        }
        return mapping.get(label.lower(), label.upper())

    def _objective(self, label: str) -> str:
        mapping = {
            "max_return": "MaxRet",
            "maxret": "MaxRet",
            "min_risk": "MinRisk",
            "minrisk": "MinRisk",
            "utility": "Utility",
            "max_rar": "Sharpe",
            "sharpe": "Sharpe",
        }
        return mapping.get(label.lower(), "Sharpe")

    def _kelly_mode(self, label: str) -> str | None:
        mapping = {
            "arithmetic": None,
            "approx": "approx",
            "approx_log": "approx",
            "log": "exact",
            "exact": "exact",
        }
        return mapping.get(label.lower(), None)

    def _risk_limit_attribute(self, rm_code: str) -> str | None:
        mapping = {
            "MV": "upperdev",
            "MSV": "uppersdev",
            "CVaR": "upperCVaR",
            "CDaR": "upperCDaR",
            "EVaR": "upperEVaR",
        }
        return mapping.get(rm_code)

    def _linear_constraints(
        self,
        metas,
        columns: List[str],
        settings: MeanRiskSettings,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Translate UI bounds/group caps into Riskfolio linear inequality matrices.

        Each override generates a row in A * w <= b ensuring per-asset caps and
        symbol/strategy aggregates never exceed their configured weights.
        """
        rows: List[np.ndarray] = []
        rhs: List[float] = []
        index = {asset: idx for idx, asset in enumerate(columns)}

        def add_row(coeffs: np.ndarray, value: float) -> None:
            rows.append(coeffs)
            rhs.append(value)

        for asset_id, bounds in settings.bounds.overrides.items():
            idx = index.get(asset_id)
            if idx is None:
                continue
            low, high = bounds
            if high is not None:
                coeffs = np.zeros(len(columns))
                coeffs[idx] = 1.0
                add_row(coeffs, float(high))
            if low is not None:
                coeffs = np.zeros(len(columns))
                coeffs[idx] = -1.0
                add_row(coeffs, float(-low))

        def build_cap(name: str, cap: float, attr: str) -> None:
            coeffs = np.zeros(len(columns))
            for idx, meta in enumerate(metas):
                value = getattr(meta, attr, None)
                if value and value.upper() == name.upper():
                    coeffs[idx] = 1.0
            if coeffs.sum() > 0:
                add_row(coeffs, cap)

        for cap in settings.symbol_caps:
            build_cap(cap.name, cap.max_weight, "symbol")
        for cap in settings.strategy_caps:
            build_cap(cap.name, cap.max_weight, "strategy")

        if not rows:
            return None, None
        return np.vstack(rows), np.array(rhs, ndmin=2).T

    def _allocation_rows(
        self,
        weights: pd.Series,
        metas,
        selection,
        capital: float,
    ) -> List[AllocationRow]:
        rows: List[AllocationRow] = []
        for weight, meta in zip(weights, metas):
            label = meta.original_filename or meta.filename
            margin = selection.margin_overrides.get(meta.file_id)
            contracts = 0.0
            if margin and margin > 0:
                contracts = max(weight * capital / margin, 0.0)
            rows.append(
                AllocationRow(
                    asset=meta.file_id,
                    label=label,
                    weight=float(weight),
                    contracts=contracts,
                    margin_per_contract=margin,
                )
            )
        return rows

    def _contract_rows(
        self,
        allocations: List[AllocationRow],
        selection,
        capital: float,
    ) -> List[ContractRow]:
        rows: List[ContractRow] = []
        for alloc in allocations:
            margin = alloc.margin_per_contract or 0.0
            notional = max(alloc.weight * capital, 0.0)
            total_margin = alloc.contracts * margin if margin else None
            rows.append(
                ContractRow(
                    asset=alloc.asset,
                    contracts=alloc.contracts,
                    notional=notional,
                    margin=total_margin,
                )
            )
        return rows

    def _build_summary(
        self,
        weights: pd.Series,
        returns: pd.DataFrame,
        portfolio: rp.Portfolio,
        settings: MeanRiskSettings,
        risk_free: float,
        capital: float,
        rm_code: str,
    ) -> OptimizerSummary:
        w_vec = weights.astype(float).to_numpy().reshape((-1, 1))
        mu = np.array(portfolio.mu, dtype=float).reshape((-1, 1))
        cov = np.array(portfolio.cov, dtype=float)
        scenarios = returns.to_numpy()
        series = scenarios @ w_vec
        expected = float(mu.T @ w_vec)
        risk_val = self._risk_value(rm_code, series, w_vec, cov, settings.alpha)
        sharpe = (expected - risk_free) / risk_val if risk_val != 0 else 0.0
        max_dd = float(rk.MDD_Abs(series))
        return OptimizerSummary(
            objective=settings.objective,
            risk_measure=rm_code,
            expected_return=expected,
            risk=risk_val,
            sharpe=sharpe,
            max_drawdown=max_dd,
            capital=capital,
        )

    def _risk_value(
        self,
        rm_code: str,
        series: np.ndarray,
        weights: np.ndarray,
        cov: np.ndarray,
        alpha: float,
    ) -> float:
        if rm_code == "MV":
            return float(np.sqrt(weights.T @ cov @ weights))
        if rm_code == "MSV":
            return float(rk.SemiDeviation(series))
        if rm_code == "CVaR":
            return float(rk.CVaR_Hist(series, alpha))
        if rm_code == "CDaR":
            return float(rk.CDaR_Abs(series, alpha))
        if rm_code == "EVaR":
            return float(rk.EVaR_Hist(series, alpha)[0])
        return float(np.sqrt(weights.T @ cov @ weights))

    def _efficient_frontier(
        self,
        settings: MeanRiskSettings,
        portfolio: rp.Portfolio,
        rm_code: str,
        kelly_mode: str | None,
        risk_free: float,
        returns: pd.DataFrame,
    ) -> List[FrontierPoint]:
        points = settings.efficient_frontier_points
        if not points or points <= 0:
            return []
        try:
            frontier = portfolio.efficient_frontier(
                model="Classic",
                rm=rm_code,
                kelly=kelly_mode,
                points=points,
                rf=risk_free,
                hist=True,
            )
        except Exception as exc:  # pragma: no cover - CVXPY runtime
            logger.warning("Efficient frontier computation failed: %s", exc)
            return []

        data: List[FrontierPoint] = []
        cov = np.array(portfolio.cov, dtype=float)
        mu = np.array(portfolio.mu, dtype=float).reshape((-1, 1))
        scenarios = returns.to_numpy()
        for column in frontier.columns:
            w = frontier[column].astype(float).to_numpy().reshape((-1, 1))
            series = scenarios @ w
            expected = float(mu.T @ w)
            risk_val = self._risk_value(rm_code, series, w, cov, settings.alpha)
            weights_dict = {asset: float(frontier.loc[asset, column]) for asset in frontier.index}
            data.append(
                FrontierPoint(
                    expected_return=expected,
                    risk=risk_val,
                    weights=weights_dict,
                )
            )
        return data

    @staticmethod
    def _annual_to_daily_rate(annual_rate: float) -> float:
        """Convert an annualized simple rate into a calendar-day daily rate."""
        return float(annual_rate) / 365.0

    def _backtested_equity_series(
        self,
        returns: pd.DataFrame,
        weights: pd.Series,
        capital: float,
    ) -> List[BacktestSeries]:
        if returns.empty or weights.empty:
            return []

        ordered = weights.reindex(returns.columns).fillna(0.0)
        optimized = self._equity_line_from_returns(returns, ordered, capital, "Optimized")

        if returns.shape[1] == 0:
            return [optimized]

        equal_weights = pd.Series(1.0 / returns.shape[1], index=returns.columns)
        equal_weight_line = self._equity_line_from_returns(returns, equal_weights, capital, "Equal Weight")
        return [optimized, equal_weight_line]

    def _equity_line_from_returns(
        self,
        returns: pd.DataFrame,
        weights: pd.Series,
        capital: float,
        label: str,
    ) -> BacktestSeries:
        aligned = returns.reindex(columns=weights.index).fillna(0.0)
        port_returns = aligned.to_numpy() @ weights.to_numpy().reshape((-1, 1))
        compounded = np.cumprod(1.0 + port_returns.flatten())
        values = capital * compounded
        points = [
            SeriesPoint(timestamp=pd.Timestamp(idx).to_pydatetime(), value=float(val))
            for idx, val in zip(aligned.index, values)
        ]
        return BacktestSeries(label=label, points=points)

    def _correlation_payload(self, returns: pd.DataFrame) -> OptimizerCorrelation | None:
        if returns.empty or returns.shape[1] < 2:
            return None
        corr = returns.corr().fillna(0.0)
        labels = list(corr.columns)
        matrix = corr.to_numpy().tolist()
        return OptimizerCorrelation(mode="returns", labels=labels, matrix=matrix)


class RiskfolioJobManager:
    """Stateful job manager that tracks optimizer execution."""

    def __init__(self, optimizer: MeanRiskOptimizer) -> None:
        self.optimizer = optimizer
        self.jobs: Dict[str, JobStatusResponse] = {}

    def submit_mean_risk(self, request: OptimizerJobRequest) -> OptimizerJobResponse:
        job_id = str(uuid4())
        self.jobs[job_id] = JobStatusResponse(job_id=job_id, status="running", progress=10)
        try:
            result = self.optimizer.optimize(request)
        except HTTPException as exc:
            self.jobs[job_id] = JobStatusResponse(
                job_id=job_id,
                status="failed",
                progress=100,
                error=str(exc.detail),
            )
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unexpected optimizer failure: %s", exc)
            self.jobs[job_id] = JobStatusResponse(
                job_id=job_id, status="failed", progress=100, error=str(exc)
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unexpected optimizer failure",
            ) from exc

        self.jobs[job_id] = JobStatusResponse(
            job_id=job_id,
            status="completed",
            progress=100,
            result=result,
        )
        return OptimizerJobResponse(job_id=job_id, status="completed", message="optimizer job completed")

    def owns_job(self, job_id: str) -> bool:
        return job_id in self.jobs

    def job_status(self, job_id: str) -> JobStatusResponse:
        if job_id not in self.jobs:
            raise KeyError(job_id)
        return self.jobs[job_id]

    async def job_events(self, job_id: str) -> AsyncGenerator[str, None]:
        status = self.job_status(job_id)
        event = JobEvent(
            job_id=job_id,
            status=status.status,
            progress=status.progress,
            message="completed" if status.status == "completed" else status.status,
        )
        yield f"data: {event.model_dump_json()}\n\n"
        await asyncio.sleep(0.01)


riskfolio_optimizer = MeanRiskOptimizer(data_store)
riskfolio_jobs = RiskfolioJobManager(riskfolio_optimizer)
