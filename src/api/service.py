"""In-memory model service for cached PyTorch GRU inference and backtest execution."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch

from scripts.backtest import run_backtest_pipeline
from src.api.schemas import (
    BacktestRequest,
    BacktestResponse,
    PredictRequest,
    PredictResponse,
)
from src.models import StockGRU, load_config, resolve_device


class ModelService:
    """Service layer managing cached PyTorch checkpoints, fast inference, and backtesting."""

    def __init__(
        self,
        models_dir: Union[str, Path] = "models",
        data_dir: Union[str, Path] = "data/processed",
        config_path: Union[str, Path] = "configs/config.yaml",
        device: Optional[str] = None,
    ) -> None:
        """Initialize ModelService and setup model cache and compute device."""
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path) if self.config_path.exists() else {}

        target_device_str = (
            device
            if device is not None
            else self.config.get("training", {}).get("device", "auto")
        )
        self.device = resolve_device(target_device_str)
        self._model_cache: Dict[str, Tuple[StockGRU, Dict[str, Any], int]] = {}
        self._data_cache: Dict[str, pd.DataFrame] = {}

    def get_available_models(self) -> List[str]:
        """Scan models directory for available PyTorch checkpoints and return symbols."""
        if not self.models_dir.exists():
            return []

        models: List[str] = []
        for p in self.models_dir.glob("gru_*.pt"):
            sym = p.stem.replace("gru_", "").upper()
            if sym:
                models.append(sym)
        return sorted(list(set(models)))

    def load_model(self, symbol: str) -> Tuple[StockGRU, Dict[str, Any], int]:
        """Load and cache PyTorch StockGRU model and metadata for the requested symbol.

        Parameters
        ----------
        symbol : str
            Stock ticker symbol.

        Returns
        -------
        Tuple[StockGRU, Dict[str, Any], int]
            (instantiated model in eval mode, checkpoint metadata, sequence length).
        """
        clean_symbol = symbol.strip().upper()
        if clean_symbol in self._model_cache:
            return self._model_cache[clean_symbol]

        checkpoint_path = self.models_dir / f"gru_{clean_symbol}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint for '{clean_symbol}' not found at '{checkpoint_path}'"
            )

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        metadata = checkpoint.get("metadata", {})
        model_config = checkpoint.get("config", {}).get(
            "model", self.config.get("model", {})
        )

        input_dim = int(metadata.get("input_dim", model_config.get("input_dim", 23)))
        hidden_dim = int(model_config.get("hidden_dim", 64))
        num_layers = int(model_config.get("num_layers", 2))
        dropout = float(model_config.get("dropout", 0.2))
        use_layer_norm = bool(model_config.get("use_layer_norm", True))
        seq_length = int(
            metadata.get("seq_length", self.config.get("data", {}).get("seq_length", 30))
        )

        model = StockGRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=1,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        self._model_cache[clean_symbol] = (model, metadata, seq_length)
        return model, metadata, seq_length

    def predict(self, request: PredictRequest) -> PredictResponse:
        """Execute single-step predictive inference with optional Monte Carlo Dropout CI.

        Parameters
        ----------
        request : PredictRequest
            Inference parameters including ticker symbol and MC sample count.

        Returns
        -------
        PredictResponse
            Validated forecast with expected return, price, CI bounds, and latency.
        """
        start_time = time.perf_counter()
        clean_symbol = request.symbol.strip().upper()

        model, _, seq_length = self.load_model(clean_symbol)

        test_data_path = self.data_dir / clean_symbol / "test.csv"
        if not test_data_path.exists():
            raise FileNotFoundError(
                f"Processed test data for '{clean_symbol}' not found at '{test_data_path}'"
            )

        if clean_symbol in self._data_cache:
            df = self._data_cache[clean_symbol]
        else:
            df = pd.read_csv(test_data_path, index_col=0, parse_dates=True)
            self._data_cache[clean_symbol] = df

        feature_cols = [c for c in df.columns if c not in ["log_return", "Close"]]
        if len(df) < seq_length:
            raise ValueError(
                f"Insufficient historical data: required at least {seq_length} rows, found {len(df)}"
            )

        window_features = df[feature_cols].iloc[-seq_length:].values.astype(np.float32)
        tensor = torch.from_numpy(window_features).unsqueeze(0).to(self.device)
        current_price = float(df["Close"].iloc[-1])

        if request.mc_samples > 1 and request.return_ci:
            # Optimized batched stochastic forward pass across sample dimension
            model.eval()
            model.dropout.train(True)
            if model.num_layers > 1 and model.dropout_p > 0:
                model.gru.train(True)

            with torch.no_grad():
                batch_tensor = tensor.expand(request.mc_samples, -1, -1)
                samples = model.forward(batch_tensor)
                mean = samples.mean(dim=0)
                std = samples.std(dim=0)
                lower_95 = torch.quantile(samples, 0.025, dim=0)
                upper_95 = torch.quantile(samples, 0.975, dim=0)

            model.eval()
            pred_ret = float(mean.item())
            unc_std = float(std.item())
            lower_ci = float(lower_95.item())
            upper_ci = float(upper_95.item())
        else:
            pred = model.predict(tensor)
            pred_ret = float(pred.item())
            unc_std = 0.0
            lower_ci = pred_ret
            upper_ci = pred_ret

        pred_price = float(current_price * math.exp(pred_ret))
        lower_price = float(current_price * math.exp(lower_ci))
        upper_price = float(current_price * math.exp(upper_ci))
        direction = "UP" if pred_ret >= 0.0 else "DOWN"

        latency_ms = float((time.perf_counter() - start_time) * 1000.0)
        timestamp = datetime.now(timezone.utc).isoformat()

        return PredictResponse(
            symbol=clean_symbol,
            current_price=current_price,
            predicted_log_return=pred_ret,
            predicted_price=pred_price,
            uncertainty_std=unc_std,
            lower_95_ci=lower_ci,
            upper_95_ci=upper_ci,
            lower_95_price=lower_price,
            upper_95_price=upper_price,
            direction=direction,
            timestamp=timestamp,
            latency_ms=latency_ms,
        )

    def run_backtest(self, request: BacktestRequest) -> BacktestResponse:
        """Execute walk-forward financial backtest simulation and return summary KPIs.

        Parameters
        ----------
        request : BacktestRequest
            Backtest parameters including strategy, fee basis points, and split.

        Returns
        -------
        BacktestResponse
            Validated backtest report with financial KPIs and equity curve points.
        """
        clean_symbol = request.symbol.strip().upper()

        # Validate checkpoint and data existence
        self.load_model(clean_symbol)
        symbol_data_dir = self.data_dir / clean_symbol
        if not symbol_data_dir.exists():
            raise FileNotFoundError(
                f"Processed data directory for '{clean_symbol}' not found at '{symbol_data_dir}'"
            )

        pipeline_output = run_backtest_pipeline(
            symbol=clean_symbol,
            config_path=self.config_path,
            data_dir=self.data_dir,
            bps=request.fee_bps,
            threshold=request.threshold,
            strategy=request.strategy,
            split=request.split,
            initial_capital=request.initial_capital,
            device=str(self.device),
            verbose=False,
        )

        res_dict = pipeline_output.result.to_dict()
        return BacktestResponse(
            symbol=clean_symbol,
            strategy=request.strategy,
            fee_bps=request.fee_bps,
            threshold=request.threshold,
            split=request.split,
            initial_capital=request.initial_capital,
            summary=res_dict["summary"],
            equity_curve=res_dict["equity_curve"],
        )
