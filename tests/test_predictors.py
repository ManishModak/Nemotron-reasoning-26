import src.eval.predictors as predictors


def test_build_predictor_returns_heuristic_by_default() -> None:
    predictor = predictors.build_predictor({"predictor": {"type": "heuristic"}})

    assert isinstance(predictor, predictors.HeuristicPredictor)


def test_build_predictor_uses_transformers_backend_factory(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class DummyPredictor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(predictors, "TransformersKagglePredictor", DummyPredictor)

    predictor = predictors.build_predictor(
        {
            "predictor": {
                "type": "transformers_kaggle",
                "model_handle": "demo/model",
                "max_new_tokens": 64,
                "batch_size": 2,
                "temperature": 0.0,
                "do_sample": False,
                "trust_remote_code": True,
                "torch_dtype": "bfloat16",
                "device_map": "auto",
            }
        }
    )

    assert isinstance(predictor, DummyPredictor)
    assert captured["model_handle"] == "demo/model"
    assert captured["batch_size"] == 2
