"""모드에 따라 TFT·TranAD의 학습·평가·추론·해석을 실행하는 단일 진입점."""

import argparse
from pathlib import Path
from time import perf_counter

import yaml
from src.udf.configuration import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    load_config,
    validate_analysis_period,
    validate_model_config,
)


def build_parser():
    """YAML 설정을 선택적으로 덮어쓸 공통 실행 옵션을 정의한다."""
    parser = argparse.ArgumentParser(
        description=(
            "GSC_EWS 딥러닝 학습·평가·온라인 추론·모델 해석을 실행합니다.\n"
            "CLI에서 생략한 값은 config.yaml을 따르며, 명시한 값만 해당 실행에서 덮어씁니다.\n"
            "YAML 안의 상대 경로는 YAML 위치 기준이고, CLI의 상대 경로는 현재 실행 폴더 기준입니다."
        ),
        epilog=(
            "사용 예시:\n"
            "  # YAML 설정 그대로 전체 파이프라인 실행\n"
            "  python main.py --config configs/config.yaml\n\n"
            "  # TranAD v2를 10 Epoch로 학습만 수행\n"
            "  python main.py --mode train --model tranad_v2 --max-epochs 10\n\n"
            "  # 저장된 모델과 Calibration으로 Test CSV 일괄 평가\n"
            "  python main.py --mode evaluate --checkpoint checkpoints/model.ckpt "
            "--calibration results/calibration.yaml --input data/test.csv\n\n"
            "  # 저장된 모델로 Attention·변수 중요도·SHAP 분석\n"
            "  python main.py --mode analyze --checkpoint checkpoints/model.ckpt "
            "--analyze-data data/test.csv --background data/train.csv "
            "--analyze-start '2026-01-01 00:00:00' --analyze-end '2026-01-02 23:59:59' "
            "--shap-target 'Motor_Power.PV' 'Pump_RPM.PV'\n\n"
            "  # 저장된 모델로 Online 입력 CSV의 새 행 감시\n"
            "  python main.py --mode online --checkpoint checkpoints/model.ckpt "
            "--calibration results/calibration.yaml --input data/online.csv\n\n"
            "  # 실제 실행 없이 최종 적용 설정 확인\n"
            "  python main.py --print-config"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        metavar="YAML",
        default=str(DEFAULT_CONFIG_PATH),
        help=("통합 실행 설정 YAML 경로. 파일은 UTF-8 YAML 매핑이어야 합니다. 기본값: %(default)s"),
    )
    parser.add_argument(
        "--mode",
        choices=["full", "train", "evaluate", "online", "analyze"],
        metavar="MODE",
        help=(
            "실행 단계 선택. full=학습·Calibration·Test 평가 전체, train=학습·Calibration 저장, "
            "evaluate=저장 모델로 Test CSV 일괄 추론, online=입력 CSV의 새 행만 지속 추론, "
            "analyze=저장 모델로 Attention·변수 중요도·SHAP만 생성. "
            "생략 시 YAML run.mode 사용"
        ),
    )
    parser.add_argument(
        "--model",
        metavar="NAME",
        help=(
            "full/train에서 학습할 모델명(tranad, tranad_v2, tft, lstm_ae). "
            "생략 시 YAML active_model 사용. evaluate/online/analyze의 모델은 체크포인트에서 결정되므로 입력 불가"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        metavar="CKPT",
        help=(
            "evaluate/online/analyze에서 불러올 .ckpt 파일 경로. evaluate/online은 같은 실행의 "
            "calibration.yaml도 필요하지만 analyze는 필요하지 않습니다. full/train은 새로 학습하므로 입력 불가. "
            "생략 시 YAML run.checkpoint_path 사용"
        ),
    )
    parser.add_argument(
        "--calibration",
        metavar="YAML",
        help=(
            "evaluate/online에서 사용할 calibration.yaml 경로. 피처별 Threshold와 점수 정규화 통계가 저장된 "
            "파일이며 checkpoint와 일치해야 합니다. 생략 시 YAML run.calibration_path 사용"
        ),
    )
    parser.add_argument(
        "--input",
        metavar="CSV",
        help=(
            "입력 CSV 경로. full/evaluate에서는 data.test_path를 덮어쓰고, online에서는 새 행을 감시할 "
            "파일로 사용합니다. 생략 시 YAML run.input_path를 사용하며, 그것도 null이면 full/evaluate는 "
            "data.test_path 사용. analyze에서는 --analyze-data 사용"
        ),
    )
    parser.add_argument(
        "--analyze-data",
        metavar="CSV",
        help=(
            "analyze에서 해석할 CSV 경로. Checkpoint의 피처 순서와 전처리를 적용합니다. "
            "생략 시 YAML run.input_path를 사용하며, 그것도 null이면 data.test_path 사용"
        ),
    )
    parser.add_argument(
        "--background",
        metavar="CSV",
        help=(
            "analyze의 SHAP 기준으로 사용할 정상 학습 CSV. analysis.shap=true일 때 필수입니다. "
            "생략 시 YAML run.background_path 사용"
        ),
    )
    parser.add_argument(
        "--analyze-start",
        metavar="DATETIME",
        help=(
            "analyze에서 분석할 첫 시각(포함). 예: '2026-01-01 00:00:00'. "
            "생략 시 YAML analysis.input_start 사용, 둘 다 null이면 입력의 처음부터 분석"
        ),
    )
    parser.add_argument(
        "--analyze-end",
        metavar="DATETIME",
        help=(
            "analyze에서 분석할 마지막 시각(포함). 예: '2026-01-02 23:59:59'. "
            "생략 시 YAML analysis.input_end 사용, 둘 다 null이면 입력의 끝까지 분석"
        ),
    )
    parser.add_argument(
        "--shap-target",
        nargs="+",
        metavar="FEATURE",
        help=(
            "analyze에서 SHAP으로 각각 설명할 하나 이상의 대상 피처명. 각 피처의 Anomaly Score를 별도로 설명하며, "
            "다른 피처는 원인 기여 변수로 계속 사용합니다. 생략 시 YAML analysis.shap_target 사용, "
            "YAML 값도 null이면 전체 피처의 평균 Anomaly Score를 설명"
        ),
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        help=(
            "이번 실행 결과를 저장할 폴더. 생략 시 YAML run.output_directory 사용. YAML 값도 null이면 "
            "<output.directory>/<설비>/<모델>/<버전>/<실행ID> 경로를 자동 생성"
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu", "cuda", "mps"],
        metavar="DEVICE",
        help="실행 장치(auto, cpu, gpu, cuda, mps). 생략 시 YAML run.device 사용(기본 YAML: auto)",
    )
    parser.add_argument(
        "--max-epochs",
        metavar="N",
        type=int,
        help=(
            "full/train에서 선택 모델의 Epoch 수를 양의 정수로 덮어씁니다. "
            "생략 시 YAML models.<active_model>.epochs 사용"
        ),
    )
    parser.add_argument(
        "--fast-dev-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "train 모드에서 학습·검증을 각각 한 Batch만 실행해 연결을 점검합니다. "
            "활성화: --fast-dev-run, 비활성화: --no-fast-dev-run, 생략 시 YAML run.fast_dev_run 사용"
        ),
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help=("학습·추론을 실행하지 않고 CLI 덮어쓰기와 절대 경로 변환이 끝난 최종 YAML만 출력합니다. 기본값: false"),
    )
    return parser


def resolve_config(args):
    """YAML의 모드·경로에 CLI 인자를 적용하며 CLI 상대 경로는 실행 폴더를 기준으로 한다."""
    overrides = {
        key: getattr(args, key) for key in ("mode", "device", "fast_dev_run") if getattr(args, key) is not None
    }
    paths = {
        "checkpoint": "checkpoint_path",
        "calibration": "calibration_path",
        "input": "input_path",
        "analyze_data": "input_path",
        "background": "background_path",
        "output": "output_directory",
    }
    for option, key in paths.items():
        value = getattr(args, option)
        if value is not None:
            overrides[key] = str(Path(value).expanduser().resolve())
    config = load_config(args.config, model=args.model, run_overrides=overrides)
    if config["run"]["mode"] not in {"full", "train"} and (args.model is not None or args.max_epochs is not None):
        raise ConfigError("--model and --max-epochs are available only in full or train mode")
    if args.max_epochs is not None:
        name = config["active_model"]
        config["models"][name]["epochs"] = args.max_epochs
        validate_model_config(name, config["models"][name])
    mode = config["run"]["mode"]
    if args.input is not None and mode == "analyze":
        raise ConfigError("analyze mode uses --analyze-data instead of --input")
    if args.analyze_data is not None and mode != "analyze":
        raise ConfigError("--analyze-data is available only in analyze mode")
    analysis_options = {
        "input_start": args.analyze_start,
        "input_end": args.analyze_end,
        "shap_target": args.shap_target,
    }
    if any(value is not None for value in analysis_options.values()) and mode != "analyze":
        raise ConfigError("--analyze-start, --analyze-end and --shap-target are available only in analyze mode")
    config["analysis"].update({key: value for key, value in analysis_options.items() if value is not None})
    validate_analysis_period(config["analysis"])
    return config


def main(argv=None):
    """모드별 공통 실행을 호출하고 생성된 체크포인트·보정값·결과 경로를 출력한다."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = resolve_config(args)
        if args.print_config:
            print(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))  # noqa: T201
            return None
        from src.workflow import CONSOLE_WIDTH, run_pipeline

        started_at = perf_counter()
        print(f"\n{'=' * CONSOLE_WIDTH}", flush=True)  # noqa: T201
        print("[Pipeline] 시작", flush=True)  # noqa: T201
        print("=" * CONSOLE_WIDTH, flush=True)  # noqa: T201
        try:
            result = run_pipeline(config)
        except Exception:
            print(f"[Pipeline] 실패 | 소요시간: {perf_counter() - started_at:.2f}초", flush=True)  # noqa: T201
            print("=" * CONSOLE_WIDTH, flush=True)  # noqa: T201
            raise
        else:
            print(f"[Pipeline] 완료 | 소요시간: {perf_counter() - started_at:.2f}초", flush=True)  # noqa: T201
            print("=" * CONSOLE_WIDTH, flush=True)  # noqa: T201
    except (ConfigError, FileNotFoundError) as error:
        parser.error(str(error))
    print(yaml.safe_dump(result, allow_unicode=True, sort_keys=False))  # noqa: T201
    return result


if __name__ == "__main__":
    main()
