# v5 구현 체크리스트

## Phase A — 배포 차단 해소 (P0)

- [ ] **A-1** pyproject.toml에 `python-source = "python"` 추가
- [ ] **A-2** lib.rs — build_config()에 trend 파라미터 추가, 10개 PyO3 함수에 `trend: Option<&str>` 추가
- [ ] **A-3** model.py — SARIMAXModel.__init__에 `trend='n'` 파라미터, fit/forecast/loglike에 전달
- [ ] **A-4** forecast.rs — forecast_pipeline()에서 trend 미래 반영 (state_intercept for h=n+1,...)
- [ ] **A-5** model.py — _generate_param_names()에 trend 이름 (`const`, `trend`) 추가
- [ ] **A-6** python_tests/test_trend.py — trend 'c','t','ct' 적합/예측/잔차/참조값 테스트

## Phase B — statsmodels 호환 (P1)

- [ ] **B-1** .github/workflows/ — ci.yml (push/PR), release.yml (tag v*)
- [ ] **B-2** model.py — plot_diagnostics() 4-패널 (잔차, 히스토그램, QQ, ACF)
- [ ] **B-3** model.py — _repr_html_() SARIMAXResult, ForecastResult
- [ ] **B-4** maturin build → 새 venv에서 설치 → sarimax_py import 검증

## Phase C — 사용성 (P2)

- [ ] **C-1** model.py — ForecastResult.to_dataframe(), SARIMAXResult.params_table()
- [ ] **C-2** lib.rs + model.py — get_prediction(start, end) in-sample 예측
- [ ] **C-3** model.py — summary() statsmodels 2열 레이아웃 + 하단 진단 통계
- [ ] **C-4** CLAUDE.md — 테스트 수, exog/std errors/trend 상태 업데이트
- [ ] **C-5** auto_arima — stepwise + grid search + Rayon 병렬

## 최종 검증

- [ ] `cargo test --all-targets` — 147+ 통과
- [ ] `pytest python_tests/ -v` — 277+ 통과 (trend 테스트 포함)
- [ ] wheel 빌드 → 새 venv 설치 → 전 기능 smoke test
- [ ] Jupyter 노트북에서 SARIMAXModel → fit → summary(HTML) → forecast → plot_diagnostics 확인
