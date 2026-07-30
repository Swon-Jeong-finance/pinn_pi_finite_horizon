# Liu 실험 실행·집계 Runbook

> 현재 Liu 코드 기준 실행 기록
>
> 최종 점검일: 2026-07-30
>
> 기준 작업 디렉터리: `~/PINN/Liu/AAAI`

이 문서는 Liu/Kim--Omberg 실험을 처음부터 다시 실행하거나, 이미 끝난
학습 결과에서 표와 그림을 재생성할 때 사용할 수 있는 실행 순서와 명령어를
정리한다. 모든 명령은 현재 코드의 실제 옵션 이름과 기본값을 기준으로
작성했다.

## 0. 가장 먼저 구분해야 하는 결과

Liu 코드에는 서로 비슷해 보이지만 의미가 다른 수렴·오차 결과가 여러 개
있다. 논문에서 서로 바꾸어 부르면 안 된다.

| 결과 | 계산 대상 | 대표 코드 | 논문상 용도 |
|---|---|---|---|
| Main Figure 2 | outer iteration별 Value/Policy 상대 \(L^2\) 오차 | `postprocess_contraction.py` | 본문 학습 수렴 그림 |
| E6 | achieved \(p_{\mathrm{res}}\) 대비 최종 오차 | `aggregate_e6.py` | residual-tolerance 실험 |
| E4 / Figure S1 | \(e_k^{\mathrm{approx}}=\|\widetilde v_k-E(\alpha_{k-1})\|_{X_{\mathrm{ev}}}\) | `liu_exact_map_fd.py`, `aggregate_e4_tolerance.py` | residual-to-approximation 관계 |
| Empirical \(X_{\mathrm{ev}}\) ratio | 인접 checkpoint 오차비 | `postprocess_empirical_xev_ratio.py` | 별도 수축 진단 |
| Exact-map ratio | frozen-policy FD map의 오차비 | `aggregate_liu_exact_map.py` | 유한영역 exact-map 진단 |

중요한 해석 규칙은 다음과 같다.

- Main Figure 2는 contraction-ratio 그림이 아니다.
- Figure 2의 Policy 곡선은 `diag_RelL2_vartheta`를 사용한다.
  `diag_RelL2_theta`는 legacy raw-dollar control 진단이므로 Figure 2에
  사용하지 않는다.
- `aggregate_e6.py`의 `RelL2_theta`는 역사적인 CSV 필드명이지만 현재
  의미는 wealth-normalized policy \(\vartheta=\theta/w\)이다.
- E4의 \(x\)축은 nominal target이 아니라 실제 achieved
  \(p_{\mathrm{res}}\)이다.
- E4 refinement 실패는 Figure S1에서 수치 민감도를 보고하는 항목이다.
  exact-map ratio가 1보다 작다는 조건과는 별개이다.
- exact-map 결과는 finite-domain, boundary-projected policy-extension
  진단이다. whole-space exact PI map의 증명으로 표현하면 안 된다.

전체 흐름은 다음과 같다.

```text
closed-form residual gate
          |
          v
main 10-seed training
          |
          +--> artifact audit --> Table 3 / E1 / Main Figure 2
          |                         |
          |                         +--> E9 / welfare (선택)
          |
          +--> main M=1 empirical/exact-map ratio (선택)

independent p_res training --------------------------+
          |                                          |
          +--> independent residual/error summary   |
          |                                          v
          +--> parallel FD audit --> E4 Figure S1 / boundary report
                                      |
                                      +--> exact-map ratio (선택)

common-warm-start E6  --> paper E6 summary
non-affine sweep      --> Figure 4
timing sweep          --> E8 computation summary
```

## 1. 공통 준비

작업 디렉터리로 이동한다.

```bash
cd ~/PINN/Liu/AAAI
```

이 문서의 대표 seed 집합은 다음과 같다.

```bash
MAIN_SEEDS='1,2,3,5,7,11,17,23,42,101'
PRES_SEEDS='1,11,23,42,101'
```

그림 생성 서버에서 Matplotlib cache 경고를 피하려면 쓰기 가능한 cache를
지정할 수 있다.

```bash
export MPLCONFIGDIR="${TMPDIR:-/tmp}/liu_matplotlib_cache"
mkdir -p "$MPLCONFIGDIR"
```

### 1.1 Output root 원칙

- main, independent \(p_{\mathrm{res}}\), common-warm-start E6, non-affine,
  timing은 서로 다른 training root를 사용한다.
- evaluation window, FD wealth domain, drift scheme, refinement rule,
  policy extension 가운데 하나라도 바꾸면 별도의 FD output root를 쓴다.
- 완료된 run은 `_SUCCESS`가 있으면 launcher가 건너뛴다.
- 새 target만 추가하고 학습 설정과 코드 계약이 같다면 같은 root에서
  다시 실행해도 된다.
- `FORCE_RERUN=1`은 기존 성공 run도 다시 계산하므로 의도적으로 전체
  재실행할 때만 사용한다.
- E6 trainer source가 바뀌면 기존 warm-start bundle은 다시 만들도록
  새 output root를 사용하는 것이 안전하다.

### 1.2 Affine closed-form residual gate

main 학습 결과를 논문에 사용하기 전, 현재 residual 구현과 affine
closed-form 해가 일치하는지 검사한다.

```bash
python3 check_residual_substitution.py \
  --solver both \
  --risk-premium-mode affine \
  --nonaffine-eps 0 \
  --require-torch \
  --json outputs/derived/liu_residual_substitution.json \
  --overwrite
```

이 검사는 PINN nonlinear HJB residual과 PI-PINN frozen-policy linear
residual을 모두 확인한다. FD refinement나 학습 성능 검사를 대신하는 것은
아니다.

## 2. Main 10-seed 학습

현재 `SWEEP_PROFILE=main`은 \(N=30\), \(M\in\{1,3,5\}\)에서 PINN과
PI-PINN을 모두 실행한다. M=1 PI-PINN에는 exact-map/E4에 필요한 모든
outer checkpoint가 저장된다.

```bash
SEEDS="1,2,3,5,7,11,17,23,42,101" \
DEVICE_LIST="cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6" \
JOBS_PER_GPU=1 \
AGGREGATE=0 \
bash tune_pipinn.sh outputs/main_10seed_20260723 6
```

`AGGREGATE=0`은 학습과 strict paper aggregation을 분리하기 위한
설정이다. 학습이 모두 끝난 뒤 아래 명령으로 집계한다.

현재 main 기본 학습 계약은 다음과 같다.

| 항목 | 현재 기본값 |
|---|---:|
| \(N\) | 30 |
| \(M\) | 1, 3, 5 |
| inner epochs | 2,000 |
| outer iterations | 20 |
| batch size | 10,000 |
| initial LR | \(3\times10^{-4}\) |
| PI-PINN LR | `carry_plateau` |
| Adam state | outer 사이 유지 |
| PE resampling | `0`, 비활성화 |
| PI-PINN initialization | myopic |
| policy clipping | 없음 |
| primary eval margin | 0.10 |
| market seed | 12 |
| diagnostic interval | 매 outer |

모든 training seed는 같은 `market_seed=12` 시장을 풀고, seed는 network
초기화·collocation·optimizer 난수만 바꾼다.

대표 M=1 PI-PINN run 이름은 다음과 같다.

```text
pipinn_rho_canonical_v1_m_states1_e3b_checkpoints1_seed1
```

### 2.1 병렬·resume 규칙

- `DEVICE_LIST`: 사용할 GPU 목록
- `JOBS_PER_GPU`: GPU당 허용할 동시 training job 수
- `tune_pipinn.sh`의 두 번째 positional argument: 전체 worker 상한
- training launcher 기본 CPU thread:
  - OMP/MKL/Torch: 2
  - OpenBLAS/NumExpr: 1
- `_SUCCESS`: 재실행 시 skip
- `_FAILED`: 같은 명령을 다시 실행하면 retry
- `_STOPPED_EARLY`: 기본 skip, `RERUN_STOPPED=1`일 때 retry
- `_manifest.tsv`: 실제 tag, device, override, output 기록
- `_job_failures.tsv`: 실패 job 목록
- `logs/`: 개별 run log

## 3. Main 결과 audit와 기본 표

### 3.1 Training artifact audit

긴 후처리를 시작하기 전에 checkpoint와 market provenance를 동결·검증한다.

```bash
python3 audit_run_artifacts.py \
  --out-root outputs/main_10seed_20260723 \
  --expected-seeds '1,2,3,5,7,11,17,23,42,101' \
  --output outputs/main_10seed_20260723/derived/posthoc_audit \
  --overwrite
```

### 3.2 Table 3 final-error aggregation

```bash
python3 aggregate_seeds.py \
  --out-root outputs/main_10seed_20260723 \
  --output outputs/main_10seed_20260723/seed_summary \
  --expected-seeds '1,2,3,5,7,11,17,23,42,101' \
  --min-runs 10 \
  --expected-n-assets 30 \
  --expected-m-states '1,3,5' \
  --expected-models 'pinn,pipinn' \
  --headline-margin 0.10
```

주요 결과는 다음과 같다.

```text
seed_summary/
  summary_headline.csv
  summary_long.csv
  success_rates.csv
  runs_index.csv
  market_hashes.csv
  groups.json
```

독립 \(p_{\mathrm{res}}\) sweep에는 이 strict Table-3 aggregator를 쓰지
않는다. target마다 별도 training group이므로 one-group 검증이 실패하는
것이 정상이다.

### 3.3 E1 assumption diagnostics

```bash
python3 aggregate_diagnostics.py \
  --out-root outputs/main_10seed_20260723 \
  --models pinn,pipinn \
  --m-states 1,3,5 \
  --expected-n-assets 30 \
  --expected-seeds '1,2,3,5,7,11,17,23,42,101' \
  --min-seeds 10 \
  --outer-min 1 \
  --output outputs/main_10seed_20260723/derived/e1_diagnostics \
  --overwrite
```

첫 outer를 제외한 burn-in sensitivity는 같은 source run을 다시 학습하지
않고 다음처럼 별도 reduction으로 만든다.

```bash
python3 aggregate_diagnostics.py \
  --out-root outputs/main_10seed_20260723 \
  --models pinn,pipinn \
  --m-states 1,3,5 \
  --expected-n-assets 30 \
  --expected-seeds '1,2,3,5,7,11,17,23,42,101' \
  --min-seeds 10 \
  --outer-min 2 \
  --output outputs/main_10seed_20260723/derived/e1_diagnostics \
  --overwrite
```

## 4. Main Figure 2

Main Figure 2는 \(M=3\) PI-PINN의 Value와 normalized Policy 상대
\(L^2\) 오차를 outer 1--20에서 그린다.

```bash
python3 postprocess_contraction.py \
  --out-root outputs/main_10seed_20260723 \
  --output outputs/main_10seed_20260723/derived/figure2_empirical_convergence \
  --n-assets 30 \
  --m-states 3 \
  --expected-seeds '1,2,3,5,7,11,17,23,42,101' \
  --min-seeds 10 \
  --primary-margin 0.10 \
  --fit-window 1-4 \
  --sensitivity-windows 1-3,1-5 \
  --endpoint-outer 20 \
  --fig-width 4.8 \
  --fig-height 3.2 \
  --font-size 10 \
  --dpi 300 \
  --formats png,pdf,eps \
  --bbox-inches tight \
  --overwrite
```

그림은 다음 규칙을 사용한다.

- y축: log scale
- Value: `diag_RelL2_V`
- Policy: `diag_RelL2_vartheta`
- 굵은 선: outer별 seed 산술평균
- 음영: 평균 \(\pm\) sample SD
- 개별 seed 선: 기본적으로 표시하지 않음
- residual 제3곡선: 기본적으로 표시하지 않음

추가 진단 곡선을 원할 때만 다음 옵션을 사용한다.

```text
--show-seed-trajectories
--include-val-pres
```

현재 조절 가능한 Figure 2 그림 옵션은 다음과 같다.

| 옵션 | 기본값 | 의미 |
|---|---:|---|
| `--fig-width` | 4.8 | inch |
| `--fig-height` | 3.2 | inch |
| `--font-size` | 10 | point |
| `--font-family` | Matplotlib default | 설치된 font 이름 |
| `--dpi` | 300 | raster 해상도 |
| `--formats` | `png,pdf` | `png,pdf,svg,eps` 중 선택 |
| `--bbox-inches` | `tight` | `tight` 또는 `standard` |

Figure 2의 mean linewidth 2.2, marker 없음, band alpha 0.18, grid alpha
0.22는 현재 CLI 옵션이 아니라 코드에 고정되어 있다.

주요 산출물은 다음과 같다.

```text
figure2_empirical_convergence.{png,pdf,eps}
figure2_trajectories.csv
figure2_pointwise_summary.csv
figure2_endpoint_summary.csv
figure2_seed_decay_fits.csv
figure2_decay_summary.csv
figure2_runs_used.csv
figure2_metadata.json
```

## 5. Main의 선택적 추가 평가

### 5.1 E9 nested evaluation windows

```bash
python3 evaluate_margin_bundle.py \
  --out-root outputs/main_10seed_20260723 \
  --models both \
  --n-assets 30 \
  --m-states 1,3,5 \
  --expected-seeds '1,2,3,5,7,11,17,23,42,101' \
  --min-seeds 10 \
  --margins 0.05,0.10,0.20,0.30 \
  --n-points 100000 \
  --base-seed 727 \
  --device cuda:1 \
  --strict-crosscheck \
  --output outputs/main_10seed_20260723/derived/e9_margin_bundle \
  --overwrite
```

### 5.2 Welfare provenance preflight

```bash
python3 evaluate_welfare.py \
  --out-root outputs/main_10seed_20260723 \
  --models both \
  --m-states 1,3,5 \
  --expected-seeds '1,2,3,5,7,11,17,23,42,101' \
  --validate-only
```

### 5.3 Welfare evaluation

```bash
python3 evaluate_welfare.py \
  --out-root outputs/main_10seed_20260723 \
  --models both \
  --m-states 1,3,5 \
  --expected-seeds '1,2,3,5,7,11,17,23,42,101' \
  --device cuda:1
```

기본값은 `w0=0.5`, `x0=xbar`, 100,000 paths, 1,000 Euler steps,
`mc_seed=2718`이다. `--include-raw`는 unprojected extension sensitivity를
추가하므로 처음부터 사용할지 결정하고, 나중에 추가할 때는 별도 output을
쓰는 것이 안전하다.

## 6. 독립 \(p_{\mathrm{res}}\) 학습

이 실험은 각 target을 ordinary initialization에서 독립적으로 시작한다.
common-warm-start E6와는 다른 실험이다.

현재 `tune_pipinn.sh`에는 별도 `pres` profile이 없으므로, launcher를
source해 동일한 queue·tag·resume 기능을 재사용한다.

### 6.1 여러 target 전체 실행

최종 기본 target 집합은 `0.1,0.05,0.02,0.01,0.005`의 다섯 값이다.
추가 pilot target이 필요하면 6.2와 같은 방식으로 별도로 더할 수 있다.

```bash
SEEDS="1,11,23,42,101" \
DEVICE_LIST="cuda:1,cuda:2,cuda:3,cuda:4,cuda:5" \
JOBS_PER_GPU=1 \
bash -lc '
  ROOT="outputs/pres_5seed"
  source ./tune_pipinn.sh "$ROOT" 5

  for PT in 0.1 0.05 0.02 0.01 0.005; do
    run_pipinn auto \
      m_states=1 \
      eval_epochs=50000 \
      pres_target="$PT"
  done

  run_all_jobs
'
```

대표 결과 이름은 다음 형태이다.

```text
pipinn_rho_canonical_v1_m_states1_eval_epochs50000_pres_target0.02_seed23
```

### 6.2 target 0.15만 추가

기존 `outputs/pres_5seed`에 다른 target이 이미 성공해 있다면 다음 명령은
0.15만 추가한다.

```bash
SEEDS="1,11,23,42,101" \
DEVICE_LIST="cuda:1,cuda:2,cuda:3,cuda:4,cuda:5" \
JOBS_PER_GPU=1 \
bash -lc '
  ROOT="outputs/pres_5seed"
  source ./tune_pipinn.sh "$ROOT" 5

  run_pipinn auto \
    m_states=1 \
    eval_epochs=50000 \
    pres_target=0.15

  run_all_jobs
'
```

같은 root와 같은 설정으로 target 목록을 확장하면 기존 `_SUCCESS` run은
skip되고 새 target만 실행된다.

### 6.3 독립 target 집계

root에 최종 다섯 target만 있을 때는 다음처럼 집계한다.

```bash
python3 aggregate_e6.py \
  --out-root outputs/pres_5seed \
  --output outputs/pres_5seed/e6_summary \
  --model-type pipinn \
  --independent-standard-runs \
  --expected-n-assets 30 \
  --expected-m-states 1 \
  --expected-seeds '1,11,23,42,101' \
  --expected-targets '0.1,0.05,0.02,0.01,0.005' \
  --min-runs-per-tolerance 5 \
  --metrics e_Xev,RelL2_V,RelL2_theta \
  --fig-width 6.4 \
  --fig-height 4.2 \
  --font-size 10 \
  --dpi 300 \
  --formats png,pdf \
  --overwrite
```

같은 root에 6.2의 `0.15` 같은 pilot target도 존재한다면
`aggregate_e6.py`의 `--expected-targets`는 그 pilot까지 포함한 실제 전체
target 집합과 일치해야 한다. 현재 `aggregate_e6.py`에는 E4 집계의
`--select-target` 같은 subset selector가 없다. 최종 다섯 target만으로
별도 E6 summary가 필요하면 처음부터 pilot을 별도 training root에 두는
것이 가장 안전하다. 반면 E4/Figure S1 집계는 같은 FD root에서도
`--select-target '0.1,0.05,0.02,0.01,0.005'`로 최종 다섯 값만 선택할
수 있다.

이 결과는 metadata에서 `aggregation_protocol=standard-independent`로
기록된다. 논문용 common-warm-start E6라고 부르면 안 된다.

E6 그림은 log-log 축에서 target별 mean \(\pm\) SD, pooled fitted slope,
slope-one guide를 표시한다. 현재 조절 가능한 그림 옵션은
`--fig-width`, `--fig-height`, `--font-size`, `--dpi`, `--formats`이다.
선·마커·음영 style은 현재 코드에 고정되어 있다.

## 7. 논문용 common-warm-start E6

통제된 E6는 전용 launcher를 사용한다.

```bash
SEEDS="1,2,3,5,7" \
DEVICE_LIST="cuda:1,cuda:2,cuda:3,cuda:4,cuda:5" \
JOBS_PER_GPU=1 \
E6_N_ASSETS=30 \
E6_M_STATES=1 \
E6_TARGETS="0.1,0.05,0.02,0.01,0.005" \
E6_OUTER_ITERS=20 \
E6_WARMUP_MAX_EPOCHS=30000 \
E6_BRANCH_MAX_EPOCHS=30000 \
E6_PE_RESAMPLE_EVERY=0 \
E6_CARRY_LR_MIN=1e-5 \
E6_CARRY_LR_MAX=3e-4 \
E6_RESET_LR_EACH_OUTER=1 \
E6_FORMATS="png,pdf" \
E6_DPI=300 \
AGGREGATE_E6=0 \
bash tune_pipinn_e6.sh outputs/liu_e6_n30_m1 5
```

launcher의 실행 순서는 다음과 같다.

1. 각 seed에서 target 1 warm-up을 정확히 한 outer 수행한다.
2. model, Adam, LR, RNG bundle을 저장한다.
3. 모든 warm-up과 bundle을 검증한다.
4. 동일 seed bundle에서 각 target branch를 만든다.
5. target branch의 매 outer 시작 LR을 `carry_lr_max=3e-4`로 복원한다.
   모델과 Adam moments는 유지한다.

`E6_PE_RESAMPLE_EVERY=0`은 main 실험과 같이 한 frozen-policy PDE 안에서
collocation batch를 다시 뽑지 않는다는 뜻이다.

학습과 집계를 분리했다면 다음처럼 그림 옵션까지 명시해 집계한다.

```bash
python3 aggregate_e6.py \
  --out-root outputs/liu_e6_n30_m1 \
  --output outputs/liu_e6_n30_m1/e6_summary \
  --model-type pipinn \
  --expected-n-assets 30 \
  --expected-m-states 1 \
  --expected-seeds '1,2,3,5,7' \
  --expected-targets '0.1,0.05,0.02,0.01,0.005' \
  --min-runs-per-tolerance 5 \
  --metrics e_Xev,RelL2_V,RelL2_theta \
  --require-common-warm-start \
  --expected-e6-reset-lr-each-outer 1 \
  --fig-width 6.4 \
  --fig-height 4.2 \
  --font-size 10 \
  --dpi 300 \
  --formats png,pdf \
  --overwrite
```

두 target만 pilot한 뒤 같은 root와 같은 설정에서 `E6_TARGETS`만
확장할 수 있다. `FORCE_RERUN=1`은 사용하지 않는다. trainer source가
바뀌면 warm-start bundle부터 다시 실행한다.

## 8. E4와 exact-map FD 프로토콜

`liu_exact_map_fd.py` 한 번의 실행은 E4와 exact-map FD 결과를 함께
계산한다.

- E4:

  \[
  e_k^{\mathrm{approx}}
  =
  \|\widetilde v_k-E(\alpha_{k-1})\|_{X_{\mathrm{ev}}}.
  \]

- Exact-map ratio:

  frozen greedy policy를 독립 FD로 평가한 finite-domain map의 오차비이다.

`E4_RATIO_MODE=none`은 FD exact-map 계산 자체를 끄는 옵션이 아니다.
각 seed의 FD 계산 뒤에 ratio 집계·그림을 자동으로 만들지 않는다는
뜻이다.

### 8.1 현재 `run_e4_fd_sweep.sh` 기본값

| 항목 | 기본값 |
|---|---|
| run family | `pres-target` |
| targets | 코드 기본값 `0.2,0.1,0.05,0.02,0.01`; 최종 실험은 `E4_TARGETS=0.1,0.05,0.02,0.01,0.005`로 명시 |
| seeds | `1,11,23,42,101` |
| checkpoints | `all` |
| workers | 5 |
| CPU threads/worker | 10 |
| CPU budget warning | 50 |
| device | `DEVICE_LIST`, 없으면 `cuda:2` |
| eval wealth lower bound | 1.0 |
| eval margin | 0.10 |
| base grid | \(41\times41\times80\) |
| eval grid | \(41\times41\) |
| grid factors | `1,2` |
| factor-domain factors | `1.25,1.50` |
| boundaries | `linearity,exact-dirichlet` |
| refinement rule | `cartesian` |
| min paper checkpoint | 0 |
| policy extension | `boundary-projection` |
| drift scheme | `adaptive` |
| linear residual limit | \(10^{-8}\) |
| boundary condition limit | \(10^{12}\) |

따라서 FD drift의 기본값은 `central`이 아니라 `adaptive`이다. central
실험은 반드시 다음을 명시하고 다른 output root를 사용한다.

```bash
E4_DRIFT_SCHEME="central"
```

### 8.2 Absolute FD wealth interval 조합

다음 설정은 네 개 Cartesian wealth interval을 만들지 않는다.

```bash
E4_FD_W_MINS="0.08,0.05"
E4_FD_W_MAXS="16,32"
```

min과 max는 위치별로 짝지어진다.

```text
[0.08, 16]  # narrow sensitivity interval
[0.05, 32]  # widest primary interval
```

이 두 wealth interval이 `E4_FACTOR_DOMAIN_FACTORS`와
`E4_GRID_FACTORS`에 대해 Cartesian product를 이룬다.

더 큰 wealth-domain sensitivity로 `[0.08,64]`, `[0.05,128]`을
사용하려면 다음처럼 바꾸되 반드시 별도 output root에 저장한다.

```bash
E4_FD_W_MINS="0.08,0.05"
E4_FD_W_MAXS="64,128"
```

### 8.3 Boundary 두 개의 의미

```bash
E4_BOUNDARIES="linearity,exact-dirichlet"
```

는 두 결과를 계산해 더 좋은 값을 고르는 설정이 아니다.

- 첫 번째 `linearity`가 primary BVP이다.
- 모든 grid/domain refinement는 primary boundary에서 판정한다.
- `exact-dirichlet`은 finest grid/largest domain의 matched replacement-BVP
  sensitivity이다.
- boundary replacement는 refinement pass/fail에 들어가지 않고 별도
  표와 그림으로 보고한다.

즉 Figure S1의 main line은 `linearity`를 사용하고,
`exact-dirichlet`은 robustness report이다.

### 8.4 Refinement evidence 규칙

현재 E4 iteration evidence는 Merton식 required set을 사용한다.

- 초기 target
- 첫 adjacent target
- 마지막 adjacent target
- \(e_{\mathrm{approx},X}\)가 가장 큰 worst target

required가 아닌 중간 iteration fail은 CSV에 남지만 required evidence를
자동으로 실패시키지는 않는다. `E4_MIN_PAPER_CHECKPOINT=0`이 기본이므로
checkpoint 1도 제외하지 않는다.

variant rule은 다음 중 선택한다.

```text
cartesian   : primary boundary의 모든 grid/domain interaction 반영, 기본값
merton-axis : one-at-a-time grid/wealth/factor axis 변화만 반영
```

boundary replacement는 어느 rule에서도 report-only이다.

## 9. E4 한 cell pilot

아래는 target 0.05, seed 23, checkpoints 1--2만 보는 간이 테스트다.

```bash
E4_TARGETS="0.05" \
E4_SEEDS="23" \
E4_CHECKPOINTS="1,2" \
E4_RATIO_MODE="none" \
E4_MAX_WORKERS=1 \
E4_CPU_THREADS=10 \
E4_CPU_BUDGET=10 \
E4_DEVICE_LIST="cuda:1" \
E4_EVAL_W_MIN="1.0" \
E4_EVAL_MARGIN="0.10" \
E4_BASE_NY=41 \
E4_BASE_NX=41 \
E4_BASE_NT=80 \
E4_EVAL_NY=41 \
E4_EVAL_NX=41 \
E4_GRID_FACTORS="1,2" \
E4_FD_W_MINS="0.08,0.05" \
E4_FD_W_MAXS="16,32" \
E4_FACTOR_DOMAIN_FACTORS="1.25,1.50" \
E4_BOUNDARIES="linearity,exact-dirichlet" \
E4_REFINEMENT_RULE="cartesian" \
E4_MIN_PAPER_CHECKPOINT=0 \
E4_DRIFT_SCHEME="adaptive" \
E4_POLICY_EXTENSION="boundary-projection" \
E4_LINEAR_RESIDUAL_TOLERANCE="1e-8" \
E4_BOUNDARY_CONDITION_LIMIT="1e12" \
bash run_e4_fd_sweep.sh \
  outputs/pres_5seed \
  outputs/pres_5seed/derived/e4_fd_pilot_w005_s23_ckpt12
```

이 결과는 checkpoint prefix pilot이므로 paper aggregation에 넣지 않는다.
E4 checkpoint subset은 반드시 `1,2,...,k`의 연속 prefix여야 한다.
`1,5,10,15,20`처럼 sparse checkpoint를 보려면 direct driver에서
`--skip-e4`를 사용하는 exact-map-only pilot이어야 한다.

직접 driver를 실행하고 싶다면 동일한 pilot은 다음 형태이다.

```bash
python3 liu_exact_map_fd.py \
  --run-dir outputs/pres_5seed/pi-pinn/pipinn_rho_canonical_v1_m_states1_eval_epochs50000_pres_target0.05_seed23 \
  --output outputs/pres_5seed/derived/e4_fd_pilot/pres_0p05_seed23 \
  --device cuda:1 \
  --checkpoints 1,2 \
  --eval-w-min 1.0 \
  --eval-margin 0.10 \
  --base-ny 41 \
  --base-nx 41 \
  --base-nt 80 \
  --eval-ny 41 \
  --eval-nx 41 \
  --grid-factors 1,2 \
  --fd-w-mins 0.08,0.05 \
  --fd-w-maxs 16,32 \
  --factor-domain-factors 1.25,1.50 \
  --boundaries linearity,exact-dirichlet \
  --verify-checkpoints all \
  --policy-extension boundary-projection \
  --drift-scheme adaptive \
  --refinement-rule cartesian \
  --min-paper-checkpoint 0 \
  --linear-residual-tolerance 1e-8 \
  --boundary-condition-limit 1e12 \
  --overwrite
```

## 10. \(p_{\mathrm{res}}\) E4 full parallel sweep

### 10.1 25 cells를 동시에 실행하는 50-thread 구성

5 targets \(\times\) 5 seeds는 25개의 독립 worker task이다.

```bash
E4_TARGETS="0.1,0.05,0.02,0.01,0.005" \
E4_SEEDS="1,11,23,42,101" \
E4_CHECKPOINTS="all" \
E4_RATIO_MODE="none" \
E4_MAX_WORKERS=25 \
E4_CPU_THREADS=2 \
E4_CPU_BUDGET=50 \
E4_DEVICE_LIST="cuda:1,cuda:2" \
E4_EVAL_W_MIN="1.0" \
E4_EVAL_MARGIN="0.10" \
E4_BASE_NY=41 \
E4_BASE_NX=41 \
E4_BASE_NT=80 \
E4_EVAL_NY=41 \
E4_EVAL_NX=41 \
E4_GRID_FACTORS="1,2" \
E4_FD_W_MINS="0.08,0.05" \
E4_FD_W_MAXS="64,128" \
E4_FACTOR_DOMAIN_FACTORS="1.25,1.50" \
E4_BOUNDARIES="linearity,exact-dirichlet" \
E4_REFINEMENT_RULE="cartesian" \
E4_MIN_PAPER_CHECKPOINT=0 \
E4_DRIFT_SCHEME="adaptive" \
E4_POLICY_EXTENSION="boundary-projection" \
E4_LINEAR_RESIDUAL_TOLERANCE="1e-8" \
E4_BOUNDARY_CONDITION_LIMIT="1e12" \
bash run_e4_fd_sweep.sh \
  outputs/pres_5seed \
  outputs/pres_5seed/derived/e4_fd_sweep_full_w64_128
```

이미 채택한 FD interval이 `[0.08,16]`, `[0.05,32]`라면
`E4_FD_W_MAXS="16,32"`로 바꾸고 이름도
`e4_fd_sweep_full_w16_32`처럼 구분한다.

### 10.2 CPU와 GPU 할당

실제 worker 수는 다음과 같다.

```text
min(E4_MAX_WORKERS, target 수 × seed 수)
```

nominal thread cap은 다음과 같다.

```text
실제 worker 수 × E4_CPU_THREADS
```

`E4_CPU_BUDGET`은 자동 throttle이 아니다. nominal cap이 budget을 넘는지
알리는 warning 기준이다. 실제 동시성을 결정하는 것은 worker 수와
worker당 thread 수다.

64-core 서버에서 약 50 thread를 목표로 할 때 비교 가능한 구성은 다음과
같다.

| 구성 | nominal cap | 특징 |
|---|---:|---|
| 5 workers × 10 threads | 50 | 보수적, worker당 thread가 많음 |
| 10 workers × 5 threads | 50 | 중간 |
| 25 workers × 2 threads | 50 | cell 병렬성이 가장 큼 |

SuperLU가 대체로 single-threaded인 구간에서는 worker 수를 늘리는 편이
더 빠를 수 있다. 다만 각 worker의 RAM과 GPU memory를 확인해야 한다.

GPU는 task 순서대로 round-robin 할당된다.

```text
cuda:1,cuda:2 + 25 tasks
  -> cuda:1에 13개, cuda:2에 12개 정도
```

한 GPU에 여러 worker를 배정할 수 있다. 각 worker 안의 checkpoint,
grid, domain, boundary solve는 직렬이다.

## 11. Figure S1 E4 집계

한 FD root에 pilot target을 추가로 계산해 두었더라도, 최종 그림에서는
기본 다섯 target만 고를 수 있다. 선택은 FD를 다시 계산하지 않는다.

아래 명령은 최종 target 집합을 명시적으로 선택한다.

```bash
python3 aggregate_e4_tolerance.py \
  --out-root outputs/pres_5seed/derived/e4_fd_sweep_full_w64_128 \
  --select-target '0.1,0.05,0.02,0.01,0.005' \
  --select-seeds '1,11,23,42,101' \
  --min-runs-per-tolerance 5 \
  --checkpoints all \
  --refinement-failure-mode report \
  --output outputs/pres_5seed/derived/e4_tolerance_selected5 \
  --plot \
  --plot-metric X \
  --figure-size '6.0,4.0' \
  --font-size 18 \
  --x-tick-count 0 \
  --dpi 300 \
  --formats png,pdf \
  --overwrite
```

`--select-target`은 이름은 단수지만 comma/space-separated 목록을 받고
반복해서 쓸 수도 있다. nominal training target을 선택하며, 그림의 실제
\(x\)좌표는 각 target의 seed-mean achieved \(p_{\mathrm{res}}\)이다.

`--min-runs-per-tolerance 5`는 선택된 target마다 seed run이 최소 5개
있어야 한다는 뜻이다. refinement를 통과한 seed가 반드시 5개여야 한다는
뜻은 아니다.

선택 뒤 target·seed set을 한 번 더 exact assertion으로 고정하고 싶다면
다음을 함께 추가한다.

```text
--expected-tolerances '0.1,0.05,0.02,0.01,0.005'
--expected-seeds '1,11,23,42,101'
```

### 11.1 Refinement failure 처리

strict 집계:

```text
--refinement-failure-mode error
```

- 기본값이다.
- required E4 grid/domain evidence가 fail 또는 incomplete이면 중단한다.

보고형 집계:

```text
--refinement-failure-mode report
```

- 실패 여부와 threshold 대비 크기를 CSV/JSON에 기록한다.
- 표와 그림은 계속 생성한다.
- failure를 pass로 바꾸지는 않는다.
- 주 Figure S1에는 red `x`를 기본적으로 넣지 않는다.

진단 그림에서 문제 target에 red `x`를 표시하려면 다음을 추가한다.

```text
--mark-refinement-issues
```

### 11.2 Figure S1 그림 규칙

기본 `--plot-metric X` 그림은 다음 요소만 포함한다.

- \(x\)축: \(p_{\mathrm{res}}\)
- \(y\)축: \(\widehat p_X\)
- target별 seed mean
- vertical sample-SD error bar
- 점선 \(C_{\mathrm{num}}p_{\mathrm{res}}\) upper envelope
- legend 없음
- log-log scale

다른 하나의 line을 선택할 수 있다.

```text
--plot-metric X
--plot-metric value
--plot-metric bundle
```

현재 조절 가능한 옵션은 다음과 같다.

| 옵션 | 기본값 |
|---|---:|
| `--figure-size` | `4.8,3.4` |
| `--font-size` | 10 |
| `--dpi` | 300 |
| `--formats` | `png` |
| `--x-tick-count` | 0, 모든 target 표시 |
| `--mark-refinement-issues` | off |

현재 코드에 고정된 style은 다음과 같다.

| 항목 | 값 |
|---|---:|
| mean/errorbar linewidth | 1.8 |
| marker | `o`, Matplotlib 기본 크기 |
| errorbar capsize | 2.5 |
| upper-envelope linewidth | 1.0, dashed |
| grid alpha | 0.25 |

\(x\) tick은 소수점 셋째 자리까지 반올림한 뒤 trailing zero를 제거한다.
예를 들어 `0.020`은 `0.02`, `0.100`은 `0.1`로 표시한다. 서로 다른
tick이 같은 문자열이 되거나 양수가 0으로 보일 때만 더 많은 자릿수를
사용한다. 통계 계산과 실제 tick 위치는 반올림하지 않는다.

### 11.3 E4 산출물

```text
e4_tolerance_errors.{png,pdf}
e4_tolerance_per_seed.csv
e4_tolerance_summary.csv
e4_tolerance_aggregate_status.json
e4_boundary_sensitivity_per_checkpoint.csv
e4_boundary_sensitivity_per_seed.csv
e4_boundary_sensitivity_summary.csv
e4_boundary_sensitivity.{png,pdf}
```

`e4_tolerance_errors`는 `linearity` primary만 사용한다. boundary 선택에
따른 변화는 별도 boundary sensitivity 결과에서 확인한다.

## 12. E4 fail 위치 빠르게 확인

exact-map ratio 판단을 제외하고 required E4 evidence만 빠르게 찾으려면
다음 명령으로 각 status file의 핵심 필드를 확인한다.

```bash
find outputs/pres_5seed/derived/e4_fd_sweep_full_w64_128 \
  -name exact_map_status.json -print0 |
  xargs -0 rg -n \
    '"(e4_refinement_evidence_status|e4_refinement_required_statuses|paper_aggregation_eligible|e4_boundary_sensitivity_incomplete_targets)"'
```

모든 primary E4 row의 refinement fail 위치를 target/seed/outer별로
정리하려면 다음 읽기 전용 진단을 사용할 수 있다.

```bash
python3 - <<'PY'
from pathlib import Path
import csv

root = Path("outputs/pres_5seed/derived/e4_fd_sweep_full_w64_128")

for path in sorted(root.glob("pres_*/seed*/e4_approximation_errors.csv")):
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            status = row.get("refinement_status", "")
            if status != "pass":
                print(
                    path.parent.parent.name,
                    path.parent.name,
                    "outer=" + row.get("target_outer_iter", "?"),
                    "status=" + status,
                    "ratio=" + row.get("numerical_tolerance_ratio", ""),
                )
PY
```

이 보고에서 exact-map의 `rho_exact` 또는 final-policy exact ellipticity는
Figure S1 E4 gate와 구분해 해석한다.

## 13. Main \(p_{\mathrm{res}}=\mathrm{None}\) exact-map 병렬 실행

main M=1 run은 target cell이 하나이고 seed 수만큼 task가 생긴다. 다음
예시는 seeds 1,2,3,5,7을 central drift로 실행한다.

```bash
E4_RUN_FAMILY="main" \
E4_MAIN_RUN_STEM="pipinn_m_states1_e3b_checkpoints1" \
E4_MAIN_LABEL="main_m1" \
E4_SEEDS="1,2,3,5,7" \
E4_CHECKPOINTS="all" \
E4_RATIO_MODE="none" \
E4_MAX_WORKERS=5 \
E4_CPU_THREADS=10 \
E4_CPU_BUDGET=50 \
E4_DEVICE_LIST="cuda:1,cuda:2" \
E4_EVAL_W_MIN="1.0" \
E4_EVAL_MARGIN="0.10" \
E4_BASE_NY=41 \
E4_BASE_NX=41 \
E4_BASE_NT=80 \
E4_EVAL_NY=41 \
E4_EVAL_NX=41 \
E4_GRID_FACTORS="1,2" \
E4_FD_W_MINS="0.08,0.05" \
E4_FD_W_MAXS="16,32" \
E4_FACTOR_DOMAIN_FACTORS="1.25,1.50" \
E4_BOUNDARIES="linearity,exact-dirichlet" \
E4_REFINEMENT_RULE="cartesian" \
E4_MIN_PAPER_CHECKPOINT=0 \
E4_DRIFT_SCHEME="central" \
E4_POLICY_EXTENSION="boundary-projection" \
E4_LINEAR_RESIDUAL_TOLERANCE="1e-8" \
E4_BOUNDARY_CONDITION_LIMIT="1e12" \
bash run_e4_fd_sweep.sh \
  outputs/main_10seed_20260723 \
  outputs/main_10seed_20260723/derived/liu_exact_map_main_m1_central
```

이 경우 `E4_TARGETS`는 무시되고 task는 5개이다.

```text
.../liu_exact_map_main_m1_central/main_m1/seed1
.../liu_exact_map_main_m1_central/main_m1/seed2
.../liu_exact_map_main_m1_central/main_m1/seed3
.../liu_exact_map_main_m1_central/main_m1/seed5
.../liu_exact_map_main_m1_central/main_m1/seed7
```

FD job을 먼저 끝내고 ratio는 수동으로 집계하면 strict 실패 기록과
exploratory partial output을 서로 다른 디렉터리에 보존하기 쉽다.

## 14. Empirical \(X_{\mathrm{ev}}\) ratio

Empirical ratio는 두 방식으로 만들 수 있다.

### 14.1 Saved training-history 사용

이 방식은 FD를 전혀 풀지 않고 저장된 `outer_history.csv`의 primary
\(X_{\mathrm{ev}}\) 오차를 사용한다. custom `eval-w-min`을 적용하는
재평가는 아니다.

```bash
E4_RUN_FAMILY="main" \
E4_MAIN_RUN_STEM="pipinn_m_states1_e3b_checkpoints1" \
E4_MAIN_LABEL="main_m1" \
E4_SEEDS="1,2,3,5,7" \
E4_RATIO_MODE="empirical" \
E4_RATIO_OUTPUT_ROOT="outputs/main_10seed_20260723/derived/ratio_outputs" \
E4_EMPIRICAL_FLOOR_MULTIPLIERS="0" \
E4_EMPIRICAL_MAIN_FLOOR_MULTIPLE="0" \
E4_RATIO_Y_SCALE="linear" \
E4_RATIO_FIG_WIDTH="6.4" \
E4_RATIO_FIG_HEIGHT="4.2" \
E4_RATIO_FONT_SIZE="18" \
E4_RATIO_DPI="300" \
E4_RATIO_FORMATS="png,pdf" \
bash run_e4_fd_sweep.sh outputs/main_10seed_20260723
```

`E4_RATIO_MODE=empirical`은 training-history ratio만 만들고 FD task를
실행하지 않는다.

### 14.2 Custom evaluation window checkpoint 재평가

아래 방식은 checkpoint를 neural/autograd로 재평가하지만 FD PDE는 풀지
않는다. exact-map과 같은 \(X_{\mathrm{ev}}\) window를 맞추고 싶을 때
사용한다.

```bash
python3 postprocess_empirical_xev_ratio.py \
  --metric-source checkpoint-reevaluation \
  --out-root outputs/main_10seed_20260723 \
  --output outputs/main_10seed_20260723/derived/empirical_ratio_m1_wmin1 \
  --run-name-regex 'pipinn_m_states1_e3b_checkpoints1_seed(1|11|23|42|101)$' \
  --target-label main_m1 \
  --expected-seeds '1,11,23,42,101' \
  --min-seeds 5 \
  --m-states 1 \
  --n-assets 30 \
  --primary-margin 0.10 \
  --checkpoints all \
  --eval-margin 0.10 \
  --eval-w-min 1.0 \
  --eval-nt 80 \
  --eval-ny 41 \
  --eval-nx 41 \
  --eval-chunk 8192 \
  --device cuda:1 \
  --floor-multipliers 0 \
  --main-floor-multiple 0 \
  --ratio-y-scale linear \
  --fig-width 6.4 \
  --fig-height 4.2 \
  --font-size 18 \
  --line-width 2.0 \
  --marker-size 6.0 \
  --band-alpha 0.18 \
  --floor-alpha 0.80 \
  --grid-alpha 0.22 \
  --dpi 300 \
  --formats png,pdf \
  --bbox-inches tight \
  --overwrite
```

`--eval-w-max`를 생략하면 symmetric margin을 적용한 기존 upper endpoint를
사용한다. 평가구간을 정확히 \([1.0,2.0]\)으로 만들고 싶으면
`--eval-w-max 2.0`을 명시한다. factor margin을 별도로 고정하려면
`--eval-x-margin 0.10`을 추가한다.

현재 empirical-ratio 그림 규칙은 다음과 같다.

- x축 label: `Iteration`
- x ticks: iteration 배열의 매 세 번째 값
- y축/legend: Empirical ratio \(\widehat{\varrho}_n\)
- legend 기본 위치: lower right
- \(y=1\): 검은 dashed contraction threshold
- regular point: 파란 원과 실선
- seed mean \(\pm\) sample SD: `fill_between`
- floor-dominated point: 회색 `x`

조절 가능한 그림 옵션과 기본값은 다음과 같다.

| 옵션 | 기본값 |
|---|---:|
| `--fig-width` | 6.4 |
| `--fig-height` | 4.2 |
| `--font-size` | 10 |
| `--font-family` | Matplotlib default |
| `--line-width` | 2.0 |
| `--marker-size` | 6.0 |
| `--band-alpha` | 0.18 |
| `--floor-alpha` | 0.80 |
| `--grid-alpha` | 0.22 |
| `--dpi` | 300 |
| `--ratio-y-scale` | `linear` |
| `--formats` | `png,pdf` |
| `--bbox-inches` | `tight` |

현재 최신 `postprocess_empirical_xev_ratio.py`는 같은 배포본의
`postprocess_contraction.py`와 함께 사용해야 한다. 두 파일의 API
version을 검사하므로 오래된 한 파일과 최신 한 파일을 섞지 않는다.

## 15. Exact-map ratio 집계와 그림

### 15.1 Strict main exact-map 집계

```bash
python3 aggregate_liu_exact_map.py \
  --out-root outputs/main_10seed_20260723/derived/liu_exact_map_main_m1_central/main_m1 \
  --output outputs/main_10seed_20260723/derived/liu_exact_map_main_m1_central/ratio_outputs/main_m1/exact_audit \
  --expected-seeds '1,2,3,5,7' \
  --min-seeds 5 \
  --plot-ratios \
  --ratio-series exact \
  --floor-multiple 0 \
  --target-label main_m1 \
  --ratio-y-scale log \
  --fig-width 6.5 \
  --fig-height 4.2 \
  --font-size 18 \
  --line-width 2.0 \
  --marker-size 4.0 \
  --band-alpha 0.18 \
  --floor-alpha 0.80 \
  --grid-alpha 0.22 \
  --dpi 300 \
  --formats png,pdf \
  --overwrite
```

### 15.2 Refinement failure를 포함한 exploratory report

strict audit가 실패했지만 실패를 숨기지 않은 상태로 표와 그림을 확인하려면
별도 output에서 `--allow-partial-sensitivity`를 사용한다.

```bash
python3 aggregate_liu_exact_map.py \
  --out-root outputs/main_10seed_20260723/derived/liu_exact_map_main_m1_central/main_m1 \
  --output outputs/main_10seed_20260723/derived/liu_exact_map_main_m1_central/ratio_outputs/main_m1/exact_audit_partial \
  --expected-seeds '1,2,3,5,7' \
  --min-seeds 5 \
  --allow-partial-sensitivity \
  --plot-ratios \
  --ratio-series exact \
  --floor-multiple 0 \
  --target-label main_m1 \
  --ratio-y-scale log \
  --fig-width 6.5 \
  --fig-height 4.2 \
  --font-size 18 \
  --line-width 2.0 \
  --marker-size 4.0 \
  --band-alpha 0.18 \
  --floor-alpha 0.80 \
  --grid-alpha 0.22 \
  --dpi 300 \
  --formats png,pdf \
  --overwrite
```

`--allow-partial-sensitivity`는 실패를 pass로 바꾸지 않는다. CSV와
metadata에 exploratory 상태와 fail row를 유지한다.

### 15.3 \(p_{\mathrm{res}}=0.005\) E4 결과에서 exact ratio

```bash
python3 aggregate_liu_exact_map.py \
  --out-root outputs/pres_5seed/derived/e4_fd_sweep_full_w64_128/pres_0p005 \
  --output outputs/pres_5seed/derived/exact_map_ratio_p005 \
  --expected-seeds '1,11,23,42,101' \
  --min-seeds 5 \
  --allow-partial-sensitivity \
  --plot-ratios \
  --ratio-series exact \
  --floor-multiple 0 \
  --target-label 0.005 \
  --ratio-y-scale log \
  --fig-width 6.5 \
  --fig-height 4.2 \
  --font-size 18 \
  --line-width 2.0 \
  --marker-size 4.0 \
  --band-alpha 0.18 \
  --floor-alpha 0.80 \
  --grid-alpha 0.22 \
  --dpi 300 \
  --formats png,pdf \
  --overwrite
```

모든 strict audit가 통과한 cell이면 `--allow-partial-sensitivity`를
제거한다.

### 15.4 Ratio series 선택

```text
--ratio-series empirical
--ratio-series exact
--ratio-series both
```

- `empirical`: matched custom-\(X_{\mathrm{ev}}\) adjacent learned-step ratio
- `exact`: FD exact-map ratio
- `both`: 두 series 비교
- `--plot-sensitivity-envelope`: FD numerical-sensitivity envelope 추가
- `--floor-multiple 0`: 모든 finite point 유지
- 양수 floor: display-only exploratory floor 분류

exact-map 그림의 조절 가능한 기본값은 다음과 같다.

| 옵션 | 기본값 |
|---|---:|
| `--fig-width` | 6.5 |
| `--fig-height` | 4.2 |
| `--font-size` | 10 |
| `--font-family` | Matplotlib default |
| `--line-width` | 2.0 |
| `--marker-size` | 4.0 |
| `--band-alpha` | 0.18 |
| `--floor-alpha` | 0.80 |
| `--grid-alpha` | 0.22 |
| `--dpi` | 300 |
| `--ratio-y-scale` | `log` |
| `--formats` | `png,pdf` |

`aggregate_liu_exact_map.py`는 항상 `bbox_inches="tight"`를 사용하며
별도 CLI 옵션은 없다.

## 16. Non-affine sweep

Figure 4용 non-affine 실험은 별도 root에서 실행한다.

```bash
SEEDS="1,2,3,5,7,11,17,23,42,101" \
M_STATES_LIST="3" \
EPS_LIST="0,0.1,1,2,3,4,5" \
NONAFFINE_LOADING_SCALE=1.0 \
NONAFFINE_SKIP_FIGURES=1 \
DEVICE_LIST="cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6" \
JOBS_PER_GPU=1 \
bash run_nonaffine_sweep.sh outputs/nonaffine_n30_m3 6
```

- \(N=30\) 고정
- PI-PINN만 실행
- `EPS_LIST`에 paired affine baseline `0`이 반드시 포함
- `M_STATES_LIST`는 1, 3, 5 중 선택
- `NONAFFINE_SKIP_FIGURES=1`은 run별 그림만 끄고 최종 평가는 유지

후처리:

```bash
python3 postprocess_nonaffine.py \
  --out-root outputs/nonaffine_n30_m3 \
  --output outputs/nonaffine_n30_m3/nonaffine_postprocess \
  --n-assets 30 \
  --m-states 3 \
  --expected-seeds '1,2,3,5,7,11,17,23,42,101' \
  --min-seeds 10 \
  --eps '0.1,1,2,3,4,5' \
  --fig-width 13.2 \
  --fig-height 4.0 \
  --value-fig-width 6.0 \
  --policy-fig-width 7.2 \
  --value-fig-height 4.0 \
  --policy-fig-height 4.0 \
  --font-size 22 \
  --x-max-ticks 4 \
  --y-max-ticks 4 \
  --cmap viridis \
  --dpi 300 \
  --formats png,pdf \
  --device cuda:1
```

주요 그림:

```text
nonaffine_figure4.{png,pdf}
V_diff_from_base.{png,pdf}
pi_diff_from_base.{png,pdf}
```

## 17. Timing / E8

1개 seed에서 \(M=1,3,5\), PINN·PI-PINN timing을 모두 측정한다.

```bash
SWEEP_PROFILE=timing \
TIMING_M_STATES_LIST="1,3,5" \
SEEDS="1" \
DEVICE_LIST="cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6" \
JOBS_PER_GPU=1 \
AGGREGATE_COMPUTE=0 \
bash tune_pipinn.sh outputs/liu_timing_seed1 6
```

총 작업은 \(3M\times2\) methods = 6개이다. timing mode는 진단,
outer snapshot, run별 그림을 끄지만 최종 평가는 유지한다.

그림 옵션까지 명시한 수동 집계:

```bash
python3 aggregate_compute.py \
  --out-root outputs/liu_timing_seed1 \
  --output outputs/liu_timing_seed1/compute_summary \
  --models pinn,pipinn \
  --m-states 1,3,5 \
  --n-assets 30 \
  --expected-seeds 1 \
  --min-runs 1 \
  --fig-width 10.5 \
  --fig-height 6.2 \
  --font-size 10 \
  --dpi 300 \
  --formats png,pdf \
  --overwrite
```

공식 training 시간은 CUDA-synchronized `core_train_wall_sec`이다.
`train_wall_sec`은 final checkpoint I/O 전후를 더 넓게 포함한
end-to-end 관측치다. 한 seed에서는 SD와 CI가 `NaN`인 것이 정상이다.

## 18. 그림 옵션 빠른 참조

| 그림 | size 기본값 | font | line | marker | band alpha | grid alpha | y scale |
|---|---|---:|---:|---:|---:|---:|---|
| Main Figure 2 | 4.8 × 3.2 | 10 | 2.2 고정 | 없음 | 0.18 고정 | 0.22 고정 | log |
| E6 error-floor | 6.4 × 4.2 | 10 | 1.7 고정 | 5 고정 | 0.18 고정 | 0.24 고정 | log-log |
| E4 Figure S1 | 4.8 × 3.4 | 10 | 1.8 고정 | `o` 고정 | errorbar | 0.25 고정 | log-log |
| Empirical ratio | 6.4 × 4.2 | 10 | 2.0 | 6.0 | 0.18 | 0.22 | linear |
| Exact-map ratio | 6.5 × 4.2 | 10 | 2.0 | 4.0 | 0.18 | 0.22 | log |
| Non-affine combined | 13.2 × 4.0 | 22 | 고정 | 고정 | 해당 없음 | 코드 style | linear |

Main Figure 2, E6, E4에서 “고정”이라고 쓴 항목은 현재 CLI 옵션으로
변경할 수 없다. Empirical/exact ratio는 명령어에서 line, marker,
alpha까지 조절할 수 있다.

## 19. 최종 논문 산출 전 checklist

### Main

- [ ] affine residual substitution gate 통과
- [ ] 10 seed × 3 \(M\) × 2 methods 성공
- [ ] 동일 canonical market snapshot 확인
- [ ] artifact audit 통과
- [ ] Table 3 strict aggregation 통과
- [ ] Figure 2에 `diag_RelL2_vartheta` 사용
- [ ] Figure 2를 contraction ratio라고 표현하지 않음

### E6

- [ ] independent run과 common-warm-start run을 분리
- [ ] paper E6는 `--require-common-warm-start`
- [ ] target branch LR reset mode 1 확인
- [ ] x축은 achieved post-restore \(p_{\mathrm{res}}\)

### E4 / Figure S1

- [ ] checkpoint schedule `all`
- [ ] `verify-checkpoints=all`
- [ ] `min-paper-checkpoint=0`
- [ ] evaluation window와 FD domain 구분
- [ ] absolute FD min/max는 위치별 paired nested interval
- [ ] drift scheme과 FD bounds가 output root 이름에 드러남
- [ ] `linearity`가 primary boundary
- [ ] `exact-dirichlet`은 report-only boundary sensitivity
- [ ] selected target × selected seed Cartesian panel 완전성 확인
- [ ] refinement report/error mode를 명시
- [ ] exact-map failure로 E4 Figure S1 자체를 잘못 무효화하지 않음

### Ratio diagnostics

- [ ] empirical ratio와 Main Figure 2를 구분
- [ ] exact-map ratio는 finite-domain 진단이라고 표기
- [ ] partial sensitivity 사용 시 exploratory라고 표기
- [ ] `floor-multiple=0`과 양수 floor 결과를 구분
- [ ] evaluation window, checkpoint set, grid가 비교 series 사이에 일치

### 재현성

- [ ] 사용한 seed set 기록
- [ ] training root와 derived output root 기록
- [ ] 실제 command와 log 보존
- [ ] source code hash·config·market provenance 보존
- [ ] PNG뿐 아니라 PDF도 함께 생성
- [ ] figure size, font size, DPI를 최종 명령에 명시

## 20. 도움말

코드가 이후 업데이트되었다면 이 문서보다 현재 parser가 우선이다.

```bash
python3 aggregate_e6.py --help
python3 postprocess_contraction.py --help
python3 postprocess_empirical_xev_ratio.py --help
python3 liu_exact_map_fd.py --help
python3 aggregate_liu_exact_map.py --help
python3 aggregate_e4_tolerance.py --help
python3 postprocess_nonaffine.py --help
python3 aggregate_compute.py --help
```

`run_e4_fd_sweep.sh`의 기본값과 환경변수는 파일 상단 주석과 실행 시작
로그에 모두 출력된다. 실행 직후 표시되는 training root, output root,
targets, seeds, task/worker 수, CPU cap, devices, drift, refinement,
boundary role을 저장해 두는 것이 가장 안전하다.
