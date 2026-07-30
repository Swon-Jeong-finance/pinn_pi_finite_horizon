# Merton 실험 전체 실행 가이드

이 문서는 현재 `merton_ND` 코드로 Merton 본실험, 진단표, welfare,
empirical/exact-map 그림, residual-tolerance 실험, FD 사후평가와
Supplementary Figure 1까지 재현하기 위한 실행 순서를 정리한 runbook이다.

작성 기준:

- Main 결과: `outputs/main_10seed_20260723`
- E6 결과: `outputs/pres_5seed_pilot`
- Main seeds: `1,2,3,5,7,11,17,23,42,101`
- E6 seeds: `1,11,23,42,101`
- Main dimensions: `N=10,50`
- E6 dimension: `N=10`
- Main outer budget: `20`
- 체크포인트: 매 outer 저장, `save_iterate_every=1`
- Main evaluation margin: 첫 값 `0.10`
- E6/FD one-sided evaluation window:
  `eval_margin=0.1`, `eval_w_min=0.5`

---

## 전체 순서

권장 순서는 다음과 같다.

1. 코드와 launcher의 활성 run 목록 점검
2. 짧은 smoke run
3. Main 10-seed × 2 dimensions × 2 methods 학습
4. Main 결과 완전성 검사와 seed aggregation
5. E1 진단표와 E9 nested-window 표
6. Figure 1 PI-PINN control convergence
7. Figure 2 empirical ratio와 선택적 relative-\(L^2\) 그림
8. E2 total-lifetime welfare
9. Main PI-PINN exact-map self-test, smoke, full FD 평가
10. E6 common-warm-start residual-target 학습
11. E6 final-checkpoint one-sided window 재평가와 Figure 3
12. E6 target branches의 full FD/E4 사후평가
13. Supplementary Figure 1 regularity-transfer 집계
14. E8 timing experiment와 compute 집계
15. 최종 paper artifact audit

Main 학습과 E6 학습은 서로 다른 output root에서 수행한다. Timing 실험도
반드시 별도의 output root에 둔다.

---

# Part I. Main Merton 학습

## 1. Launcher의 활성 run 목록 확인

`N_ASSETS_LIST="10,50"`만 설정해도 일반 `tune_merton.sh`가 자동으로
네 종류의 run을 만드는 것은 아니다. 실제 queue는 파일 하단의 활성화된
`run_*` 줄이 결정한다.

확인:

```bash
grep -E '^run_(pinn|pipinn)[[:space:]]' tune_merton.sh
```

본문용 40-job sweep 전에는 다음 네 줄이 모두 활성화되어 있어야 한다.

```bash
run_pinn   n_assets=10
run_pipinn n_assets=10
run_pinn   n_assets=50
run_pipinn n_assets=50
```

정상 queue 크기는 다음과 같다.

```text
10 seeds × 2 dimensions × 2 methods = 40 jobs
```

현재 paper baseline의 핵심 설정은 다음과 같다.

| 항목 | Direct PINN | PI-PINN |
|---|---:|---:|
| learning rate | `3e-4` | `3e-4` |
| outer iterations | `20` | `20` |
| inner/eval epochs | `2000` | `2000` |
| batch size | `10000` | `10000` |
| LR schedule | held-out `Q_sel` plateau | `carry_plateau` |
| checkpoint cadence | every outer | every outer |
| bounds | stabilized, \(\pi_i\in[-2,2]\) | 동일 |
| initial portfolio | 해당 없음 | \(0.5\times\) myopic |
| shape weight | `w_eta=1.5` | `w_eta=20` |
| resampling | `resample_every=200` | `pe_resample_every=0` |

Direct PINN의 Q_sel rollback baseline은 다음과 같다.

```text
qsel_rollback_factor=100
qsel_rollback_lr_factor=1
qsel_rollback_max_rescues=100
```

`qsel_rollback_lr_factor=1`은 rollback 자체가 추가로 LR을 줄이지 않는다는
뜻이다. Model과 Adam state를 마지막 admissible snapshot으로 함께 복원하고,
plateau state와 collocation batch를 초기화한다.

PI-PINN의 초기 portfolio를 exact myopic으로 바꾸려면
`theta_init_scale=1.0`, 완전 unbounded policy 실험은
`policy_bounds_mode=none`을 명시한다. Main baseline은 각각 `0.5`와
`stabilized`이다.

## 2. 선택적 smoke run

본 sweep 전 seed 하나와 각 방법 하나씩 짧게 확인하려면 launcher의
`run_*` block을 임시로 필요한 줄만 활성화하고 다음처럼 override한다.

```bash
SEEDS="1" \
DEVICE_LIST="cuda:1,cuda:2" \
PINN_OVERRIDES="outer_iters=2 eval_epochs=20 test_points=0 n_tau=20 n_x=20" \
PIPINN_OVERRIDES="outer_iters=2 eval_epochs=20 test_points=0 n_tau=20 n_x=20" \
AGGREGATE=0 \
bash tune_merton.sh outputs/merton_smoke 2
```

Smoke root와 paper root를 섞지 않는다.

## 3. Main 10-seed sweep

네 개의 `run_*` 줄을 활성화한 뒤 실행한다.

```bash
SEEDS="1,2,3,5,7,11,17,23,42,101" \
N_ASSETS_LIST="10,50" \
DEVICE_LIST="cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6" \
bash tune_merton.sh outputs/main_10seed_20260723 6
```

마지막 `6`은 outer 수가 아니라 최대 동시 worker 수다.

Launcher 기본값에 이미 `save_iterate_every=1`이 들어 있지만, 실험 계약을
명시적으로 남기려면 다음처럼 실행해도 된다.

```bash
SEEDS="1,2,3,5,7,11,17,23,42,101" \
N_ASSETS_LIST="10,50" \
DEVICE_LIST="cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6" \
PINN_OVERRIDES="save_iterate_every=1" \
PIPINN_OVERRIDES="save_iterate_every=1 e3b_checkpoints=false" \
bash tune_merton.sh outputs/main_10seed_20260723 6
```

`e3b_checkpoints=true`는 과거의 sparse schedule이다. Full E4를 위해 모든
outer를 저장할 때는 사용하지 않는다.

### 재개와 재실행

- `_SUCCESS`만 있는 run은 자동 skip한다.
- 실패, 중단, marker가 없는 partial run은 다음 실행에서 다시 수행한다.
- 성공한 run까지 다시 돌리려면 `FORCE_RERUN=1`을 사용한다.
- 하나라도 실패하면 launcher의 자동 aggregation은 실행되지 않는다.
- 자동 aggregation을 생략하려면 `AGGREGATE=0`을 사용한다.

```bash
FORCE_RERUN=1 \
SEEDS="1,2" \
DEVICE_LIST="cuda:1,cuda:2" \
AGGREGATE=0 \
bash tune_merton.sh outputs/main_10seed_20260723 2
```

## 4. Main 결과 완전성 및 seed aggregation

엄격한 집계:

```bash
python3 aggregate_seeds.py \
  --out-root outputs/main_10seed_20260723 \
  --expected-seeds '1,2,3,5,7,11,17,23,42,101' \
  --expected-n-assets '10,50' \
  --expected-m-states 1 \
  --expected-models 'pinn,pipinn' \
  --strict-market-snapshots
```

기본 결과:

```text
outputs/main_10seed_20260723/seed_summary/
```

주요 파일:

- `runs_index.csv`
- `groups.json`
- `market_hashes.csv`
- `summary_long.csv`
- `summary_headline.csv`
- `summary_e9.csv`
- `success_rates.csv`

### 재시작 과정에서 config hash가 나뉜 경우

실제 scientific setting은 같지만 재시작 때문에 group hash만 나뉜 경우에만
명시적으로 병합한다.

```bash
python3 aggregate_seeds.py \
  --out-root outputs/main_10seed_20260723 \
  --expected-seeds '1,2,3,5,7,11,17,23,42,101' \
  --expected-n-assets '10,50' \
  --expected-m-states 1 \
  --expected-models 'pinn,pipinn' \
  --strict-market-snapshots \
  --merge-config-groups
```

이 옵션은 hash를 수정하지 않는다. `(method, N, m_states)` cell 안에서
집계용으로 합칠 뿐이며 원래 group 정보는 audit 파일에 남는다. 실제
hyperparameter가 다른 run을 합치는 용도로 사용하면 안 된다.

---

# Part II. Main diagnostics와 paper figures

## 5. E1 진단표

E1은 final-only `metrics.csv`가 아니라 각 run의 `outer_history.csv`에서
집계한다. Outer 1은 초기화 burn-in의 영향이 있으므로 전 구간과
`outer>=2` 구간을 모두 만든다.

전 구간:

```bash
python3 aggregate_diagnostics.py \
  --out-root outputs/main_10seed_20260723 \
  --output outputs/main_10seed_20260723/derived/e1_diagnostics \
  --expected-seeds '1,2,3,5,7,11,17,23,42,101' \
  --expected-n-assets '10,50' \
  --expected-models 'pinn,pipinn' \
  --min-seeds 10 \
  --outer-min 1
```

Burn-in 제외:

```bash
python3 aggregate_diagnostics.py \
  --out-root outputs/main_10seed_20260723 \
  --output outputs/main_10seed_20260723/derived/e1_diagnostics \
  --expected-seeds '1,2,3,5,7,11,17,23,42,101' \
  --expected-n-assets '10,50' \
  --expected-models 'pinn,pipinn' \
  --min-seeds 10 \
  --outer-min 2
```

파일명에는 각각 `outer_ge_1`, `outer_ge_2`가 들어가므로 같은 output에
공존할 수 있다.

해석:

- `outer_min=1`: supplement의 “across iterations” 계약
- `outer_min=2`: margin 조건과 guard/clip inactivity를 보는 post-burn-in 구간
- seed 내부에서 먼저 outer extreme을 계산한 뒤 seed 간 mean, sample SD,
  CI를 계산한다.
- Merton의 frozen state covariance는 1차원이므로
  \(\pi^\top\Sigma\pi\)가 유일한 eigenvalue다.

## 6. E9 nested-window value/bundle/control 표

현재 배포 코드에는 별도 `evaluate_margin_bundle.py`가 없다. 훈련 중 각
`eval_margin`에서 이미 생성된 `metrics.csv`를 집계한다.

```bash
python3 aggregate_seeds.py \
  --out-root outputs/main_10seed_20260723 \
  --output outputs/main_10seed_20260723/seed_summary \
  --expected-seeds '1,2,3,5,7,11,17,23,42,101' \
  --expected-n-assets '10,50' \
  --expected-m-states 1 \
  --expected-models 'pinn,pipinn' \
  --strict-market-snapshots \
  --merge-config-groups \
  --e9-margins '0.05,0.10,0.20,0.30'
```

결과는 `seed_summary/summary_e9.csv`다. `--merge-config-groups`는 실제로
split group recovery가 필요할 때만 남긴다.

Merton의 derivative bundle은 wealth-coordinate
\((V_w,V_{ww})\)이며, exogenous factor가 없으므로 \(V_{wx}\)는 없다.

## 7. Figure 1: PI-PINN control convergence

원고의 \(N=50\) 예시:

```bash
python3 postprocess_pipinn_figure1.py \
  --out-root outputs/main_10seed_20260723 \
  --output outputs/main_10seed_20260723/derived/figure1_pipinn_N50 \
  --n-assets 50 \
  --outer-iters 20 \
  --expected-seeds '1,2,3,5,7,11,17,23,42,101' \
  --min-seeds 10 \
  --primary-margin 0.10 \
  --formats png,eps \
  --dpi 300 \
  --fig-width 6.0 \
  --fig-height 4.0 \
  --font-size 22 \
  --line-width 1.5 \
  --line-alpha 1.0 \
  --bbox-inches tight \
  --overwrite
```

별도 파일 두 개가 생성된다.

- `figure1_pipinn_control_convergence_cf.{png,eps}`
- `figure1_pipinn_control_convergence_diff.{png,eps}`

각 outer의 seed mean과 \(\pm1\) sample SD를 표시한다. 제목과 y-label은
원고 caption에서 설명하므로 의도적으로 생략되어 있다.

Figure 1 style 옵션:

| 옵션 | 기본값 |
|---|---:|
| `--formats` | `png,pdf` |
| `--dpi` | `300` |
| `--fig-width` | `6.0` |
| `--fig-height` | `4.0` |
| `--font-size` | `22` |
| `--font-family` | Matplotlib 기본 |
| `--line-width` | `1.5` |
| `--line-alpha` | `1.0` |
| `--bbox-inches` | `tight` |

## 8. Figure 2A: empirical ratio

현재 선택한 본문 그림은 floor filtering 없이 모든 finite adjacent ratio를
표시한다.

```bash
python3 postprocess_contraction.py \
  --out-root outputs/main_10seed_20260723 \
  --output outputs/main_10seed_20260723/derived/figure2_empirical_ratio_N10 \
  --figure-mode empirical-ratio \
  --n-assets 10 \
  --outer-iters 20 \
  --expected-seeds '1,2,3,5,7,11,17,23,42,101' \
  --min-seeds 10 \
  --primary-margin 0.10 \
  --endpoint-outer 20 \
  --floor-multipliers '0' \
  --main-floor-multiple 0 \
  --ratio-y-scale linear \
  --iteration-tick-step 3 \
  --formats png,eps \
  --dpi 300 \
  --fig-width 6.1 \
  --fig-height 4.0 \
  --font-size 18 \
  --line-width 1.8 \
  --marker-size 4.0 \
  --band-alpha 0.18 \
  --overwrite
```

X ticks는 `0,3,6,9,12,15,18`이며, y-label은
\(\widehat{\varrho}_n\)을 사용한다. Legend는 plot 내부 `lower right`다.
Seed trajectory는 기본적으로 숨겨지고 `--show-seed-trajectories`를
명시할 때만 표시된다.

N=50 그림:

```bash
python3 postprocess_contraction.py \
  --out-root outputs/main_10seed_20260723 \
  --output outputs/main_10seed_20260723/derived/figure2_empirical_ratio_N50 \
  --figure-mode empirical-ratio \
  --n-assets 50 \
  --outer-iters 20 \
  --expected-seeds '1,2,3,5,7,11,17,23,42,101' \
  --min-seeds 10 \
  --primary-margin 0.10 \
  --endpoint-outer 20 \
  --floor-multipliers '0' \
  --main-floor-multiple 0 \
  --ratio-y-scale linear \
  --iteration-tick-step 3 \
  --formats png,eps \
  --dpi 300 \
  --fig-width 6.1 \
  --fig-height 4.0 \
  --font-size 18 \
  --line-width 1.8 \
  --marker-size 4.0 \
  --band-alpha 0.18 \
  --overwrite
```

Empirical-ratio style 옵션:

| 옵션 | 기본값 |
|---|---:|
| `--fig-width`, `--fig-height` | `6.0`, `4.0` |
| `--font-size` | `12` |
| `--line-width` | `1.8` |
| `--marker-size` | `4.0` |
| `--band-alpha` | `0.18` |
| `--seed-line-width` | `0.8` |
| `--seed-alpha` | `0.22` |
| `--iteration-tick-step` | `3` |
| `--ratio-y-scale` | `linear` |
| `--bbox-inches` | `tight` |

현재 legend 글씨는 `0.8 × --font-size`다.

## 9. 선택적 Figure 2B: relative-\(L^2\) convergence

Empirical ratio 대신 relative-\(L^2\)를 점검하려면:

```bash
python3 postprocess_contraction.py \
  --out-root outputs/main_10seed_20260723 \
  --output outputs/main_10seed_20260723/derived/figure2_relative_l2_N10 \
  --figure-mode relative-l2 \
  --n-assets 10 \
  --outer-iters 20 \
  --expected-seeds '1,2,3,5,7,11,17,23,42,101' \
  --min-seeds 10 \
  --endpoint-outer 20 \
  --policy-curve rms \
  --formats png,eps \
  --dpi 300 \
  --fig-width 6.0 \
  --fig-height 4.0 \
  --font-size 12 \
  --line-width 1.8 \
  --band-alpha 0.18 \
  --overwrite
```

Empirical ratio와 FD exact-map ratio는 서로 다른 양이다.

- empirical:
  \(e_{n+1}/e_n\), 두 learned iterates의 오차 궤적
- exact-map:
  \(\|E(G(\widetilde v_n))-V^*\|_{X_{\rm ev}}/
  \|\widetilde v_n-V^*\|_{X_{\rm ev}}\), frozen policy PDE의 독립 FD 풀이

## 10. E2 total-lifetime welfare

Merton welfare는 intermediate consumption과 terminal bequest를 모두 포함한
discounted lifetime objective를 사용한다.

먼저 checkpoint, market, seed contract만 검증한다.

```bash
python3 evaluate_welfare.py \
  --out-root outputs/main_10seed_20260723 \
  --output outputs/main_10seed_20260723/derived/e2_welfare \
  --models both \
  --n-assets '10,50' \
  --outer-iters 20 \
  --expected-seeds '1,2,3,5,7,11,17,23,42,101' \
  --min-seeds 10 \
  --n-paths 100000 \
  --n-steps 1000 \
  --w0 0.5 \
  --mc-seed 2718 \
  --validate-only
```

본 계산:

```bash
python3 evaluate_welfare.py \
  --out-root outputs/main_10seed_20260723 \
  --output outputs/main_10seed_20260723/derived/e2_welfare \
  --models both \
  --n-assets '10,50' \
  --outer-iters 20 \
  --expected-seeds '1,2,3,5,7,11,17,23,42,101' \
  --min-seeds 10 \
  --n-paths 100000 \
  --n-steps 1000 \
  --w0 0.5 \
  --mc-seed 2718 \
  --path-batch 4096 \
  --policy-chunk 4096 \
  --device cuda:0
```

Common random numbers를 사용하며 optimal-policy Monte Carlo objective를
같은 discretization의 denominator로 사용한다.

주요 결과:

- `welfare_metrics.csv`
- `welfare_seed_summary.csv`
- `welfare_validation.csv`
- `welfare_config.json`
- `optimal_paths_N*.npz`

---

# Part III. Main exact-map FD 사후평가

## 11. Fixed production FD protocol

현재 고정한 production protocol:

| 항목 | 값 |
|---|---|
| \(Q_{\rm ev}\) | `eval_margin=0.1`, `eval_w_min=0.5` |
| effective wealth range | 약 `[0.5, 1.721783]` |
| base grid | `base_ny=401`, `base_nt=400` |
| evaluation grid | `eval_ny=401` |
| refinement | `grid_factors=1,2` |
| enlarged domains | `fd_margins=-0.5,-1.0` |
| domain factors | `1.5,2.0` |
| primary boundary | `robin` |
| sensitivity boundary | `exact-dirichlet` |
| drift | `central` |
| policy extension | `boundary-projection` |
| checkpoint verification | `all` |
| paper plot floor | `0` |
| linear residual tolerance | `1e-8` |
| refinement tolerances | abs `1e-2`, relative `2e-2` |

CLI의 `robin`은 normalized CRRA value에 대한 homogeneous linear/Robin
closure이며 paper-primary “linearity closure”에 해당한다.
`exact-dirichlet`은 boundary sensitivity audit일 뿐 exact whole-space
oracle이 아니다.

최근 production 결과는 `eval_ny=401`을 사용했다. 코드 기본값 201에
맡기면 protocol hash가 달라지므로 반드시 명시한다.

## 12. FD self-test

```bash
python3 merton_exact_map_fd.py --self-test
```

이 단계는 GPU가 필요하지 않다.

## 13. Main exact-map smoke test

N=10, seed 1, checkpoints 1–3:

```bash
python3 merton_exact_map_fd.py \
  --run-dir outputs/main_10seed_20260723/pipinn/pipinn_n_assets10_seed1 \
  --output outputs/main_10seed_20260723/derived/exact_map_smoke_N10_seed1 \
  --device cuda:1 \
  --checkpoints 1,2,3 \
  --skip-e4 \
  --eval-margin 0.1 \
  --eval-w-mins '0.5' \
  --base-ny 41 \
  --base-nt 80 \
  --eval-ny 41 \
  --grid-factors 1,2 \
  --fd-margins=-0.5,-1.0 \
  --boundaries robin,exact-dirichlet \
  --verify-checkpoints all \
  --drift-scheme central \
  --policy-extension boundary-projection \
  --solver-ellipticity-tolerance 0 \
  --linear-residual-tolerance 1e-8 \
  --boundary-condition-limit 1e12
```

Sparse `--checkpoints`에는 `--skip-e4`가 필요하다. 이 결과는 smoke
진단용이며 paper E4 aggregation 대상이 아니다.

## 14. Main N=10 full exact-map

```bash
python3 merton_exact_map_fd.py \
  --out-root outputs/main_10seed_20260723 \
  --run-name-regex '^pipinn_n_assets10_seed(1|2|3|5|7|11|17|23|42|101)$' \
  --aggregate-output outputs/main_10seed_20260723/derived/merton_exact_map_N10_wmin05 \
  --device cuda:1 \
  --eval-margin 0.1 \
  --eval-w-mins '0.5' \
  --base-ny 401 \
  --base-nt 400 \
  --eval-ny 401 \
  --grid-factors 1,2 \
  --fd-margins=-0.5,-1.0 \
  --boundaries robin,exact-dirichlet \
  --verify-checkpoints all \
  --drift-scheme central \
  --peclet-limit 1.0 \
  --theta-method 0.5 \
  --rannacher-steps 2 \
  --policy-extension boundary-projection \
  --denominator-tolerance 1e-12 \
  --refinement-abs-tolerance 1e-2 \
  --refinement-rel-tolerance 2e-2 \
  --solver-ellipticity-tolerance 0 \
  --linear-residual-tolerance 1e-8 \
  --boundary-condition-limit 1e12 \
  --expected-seeds '1,2,3,5,7,11,17,23,42,101' \
  --min-seeds 10 \
  --floor-multiple 0 \
  --formats png,eps \
  --dpi 300 \
  --fig-width 6.0 \
  --fig-height 4.0 \
  --font-size 12 \
  --line-width 1.8 \
  --line-alpha 1.0 \
  --marker-size 4.0 \
  --band-alpha 0.18 \
  --iteration-tick-step 3 \
  --grid-alpha 0.3 \
  --floor-alpha 0.8 \
  --bbox-inches tight
```

## 15. Main N=50 full exact-map

```bash
python3 merton_exact_map_fd.py \
  --out-root outputs/main_10seed_20260723 \
  --run-name-regex '^pipinn_n_assets50_seed(1|2|3|5|7|11|17|23|42|101)$' \
  --aggregate-output outputs/main_10seed_20260723/derived/merton_exact_map_N50_wmin05 \
  --device cuda:2 \
  --eval-margin 0.1 \
  --eval-w-mins '0.5' \
  --base-ny 401 \
  --base-nt 400 \
  --eval-ny 401 \
  --grid-factors 1,2 \
  --fd-margins=-0.5,-1.0 \
  --boundaries robin,exact-dirichlet \
  --verify-checkpoints all \
  --drift-scheme central \
  --peclet-limit 1.0 \
  --theta-method 0.5 \
  --rannacher-steps 2 \
  --policy-extension boundary-projection \
  --denominator-tolerance 1e-12 \
  --refinement-abs-tolerance 1e-2 \
  --refinement-rel-tolerance 2e-2 \
  --solver-ellipticity-tolerance 0 \
  --linear-residual-tolerance 1e-8 \
  --boundary-condition-limit 1e12 \
  --expected-seeds '1,2,3,5,7,11,17,23,42,101' \
  --min-seeds 10 \
  --floor-multiple 0 \
  --formats png,eps \
  --dpi 300 \
  --fig-width 6.0 \
  --fig-height 4.0 \
  --font-size 12 \
  --line-width 1.8 \
  --line-alpha 1.0 \
  --marker-size 4.0 \
  --band-alpha 0.18 \
  --iteration-tick-step 3 \
  --grid-alpha 0.3 \
  --floor-alpha 0.8 \
  --bbox-inches tight
```

Exact-map을 5-seed panel로 고정한다면 regex, expected seeds, minimum을 모두
함께 바꿔야 한다.

```text
seed(1|11|23|42|101)
--expected-seeds '1,11,23,42,101'
--min-seeds 5
```

## 16. Exact-map plot 해석과 style

`floor_multiple=0`은 모든 finite ratio를 유지한다. 이때 floor-dominated
marker와 eligibility legend는 표시하지 않는다.

현재 exact-map plot:

- y-axis: logarithmic
- x ticks: `0,3,6,9,12,15,18,...`
- threshold: horizontal \(y=1\)
- legend: plot 내부 upper-right, 약간 아래
- legend anchor: `(0.98, 0.92)`
- y-label: \(\varrho_n^{\rm FD}\)

Style 기본값:

| 옵션 | 기본값 |
|---|---:|
| `--fig-width`, `--fig-height` | `6.0`, `4.0` |
| `--font-size` | `12` |
| `--line-width` | `1.8` |
| `--line-alpha` | `1.0` |
| `--marker-size` | `4.0` |
| `--band-alpha` | `0.18` |
| `--iteration-tick-step` | `3` |
| `--grid-alpha` | `0.3` |
| `--floor-alpha` | `0.8` |
| `--bbox-inches` | `tight` |

현재 legend 글씨는 `0.8 × --font-size`, tick 글씨는
`0.9 × --font-size`다.

## 17. Exact-map 결과 경로와 재집계 주의

`--eval-w-mins '0.5'`의 raw 결과는 각 run 아래에 저장된다.

```text
<run-dir>/exact_map_fd/eval_w_min_0p5/
```

`--floor-multiple`은 FD solve에 영향을 주지 않고 aggregation/plot
filter에만 영향을 준다. Floor 0, 5, 10을 비교하려고 FD를 다시 풀 필요는
없다. Aggregate output만 floor별로 다르게 둔다.

`--aggregate-only --out-root ...`는 root 아래 모든 successful exact-map
결과를 재귀적으로 읽으며 이 단계에서는 `--run-name-regex`가 적용되지
않는다. Smoke, 다른 N, 다른 grid, 다른 eval window가 같은 root에 섞여
있다면 원하는 `--result-dir`을 반복해서 명시하는 것이 가장 안전하다.

`--allow-unverified`는 검증을 수행하는 옵션이 아니다. 검증이 없거나
실패한 checkpoint까지 exploratory plot에 포함시키는 우회 옵션이다.
Paper 결과는 `--verify-checkpoints all`을 사용하고
`--allow-unverified`를 사용하지 않는다.

동일한 per-run exact-map result path에 서로 다른 프로토콜을 동시에
실행하지 않는다.

---

# Part IV. E6 residual-tolerance experiment

## 18. Common-warm-start E6 학습

```bash
SEEDS="1,11,23,42,101" \
N_ASSETS_LIST="10" \
DEVICE_LIST="cuda:1,cuda:2,cuda:3,cuda:4,cuda:5" \
E6_TARGETS="1,0.7,0.5,0.3,0.15" \
E6_OUTER_ITERS=20 \
PIPINN_OVERRIDES="eval_epochs=50000 val_every=5 val_points=10000 val_terminal_points=5000 scheduler_patience=100 scheduler_min_lr=1e-6 carry_lr_min=1e-6 sel_patience=0 pe_resample_every=1000" \
bash tune_merton_e6.sh outputs/pres_5seed_pilot 5
```

마지막 `5`는 worker 수이며 warm-up 횟수가 아니다.

Protocol:

1. 각 `(N, seed)`에 대해 target 1, `outer_iters=1`로 warm-up policy
   evaluation을 정확히 한 번 수행한다.
2. Model, optimizer, RNG bundle을 저장한다.
3. 같은 seed의 모든 target이 동일한 bundle에서 branch한다.
4. 각 branch는 warm-up 이후 20개의 policy evaluation을 수행한다.

Warm-up bundle:

```text
outputs/pres_5seed_pilot/e6_warm_starts/n_assets10/seed*/e6_warm_start.pt
```

E6 branch에서는 매 outer가 `carry_lr_max`에서 시작하고 plateau
best/patience는 새로 초기화된다. Model과 Adam moments는
`adam_reset=keep`에 따라 유지된다.

`achieved_pres`는 nominal target이 아니다. Warm-up을 제외한 target-phase
outer들의 official post-restore fixed-\(Q_{\rm res}\) residual 최댓값이다.
`sel_points`의 \(Q_{\rm sel}\)은 inner checkpoint selection에 쓰이며
\(p_{\rm res}\)를 정의하지 않는다.

`pe_resample_every=1000`은 inner optimizer step 1000회마다 policy-evaluation
training batch를 다시 뽑는다. `0`이면 inner loop 동안 고정한다.

### 기존 E6 root에 target을 추가할 때

- 같은 설정과 같은 target을 재실행하면 successful run은 skip한다.
- `aggregate_e6.py --expected-targets`는 selector 역할도 한다.
- 기존 root에 다른 target이 더 있어도 요청한 subset만 집계할 수 있다.
- `FORCE_RERUN=1`로 warm bundle만 바꾸고 과거 branch를 남기면 bundle SHA가
  섞일 수 있으므로 새 root를 쓰거나 전체 target union을 함께 재실행한다.

## 19. E6 final-checkpoint one-sided window eval-only

재학습 없이 official final checkpoint를
\(w\in[0.5,1.721783\ldots]\)에서 다시 평가한다. Training-identifying
overrides는 원래 학습 때와 정확히 같아야 기존 run tag를 찾는다.

```bash
EVAL_ONLY=1 \
AGGREGATE=1 \
SEEDS="1,11,23,42,101" \
N_ASSETS_LIST="10" \
DEVICE_LIST="cuda:1,cuda:2,cuda:3,cuda:4,cuda:5" \
E6_TARGETS="1,0.7,0.5,0.3,0.15" \
E6_OUTER_ITERS=20 \
E6_EVAL_MARGIN=0.1 \
E6_EVAL_W_MIN=0.5 \
E6_EVAL_TEST_POINTS=0 \
E6_EVAL_N_TAU=100 \
E6_EVAL_N_X=100 \
PIPINN_OVERRIDES="eval_epochs=50000 val_every=5 val_points=10000 val_terminal_points=5000 scheduler_patience=100 scheduler_min_lr=1e-6 carry_lr_min=1e-6 sel_patience=0 pe_resample_every=1000" \
bash tune_merton_e6.sh outputs/pres_5seed_pilot 5
```

이 명령은 optimizer step을 수행하지 않고 \(p_{\rm res}\), target-reached
상태, \(Q_{\rm res}\), \(Q_{\rm sel}\)을 변경하지 않는다.

## 20. E6 Figure 3 집계

```bash
python3 aggregate_e6.py \
  --out-root outputs/pres_5seed_pilot \
  --output outputs/pres_5seed_pilot/derived/e6_summary_wmin05 \
  --expected-seeds '1,11,23,42,101' \
  --expected-n-assets '10' \
  --expected-targets '1,0.7,0.5,0.3,0.15' \
  --outer-iters 20 \
  --metrics 'e_Xev,RelL2_V,RelL2_pi,RelL2_c' \
  --require-common-warm-start \
  --strict-market-snapshots \
  --e-xev-source final-metrics \
  --expected-eval-w-min 0.5 \
  --formats png,eps \
  --dpi 300 \
  --fig-width 4.2 \
  --fig-height 3.65 \
  --font-size 10 \
  --x-tick-count 3 \
  --legend-font-size 7.5 \
  --font-family sans-serif \
  --line-width 1.5 \
  --reference-line-width 1.35 \
  --line-alpha 1.0 \
  --band-alpha 0.55 \
  --grid-alpha 0.24 \
  --legend-location best \
  --portfolio-color '#1F4E79' \
  --consumption-color '#E45756' \
  --reference-color '#333333' \
  --overwrite
```

세 그림이 개별 저장된다.

1. \(e_{X_{\rm ev}}\)
2. value relative-\(L^2\)
3. portfolio와 consumption relative-\(L^2\)를 합친 controls panel

X-axis는 nominal target이 아니라 achieved \(p_{\rm res}\)다. 개별 seed
point와 target-reached styling은 표시하지 않으며, target-level mean과
\(\pm1\) sample-SD whisker를 표시한다.

E6 style 기본값:

| 옵션 | 기본값 |
|---|---:|
| `--fig-width`, `--fig-height` | `4.2`, `3.65` |
| `--font-size` | `10` |
| `--x-tick-count` | `3` |
| `--legend-font-size` | `7.5` |
| `--legend-location` | `best` |
| `--line-width` | `1.5` |
| `--reference-line-width` | `1.35` |
| `--line-alpha` | `1.0` |
| `--band-alpha` | `0.55` |
| `--grid-alpha` | `0.24` |
| portfolio | `#1F4E79` |
| consumption | `#E45756` |
| reference | `#333333` |

세 파일의 크기와 폰트는 독립적으로 바꿀 수 있다.

```bash
--xev-fig-width 6.0 --xev-fig-height 4.0 --xev-font-size 14 \
--value-fig-width 6.0 --value-fig-height 4.0 --value-font-size 14 \
--controls-fig-width 6.0 --controls-fig-height 4.0 --controls-font-size 14
```

`--band-alpha`는 현재 sample-SD whisker opacity를 조절한다.

---

# Part V. E6 exact-map과 Supplementary Figure 1

## 21. E6 target-branch FD smoke

Seed 1, target 0.15, checkpoints 1–3:

```bash
python3 merton_exact_map_fd.py \
  --out-root outputs/pres_5seed_pilot \
  --run-name-regex '^pipinn_.*n_assets10_.*seed1_e6_roletarget_branch_outer_iters20_pres_target0\.15$' \
  --output outputs/pres_5seed_pilot/derived/exact_map_smoke_N10_seed1_target0p15 \
  --device cuda:1 \
  --checkpoints 1,2,3 \
  --skip-e4 \
  --eval-margin 0.1 \
  --eval-w-mins '0.5' \
  --base-ny 41 \
  --base-nt 80 \
  --eval-ny 41 \
  --grid-factors 1,2 \
  --fd-margins=-0.5,-1.0 \
  --boundaries robin,exact-dirichlet \
  --verify-checkpoints all \
  --drift-scheme central \
  --policy-extension boundary-projection \
  --solver-ellipticity-tolerance 0 \
  --linear-residual-tolerance 1e-8 \
  --boundary-condition-limit 1e12
```

E6 branch에서 local checkpoint \(K\)는 source value iterate
\(\widetilde v_K\)다. Standard run의 checkpoint \(K\)가
\(\widetilde v_{K-1}\)인 것과 다르다.

## 22. E6 5-seed × 5-target full FD/E4

Full E4에서는 `--checkpoints`와 `--skip-e4`를 넣지 않는다.

```bash
python3 merton_exact_map_fd.py \
  --out-root outputs/pres_5seed_pilot \
  --run-name-regex '^pipinn_.*n_assets10_.*seed(1|11|23|42|101)_e6_roletarget_branch_outer_iters20_pres_target(1|0\.7|0\.5|0\.3|0\.15)$' \
  --aggregate-output outputs/pres_5seed_pilot/derived/e6_exact_map_wmin05 \
  --device cuda:1 \
  --eval-margin 0.1 \
  --eval-w-mins '0.5' \
  --base-ny 401 \
  --base-nt 400 \
  --eval-ny 401 \
  --grid-factors 1,2 \
  --fd-margins=-0.5,-1.0 \
  --boundaries robin,exact-dirichlet \
  --verify-checkpoints all \
  --drift-scheme central \
  --peclet-limit 1.0 \
  --theta-method 0.5 \
  --rannacher-steps 2 \
  --policy-extension boundary-projection \
  --denominator-tolerance 1e-12 \
  --refinement-abs-tolerance 1e-2 \
  --refinement-rel-tolerance 2e-2 \
  --solver-ellipticity-tolerance 0 \
  --linear-residual-tolerance 1e-8 \
  --boundary-condition-limit 1e12 \
  --expected-seeds '1,11,23,42,101' \
  --min-seeds 5 \
  --floor-multiple 0 \
  --formats png,eps \
  --dpi 300 \
  --fig-width 6.0 \
  --fig-height 4.0 \
  --font-size 12 \
  --line-width 1.8 \
  --line-alpha 1.0 \
  --marker-size 4.0 \
  --band-alpha 0.18 \
  --iteration-tick-step 3
```

이 명령이 자동으로 만드는 exact-map convergence 그림에는 target별로
한 curve가 생기므로 다섯 curve가 보인다. Supplementary Figure 1은 이
그림이 아니라 다음 regularity-transfer 집계에서 생성한다.

## 23. Supplementary Figure 1: regularity transfer

`derived` 아래의 smoke output이 섞이지 않도록 `out-root`를
`outputs/pres_5seed_pilot/pipinn`으로 제한한다.

```bash
python3 postprocess_regularity_transfer.py \
  --out-root outputs/pres_5seed_pilot/pipinn \
  --output outputs/pres_5seed_pilot/derived/supplement_figure1_wmin05 \
  --n-assets 10 \
  --run-name-regex 'e6_roletarget_branch_outer_iters20_pres_target(1|0\.7|0\.5|0\.3|0\.15)$' \
  --expected-seeds '1,11,23,42,101' \
  --min-seeds 5 \
  --formats png,eps \
  --dpi 300 \
  --fig-width 4.8 \
  --fig-height 3.4 \
  --font-size 10 \
  --x-tick-count 3 \
  --failure-mode warn \
  --overwrite
```

이 집계는 저장된 CSV/NPZ/JSON을 읽는 CPU post-processing이며 GPU가
필요하지 않다.

`--failure-mode warn`은 일부 seed/target의 refinement가 실패해도 전체
audit를 저장하고 성공 코드로 끝낸다. `error`는 같은 audit를 먼저 쓴 뒤
nonzero로 종료한다.

주요 결과:

- `regularity_transfer.{png,eps}`
- `regularity_transfer_runs.csv`
- `regularity_transfer_per_target.csv`
- `regularity_transfer_fit.csv`
- `regularity_transfer_refinement_audit.csv`
- `regularity_transfer_boundary_runs.csv`
- `regularity_transfer_boundary_audit.csv`
- `regularity_transfer_boundary_per_target.csv`
- `regularity_transfer_boundary_fit.csv`
- `regularity_transfer_boundary_pairs.csv`
- `regularity_transfer_evidence_manifest.csv`
- `regularity_transfer_status.json`

Supplementary Figure 1은
\(\widehat p_X\) 대 achieved \(p_{\rm res}\)의 log-log plot이다.
Primary Robin result와 exact-Dirichlet sensitivity는 별도로 집계된다.

Regularity-transfer plot 옵션:

| 옵션 | 기본값 |
|---|---:|
| `--formats` | `png,pdf` |
| `--dpi` | `300` |
| `--fig-width` | `4.8` |
| `--fig-height` | `3.4` |
| `--font-size` | `10` |
| `--x-tick-count` | `3` |
| `--failure-mode` | `warn` |

현재 이 script에는 font family, line width, marker size, alpha를 개별적으로
바꾸는 CLI는 없다.

## 24. E4의 refinement/enlargement 문구에 넣을 수치

현재 production command의 수치는 다음과 같다.

```text
grid factors: 1 and 2
domain factors: 1.5 and 2.0
absolute refinement tolerance: 1e-2
relative refinement tolerance: 2e-2
normalized linear solve residual tolerance: 1e-8
```

실제 paper 문구에는 command의 nominal tolerance만 쓰지 말고
`regularity_transfer_refinement_audit.csv`에서 required defects의 pass
상태와 observed grid/domain changes도 함께 확인한다.

Boundary replacement는 grid/domain refinement와 동일한 BVP convergence
gate로 해석하지 않는다. Primary Robin과 exact-Dirichlet에 대해
\(\widehat p_X\), fitted slope, \(C_{\rm num}\)을 별도로 보고
`regularity_transfer_boundary_*` 파일에서 차이를 확인한다.

---

# Part VI. E8 compute experiment

## 25. Timing run

Timing 결과는 Main root와 분리한다. 일반 launcher의 네 개
`run_*` 줄이 모두 활성화되어 있어야 한다.

1-seed timing:

```bash
SEEDS="1" \
N_ASSETS_LIST="10,50" \
DEVICE_LIST="cuda:1,cuda:2,cuda:3,cuda:4" \
PINN_OVERRIDES="timing_mode=true skip_figures=true save_iterate_every=0" \
PIPINN_OVERRIDES="timing_mode=true skip_figures=true save_iterate_every=0 e3b_checkpoints=false" \
AGGREGATE=0 \
bash tune_merton.sh outputs/1seed_timing 4
```

`timing_mode=true`는 불필요한 training diagnostics와 iterate checkpoint
I/O를 제거한다. `skip_figures=true`가 effective config에 있어야 evaluation
peak memory에 plotting이 섞이지 않았음을 `aggregate_compute.py`가
검증할 수 있다.

Timing 필드는 각 run의 `status.json`에 저장된다.

```text
train_wall_sec
total_optimizer_steps
train_gpu_peak_mem_bytes
eval_gpu_peak_mem_bytes
```

## 26. Compute 집계

```bash
python3 aggregate_compute.py \
  --out-root outputs/1seed_timing \
  --expected-seeds '1' \
  --expected-n-assets '10,50' \
  --expected-methods 'pinn,pipinn' \
  --min-seeds 1 \
  --formats png,pdf \
  --overwrite
```

기본 출력:

```text
outputs/1seed_timing/compute_summary/
```

한 seed에서는 sample SD가 `NA`다. Paper에서 seed 변동성까지 보고하려면
동일 GPU model에서 최소 두 seed를 돌린 뒤 `--require-sample-sd`를
추가한다.

---

# Part VII. 최종 paper audit

## 27. 학습 artifact

각 run에 대해 확인:

- terminal marker가 `_SUCCESS` 하나인지
- `status.json`의 status가 success인지
- `value_net_final.pt`가 official model인지
- `value_net_last.pt`와 checkpoint provenance가 일관적인지
- PI-PINN `checkpoint_manifest.json`이 존재하는지
- `save_iterate_every=1`에 맞는 모든 outer checkpoint가 있는지
- `metrics.csv`가 모든 요청 margin을 포함하는지
- `outer_history.csv`가 outer 1–20을 완전히 포함하는지
- Main seeds가 동일한 `market_hash`를 쓰는지

## 28. Main paper outputs

- Main final metrics:
  `outputs/main_10seed_20260723/seed_summary/`
- E1:
  `outputs/main_10seed_20260723/derived/e1_diagnostics/`
- Figure 1:
  `outputs/main_10seed_20260723/derived/figure1_pipinn_N50/`
- Figure 2 empirical:
  `outputs/main_10seed_20260723/derived/figure2_empirical_ratio_N10/`
- Welfare:
  `outputs/main_10seed_20260723/derived/e2_welfare/`
- Exact map:
  `outputs/main_10seed_20260723/derived/merton_exact_map_N*/`
- E9:
  `outputs/main_10seed_20260723/seed_summary/summary_e9.csv`

## 29. E6/E4 outputs

- E6 target/error scaling:
  `outputs/pres_5seed_pilot/derived/e6_summary_wmin05/`
- E6 full exact-map:
  `outputs/pres_5seed_pilot/derived/e6_exact_map_wmin05/`
- Supplementary Figure 1:
  `outputs/pres_5seed_pilot/derived/supplement_figure1_wmin05/`

## 30. Exact-map paper eligibility

Paper exact-map 결과는 다음을 모두 만족해야 한다.

- `--verify-checkpoints all`
- 모든 required refinement status pass
- central drift protocol
- primary `robin`, sensitivity `exact-dirichlet`
- `boundary-projection`
- normalized linear residual \(\le 10^{-8}\)
- boundary elimination full rank와 finite/acceptable condition
- sampled \(\pi^\top\Sigma\pi>0\)
- locally active guard/clipping이 없거나 명시적으로 audit됨
- 동일 \(Q_{\rm ev}\), grid, domain, derivative-coordinate protocol
- `--allow-unverified`를 사용하지 않음

`exact_map_status=success`만으로 paper eligibility가 보장되는 것은 아니다.
Refinement, boundary, linear solve, ellipticity, extension, evaluation-window
provenance를 함께 확인한다.

## 31. 해석 시 반드시 유지할 구분

- `e_Xev`는 relative error가 아니라 value sup와 wealth-derivative bundle
  sup의 합이다.
- E6 x-axis는 nominal target이 아니라 official achieved post-restore
  \(p_{\rm res}\)다.
- Empirical ratio와 FD exact-map ratio는 별도 evidence다.
- `exact-dirichlet`은 whole-space exact boundary가 아니다.
- `floor_multiple=0`은 모든 finite ratio를 표시하지만 검증 실패를
  무시한다는 뜻은 아니다.
- 높은 `eval_w_min=0.5`에서 좋은 결과는 low-wealth tail이
  \(V_{ww}\) absolute error를 지배한다는 sensitivity evidence다. 더 넓은
  원래 window의 contraction을 소급해 증명하지 않는다.

---

# Appendix A. Figure option 요약

## A.1 Figure 1

```text
--formats
--dpi
--fig-width
--fig-height
--font-size
--font-family
--line-width
--line-alpha
--bbox-inches
```

## A.2 Empirical ratio / relative-\(L^2\)

```text
--formats
--dpi
--fig-width
--fig-height
--font-size
--font-family
--line-width
--band-alpha
--seed-line-width
--seed-alpha
--marker-size
--iteration-tick-step
--ratio-y-scale
--bbox-inches
```

## A.3 Exact-map

```text
--formats
--dpi
--fig-width
--fig-height
--font-size
--font-family
--line-width
--line-alpha
--marker-size
--band-alpha
--iteration-tick-step
--grid-alpha
--floor-alpha
--bbox-inches
```

## A.4 E6

```text
--fig-width --fig-height --font-size
--xev-fig-width --xev-fig-height --xev-font-size
--value-fig-width --value-fig-height --value-font-size
--controls-fig-width --controls-fig-height --controls-font-size
--x-tick-count
--font-family
--legend-font-size
--legend-location
--line-width
--reference-line-width
--line-alpha
--band-alpha
--grid-alpha
--xev-color
--value-color
--portfolio-color
--consumption-color
--reference-color
--formats
--dpi
```

## A.5 Supplementary Figure 1

```text
--formats
--dpi
--fig-width
--fig-height
--font-size
--x-tick-count
--failure-mode
```

---

# Appendix B. 자주 발생하는 오류

## B.1 Aggregation에서 seed가 여러 group으로 분리됨

실제 설정이 같은 재시작 결과라면 `--merge-config-groups`를 사용한다.
실제 설정이 다르면 재학습하거나 별도 setting으로 보고한다.

## B.2 Timing aggregation이 `skip_figures=true`를 요구함

Timing run을 다음 effective config로 다시 생성한다.

```text
timing_mode=true
skip_figures=true
save_iterate_every=0
```

## B.3 E6 aggregation에서 root에 다른 targets가 있음

`--expected-targets`에 원하는 subset만 명시한다. 선택된 각 target에
요청한 모든 seed가 있어야 한다.

## B.4 Common warm-start SHA가 targets 사이에서 다름

서로 다른 시점에 생성된 warm bundle이 섞인 것이다. 새 root를 사용하거나
선택한 target union 전체를 같은 warm bundle에서 다시 branch한다.

## B.5 Exact-map aggregation이 broad root의 smoke 결과까지 읽음

`--aggregate-only --out-root`는 모든 successful exact-map directory를
재귀적으로 읽는다. `--result-dir`을 반복해서 paper result만 명시하거나
full evaluator와 aggregation을 한 명령에서 수행한다.

## B.6 Exact-map이 unverified checkpoints를 거부함

Paper 결과라면 `--verify-checkpoints all`로 FD를 다시 수행한다.
`--allow-unverified`는 exploratory plot 전용이다.

## B.7 EPS transparency warning

PostScript backend는 transparency를 직접 지원하지 않는다. 현재 scripts는
가능한 경우 흰 배경과 혼합한 opaque color로 저장한다. PNG가 정상이고
EPS가 생성되었다면 경고 자체는 계산 오류가 아니다.

